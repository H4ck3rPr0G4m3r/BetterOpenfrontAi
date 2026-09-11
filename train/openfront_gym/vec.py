"""Batched OpenFront environments over W worker processes."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space

from . import spaces
from .protocol import TerminalCode
from .worker import WorkerClient, WorkerConfig, connect_worker, find_free_port

_TRUNCATING_CODES = {TerminalCode.TICK_CAP}


class OpenFrontVecEnv(gymnasium.vector.VectorEnv):
    """W workers times M envs behind one batched Gymnasium interface.

    Workers step in parallel threads in phase 6. Each worker owns its own
    socket and sequence numbers, so blocking socket calls run concurrently
    while result assembly stays single-threaded for deterministic row
    order. The Node side already auto-resets finished episodes, so a
    ``done`` row carries the next episode's obs with the finished
    episode's reward and terminal reason, which is the same-step
    autoreset mode below. A dead worker is respawned and its M rows come
    back done with ``terminal_reason == "worker_death"``.
    """

    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(
        self,
        num_workers: int,
        envs_per_worker: int,
        map: str = "World",
        bots: int = 20,
        nations: str = "default",
        ticks_per_step: int = 10,
        tick_cap: int = 6000,
        maps_dir: str | None = None,
        strict_illegal_actions: bool = True,
        socket_timeout: float = 180.0,
        env_root: Path | str | None = None,
        seeds: Sequence[str | int] | None = None,
        worker_factory: Callable[[int], WorkerClient] | None = None,
        disabled_intents: Sequence[int] = (),
        disabled_builds: Sequence[str] = (),
        reward_weights: Sequence[float] | None = None,
        spawn_timeout: float = 120.0,
        parallel: bool = True,
    ):
        if num_workers < 1 or envs_per_worker < 1:
            raise ValueError("num_workers and envs_per_worker must be positive")
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        num_envs = num_workers * envs_per_worker
        self._config = WorkerConfig(
            map=map,
            bots=bots,
            nations=nations,
            ticks_per_step=ticks_per_step,
            tick_cap=tick_cap,
            maps_dir=maps_dir,
            num_envs=envs_per_worker,
            strict_illegal_actions=strict_illegal_actions,
            socket_timeout=socket_timeout,
            disabled_intents=tuple(disabled_intents),
            disabled_builds=tuple(disabled_builds),
            reward_weights=tuple(reward_weights) if reward_weights is not None else None,
            spawn_timeout=float(spawn_timeout),
        )
        self._env_root = Path(env_root) if env_root is not None else None
        self._worker_factory = worker_factory
        self._base_seeds = self._initial_seeds(seeds, num_envs)
        self._clients: list[WorkerClient | None] = [None] * num_workers
        self._client_lock = threading.Lock()
        self._parallel = bool(parallel) and num_workers > 1
        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="openfront-step")
            if self._parallel
            else None
        )
        self._masks = np.ones((num_envs, spaces.MASK_TOTAL), dtype=bool)
        self._ticks = np.zeros(num_envs, dtype=np.int64)
        self._obs_size = spaces.OBSERVATION_SIZE
        for worker_index in range(num_workers):
            self._clients[worker_index] = self._spawn(worker_index)
        client = self._clients[0]
        assert client is not None
        self._obs_size = client.obs_size
        self.single_observation_space = spaces.make_observation_space(self._obs_size)
        self.single_action_space = spaces.make_action_space()
        self.observation_space = batch_space(self.single_observation_space, n=num_envs)
        self.action_space = batch_space(self.single_action_space, n=num_envs)
        self.reset()

    @property
    def num_envs(self) -> int:
        return self.num_workers * self.envs_per_worker

    @staticmethod
    def _initial_seeds(seeds: Sequence[str | int] | None, num_envs: int) -> list[str]:
        if seeds is not None:
            seed_list = [str(seed) for seed in seeds]
            if len(seed_list) != num_envs:
                raise ValueError(f"need {num_envs} seeds, got {len(seed_list)}")
            return seed_list
        suffix = int(np.random.SeedSequence().entropy % 10**8)
        return [f"openfront-{suffix}-{i}" for i in range(num_envs)]

    def _spawn(self, worker_index: int) -> WorkerClient:
        if self._worker_factory is not None:
            return self._worker_factory(worker_index)
        port = find_free_port()
        return connect_worker(self._config, port, self._env_root)

    def _worker_seeds(self, worker_index: int) -> list[str]:
        start = worker_index * self.envs_per_worker
        return self._base_seeds[start : start + self.envs_per_worker]

    def reset(self, *, seed=None, options=None):
        # The Gymnasium base only takes int seeds. Lists of per-env seeds
        # are this wrapper's own feature, normalized below, so they must
        # not reach the base call.
        super().reset(seed=seed if isinstance(seed, int) or seed is None else None)
        if seed is not None:
            if isinstance(seed, int):
                self._base_seeds = [f"{seed}-{i}" for i in range(self.num_envs)]
            else:
                given = [str(value) for value in seed]
                if len(given) != self.num_envs:
                    raise ValueError(f"need {self.num_envs} seeds, got {len(given)}")
                self._base_seeds = given
        obs = np.zeros((self.num_envs, self._obs_size), dtype=np.float32)
        if self._pool is None:
            for worker_index in range(self.num_workers):
                rows = self._rows(worker_index)
                client = self._require_client(worker_index)
                obs_list, ticks, masks = client.reset(self._worker_seeds(worker_index))
                obs[rows] = np.asarray(obs_list, dtype=np.float32)
                self._ticks[rows] = np.asarray(ticks, dtype=np.int64)
                self._masks[rows] = np.stack([spaces.mask_bytes_to_array(m) for m in masks])
        else:
            # Only the blocking socket call runs in threads. Results are
            # fully collected before assembly so a failing worker can never
            # leave a sibling socket busy when the error propagates.
            def _reset_one(worker_index: int):
                try:
                    return worker_index, self._snapshot_client(worker_index).reset(
                        self._worker_seeds(worker_index)
                    ), None
                except Exception as exc:  # noqa: BLE001, collected below
                    return worker_index, None, exc

            completed = list(self._pool.map(_reset_one, range(self.num_workers)))
            for worker_index, result, exc in completed:
                if exc is not None or result is None:
                    raise exc
                rows = self._rows(worker_index)
                obs_list, ticks, masks = result
                obs[rows] = np.asarray(obs_list, dtype=np.float32)
                self._ticks[rows] = np.asarray(ticks, dtype=np.int64)
                self._masks[rows] = np.stack([spaces.mask_bytes_to_array(m) for m in masks])
        return obs, self._infos([None] * self.num_envs)

    def step(self, actions):
        actions = np.asarray(actions).reshape(self.num_envs, len(spaces.HEAD_NAMES))
        obs = np.zeros((self.num_envs, self._obs_size), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        reasons: list[str | None] = [None] * self.num_envs
        if self._pool is None:
            for worker_index in range(self.num_workers):
                rows = self._rows(worker_index)
                try:
                    client = self._require_client(worker_index)
                    result = client.step(
                        [spaces.action_array_to_dict(actions[i]) for i in rows],
                        [""] * self.envs_per_worker,
                    )
                except (ConnectionError, OSError):
                    self._respawn(worker_index, obs, rewards, terminated, reasons)
                    continue
                self._assemble_step(worker_index, rows, result, obs, rewards, terminated, truncated, reasons)
        else:
            # Fan the blocking step calls out to threads, collect every
            # result first, then assemble and respawn sequentially in index
            # order. Helpers never raise: a fatal error is re-raised only
            # after all sockets are idle, so a retry can never issue a
            # second request on a socket that is still busy.
            def _step_one(worker_index: int):
                rows = self._rows(worker_index)
                try:
                    result = self._snapshot_client(worker_index).step(
                        [spaces.action_array_to_dict(actions[i]) for i in rows],
                        [""] * self.envs_per_worker,
                    )
                except (ConnectionError, OSError) as exc:
                    return worker_index, None, exc, True
                except Exception as exc:  # noqa: BLE001, re-raised below
                    return worker_index, None, exc, False
                return worker_index, result, None, False

            completed = list(self._pool.map(_step_one, range(self.num_workers)))
            for worker_index, result, exc, respawnable in completed:
                rows = self._rows(worker_index)
                if exc is not None:
                    if respawnable:
                        self._respawn(worker_index, obs, rewards, terminated, reasons)
                        continue
                    raise exc
                if result is None:
                    raise RuntimeError(f"worker {worker_index} returned no result")
                self._assemble_step(worker_index, rows, result, obs, rewards, terminated, truncated, reasons)
        return obs, rewards, terminated, truncated, self._infos(reasons)

    def _assemble_step(self, worker_index: int, rows, result, obs, rewards, terminated, truncated, reasons) -> None:
        obs_list, reward_list, done, terminal_codes, ticks, masks = result
        obs[rows] = np.asarray(obs_list, dtype=np.float32)
        rewards[rows] = np.asarray(reward_list, dtype=np.float32)
        self._ticks[rows] = np.asarray(ticks, dtype=np.int64)
        self._masks[rows] = np.stack([spaces.mask_bytes_to_array(m) for m in masks])
        for row, is_done, code in zip(rows, done, terminal_codes):
            if not is_done:
                continue
            terminated[row] = code not in _TRUNCATING_CODES
            truncated[row] = code in _TRUNCATING_CODES
            reasons[row] = spaces.TERMINAL_NAMES.get(code)

    def _respawn(self, worker_index: int, obs, rewards, terminated, reasons) -> None:
        """Replace a dead worker, reset its envs, mark its rows done."""
        warnings.warn(f"worker {worker_index} died, respawning", RuntimeWarning, stacklevel=3)
        with self._client_lock:
            old = self._clients[worker_index]
            self._clients[worker_index] = None
        if old is not None:
            try:
                old.close()
            except OSError:
                pass
        rows = self._rows(worker_index)
        client = self._spawn(worker_index)
        with self._client_lock:
            current = self._clients[worker_index]
            if current is not None and current is not client and not current.closed:
                # A concurrent lookup replaced the slot while we spawned.
                # Our reset data came from the new client, so keep ours and
                # drop the orphan instead of leaking it.
                try:
                    current.close()
                except OSError:
                    pass
            self._clients[worker_index] = client
        obs_list, ticks, masks = client.reset(self._worker_seeds(worker_index))
        obs[rows] = np.asarray(obs_list, dtype=np.float32)
        rewards[rows] = 0.0
        terminated[rows] = True
        self._ticks[rows] = np.asarray(ticks, dtype=np.int64)
        self._masks[rows] = np.stack([spaces.mask_bytes_to_array(m) for m in masks])
        for row in rows:
            reasons[row] = spaces.WORKER_DEATH_REASON

    def _require_client(self, worker_index: int) -> WorkerClient:
        # Fast path under a short lock; the slow spawn and connect happen
        # outside it so one dead worker cannot stall healthy workers.
        with self._client_lock:
            client = self._clients[worker_index]
            if client is not None and not client.closed:
                return client
        client = self._spawn(worker_index)
        with self._client_lock:
            current = self._clients[worker_index]
            if current is not None and not current.closed:
                # Another thread already replaced it while we spawned.
                # Prefer the live one and drop ours.
                try:
                    client.close()
                except OSError:
                    pass
                return current
            self._clients[worker_index] = client
            return client

    def _snapshot_client(self, worker_index: int) -> WorkerClient:
        # Workers own their socket, so threads can block in parallel. The
        # list lookup is the only shared state, guarded by a short lock.
        with self._client_lock:
            client = self._clients[worker_index]
        if client is None or client.closed:
            return self._require_client(worker_index)
        return client

    def _rows(self, worker_index: int) -> list[int]:
        start = worker_index * self.envs_per_worker
        return list(range(start, start + self.envs_per_worker))

    def _infos(self, reasons: list[str | None]) -> dict:
        return {
            "action_masks": self.action_masks(),
            "terminal_reason": np.asarray(reasons, dtype=object),
            "ticks": self._ticks.copy(),
        }

    def action_masks(self) -> np.ndarray:
        """Stacked flat bool masks, one row per env, in head order."""
        return self._masks.copy()

    def worker_log_paths(self) -> list[str | None]:
        """Stderr log file per worker, for surfacing boot failures."""
        with self._client_lock:
            clients = list(self._clients)
        return [client.log_path if client is not None else None for client in clients]

    def env_method(self, name: str, *args, **kwargs) -> list:
        """Subset of the SB3 VecEnv protocol, enough for mask retrieval."""
        if name == "action_masks":
            return [row.copy() for row in self.action_masks()]
        raise AttributeError(f"unknown env method {name}")

    def has_attr(self, name: str) -> bool:
        return name in ("action_masks", "num_envs")

    def get_attr(self, name: str) -> list:
        if name == "action_masks":
            return self.env_method(name)
        if name == "num_envs":
            return [self.num_envs] * self.num_envs
        raise AttributeError(f"unknown attr {name}")

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)
        with self._client_lock:
            clients = list(self._clients)
            self._clients = [None] * self.num_workers
        for client in clients:
            if client is None:
                continue
            try:
                client.close()
            except OSError:
                pass
        self.closed = True
