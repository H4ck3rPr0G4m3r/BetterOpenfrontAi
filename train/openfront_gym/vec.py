"""Batched OpenFront environments over W worker processes."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
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

    Workers step sequentially in phase 1. The Node side already auto-resets
    finished episodes, so a ``done`` row carries the next episode's obs
    with the finished episode's reward and terminal reason, which is the
    same-step autoreset mode below. A dead worker is respawned and its M
    rows come back done with ``terminal_reason == "worker_death"``.
    """

    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(
        self,
        num_workers: int,
        envs_per_worker: int,
        map: str = "World",
        bots: int = 20,
        ticks_per_step: int = 10,
        tick_cap: int = 6000,
        maps_dir: str | None = None,
        strict_illegal_actions: bool = True,
        socket_timeout: float = 180.0,
        env_root: Path | str | None = None,
        seeds: Sequence[str | int] | None = None,
        worker_factory: Callable[[int], WorkerClient] | None = None,
    ):
        if num_workers < 1 or envs_per_worker < 1:
            raise ValueError("num_workers and envs_per_worker must be positive")
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        num_envs = num_workers * envs_per_worker
        self._config = WorkerConfig(
            map=map,
            bots=bots,
            ticks_per_step=ticks_per_step,
            tick_cap=tick_cap,
            maps_dir=maps_dir,
            num_envs=envs_per_worker,
            strict_illegal_actions=strict_illegal_actions,
            socket_timeout=socket_timeout,
        )
        self._env_root = Path(env_root) if env_root is not None else None
        self._worker_factory = worker_factory
        self._base_seeds = self._initial_seeds(seeds, num_envs)
        self._clients: list[WorkerClient | None] = [None] * num_workers
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
        super().reset(seed=seed)
        if seed is not None:
            if isinstance(seed, int):
                self._base_seeds = [f"{seed}-{i}" for i in range(self.num_envs)]
            else:
                given = [str(value) for value in seed]
                if len(given) != self.num_envs:
                    raise ValueError(f"need {self.num_envs} seeds, got {len(given)}")
                self._base_seeds = given
        obs = np.zeros((self.num_envs, self._obs_size), dtype=np.float32)
        for worker_index in range(self.num_workers):
            rows = self._rows(worker_index)
            client = self._require_client(worker_index)
            obs_list, ticks, masks = client.reset(self._worker_seeds(worker_index))
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
        return obs, rewards, terminated, truncated, self._infos(reasons)

    def _respawn(self, worker_index: int, obs, rewards, terminated, reasons) -> None:
        """Replace a dead worker, reset its envs, mark its rows done."""
        warnings.warn(f"worker {worker_index} died, respawning", RuntimeWarning, stacklevel=3)
        old = self._clients[worker_index]
        self._clients[worker_index] = None
        if old is not None:
            try:
                old.close()
            except OSError:
                pass
        rows = self._rows(worker_index)
        client = self._spawn(worker_index)
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
        client = self._clients[worker_index]
        if client is None or client.closed:
            client = self._spawn(worker_index)
            self._clients[worker_index] = client
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
            return [self.num_envs] * self.num_workers
        raise AttributeError(f"unknown attr {name}")

    def close(self) -> None:
        for worker_index, client in enumerate(self._clients):
            self._clients[worker_index] = None
            if client is not None:
                try:
                    client.close()
                except OSError:
                    pass
        self.closed = True
