"""Single OpenFront environment behind the standard Gymnasium API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gymnasium
import numpy as np

from . import spaces
from .protocol import ProtocolError, TerminalCode
from .worker import WorkerClient, WorkerConfig, connect_worker, find_free_port

_TRUNCATING_CODES = {TerminalCode.TICK_CAP}


class OpenFrontEnv(gymnasium.Env):
    """One agent seat in a headless OpenFront game.

    The constructor only records settings. The worker process boots on the
    first ``reset`` so importing and building the env stay cheap. Actions
    are five integers in :data:`spaces.HEAD_NAMES` order, matching the
    ``MultiDiscrete`` action space. Masks come back in ``info`` under
    ``"action_masks"`` and from :meth:`action_masks`, flat bool vectors the
    MaskablePPO convention reads.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map: str = "World",
        bots: int = 20,
        ticks_per_step: int = 10,
        tick_cap: int = 6000,
        maps_dir: str | None = None,
        strict_illegal_actions: bool = True,
        socket_timeout: float = 180.0,
        env_root: Path | str | None = None,
        worker_factory: Callable[[], WorkerClient] | None = None,
    ):
        super().__init__()
        self._config = WorkerConfig(
            map=map,
            bots=bots,
            ticks_per_step=ticks_per_step,
            tick_cap=tick_cap,
            maps_dir=maps_dir,
            num_envs=1,
            strict_illegal_actions=strict_illegal_actions,
            socket_timeout=socket_timeout,
        )
        self._env_root = Path(env_root) if env_root is not None else None
        self._worker_factory = worker_factory
        self._client: WorkerClient | None = None
        self._masks: np.ndarray | None = None
        self._ticks = 0
        self._seed = ""
        self.action_space = spaces.make_action_space()
        # The true size arrives with HELLO. Until then the planned size
        # keeps check_env-style construction honest.
        self.observation_space = spaces.make_observation_space()

    def _connect(self) -> WorkerClient:
        if self._worker_factory is not None:
            client = self._worker_factory()
        else:
            port = find_free_port()
            client = connect_worker(self._config, port, self._env_root)
        self.observation_space = spaces.make_observation_space(client.obs_size)
        return client

    def _ensure_client(self) -> WorkerClient:
        if self._client is None or self._client.closed:
            self._client = self._connect()
        return self._client

    @staticmethod
    def _seed_to_string(seed) -> str:
        if seed is None:
            return f"np{np.random.SeedSequence().entropy % 10**12:012d}"
        return str(seed)

    def reset(self, *, seed=None, options=None):
        # Gymnasium seeding only takes ints. String seeds pass through to
        # the worker untouched and skip the local generator.
        super().reset(seed=seed if isinstance(seed, int) or seed is None else None)
        attempts = 0
        while True:
            client = self._ensure_client()
            seed_string = self._seed_to_string(
                seed if seed is not None else self.np_random.integers(10**12)
            )
            try:
                obs_list, ticks, masks = client.reset([seed_string])
            except (ConnectionError, OSError, ProtocolError):
                self._drop_client()
                attempts += 1
                if attempts > 1:
                    raise
                continue
            break
        self._seed = seed_string
        self._ticks = int(ticks[0])
        self._masks = spaces.mask_bytes_to_array(masks[0])
        obs = np.asarray(obs_list[0], dtype=np.float32)
        return obs, self._info(None)

    def step(self, action):
        client = self._ensure_client()
        try:
            obs_list, rewards, done, terminal_codes, ticks, masks = client.step(
                [spaces.action_array_to_dict(action)], [""]
            )
        except (ConnectionError, OSError):
            return self._worker_death()
        terminal = spaces.TERMINAL_NAMES.get(terminal_codes[0])
        self._ticks = int(ticks[0])
        self._masks = spaces.mask_bytes_to_array(masks[0])
        obs = np.asarray(obs_list[0], dtype=np.float32)
        terminated = bool(done[0]) and terminal_codes[0] not in _TRUNCATING_CODES
        truncated = bool(done[0]) and terminal_codes[0] in _TRUNCATING_CODES
        return obs, float(rewards[0]), terminated, truncated, self._info(terminal)

    def _worker_death(self):
        """Respawn after losing the worker mid-step and end the episode."""
        self._drop_client()
        client = self._ensure_client()
        obs_list, ticks, masks = client.reset([self._seed or "respawn"])
        self._ticks = int(ticks[0])
        self._masks = spaces.mask_bytes_to_array(masks[0])
        obs = np.asarray(obs_list[0], dtype=np.float32)
        return obs, 0.0, True, False, self._info(spaces.WORKER_DEATH_REASON)

    def _info(self, terminal_reason: str | None) -> dict:
        return {
            "action_masks": self.action_masks(),
            "terminal_reason": terminal_reason,
            "ticks": self._ticks,
            "seed": self._seed,
        }

    def action_masks(self) -> np.ndarray:
        """Flat bool mask in head order, True where the action head is open."""
        if self._masks is None:
            raise RuntimeError("action_masks called before reset")
        return self._masks.copy()

    def close(self) -> None:
        self._drop_client()

    def _drop_client(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except (OSError, ProtocolError):
                pass
