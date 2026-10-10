"""W worker processes x G games x S seats behind one batched interface.

Rows are worker-major, then game, then seat: ``row = (w * G + g) * S + s``.
Every call returns the same preallocated batch dict (raster, features,
masks, reward, stats, terminal, flags), so callers must copy what they keep.
Workers step in parallel threads (socket waits release the GIL). A worker
that dies is respawned; its rows come back DONE with ``worker_death``.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from . import layout as L
from .protocol import ProtocolError
from .worker import WorkerClient, WorkerConfig, start_worker

BATCH_FIELDS = ("raster", "features", "masks", "reward", "stats", "terminal", "flags")


class VecGames:
    def __init__(
        self,
        config: WorkerConfig,
        num_workers: int,
        seed: str = "train",
        worker_factory: Callable[[int], WorkerClient] | None = None,
    ):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.config = config
        self.num_workers = num_workers
        self.seats = config.seats
        self.games_per_worker = config.games
        self.num_games = num_workers * config.games
        self.rows_per_worker = config.games * config.seats
        self.num_rows = num_workers * self.rows_per_worker
        self.seed = seed
        self._factory = worker_factory or (lambda _i: start_worker(config))
        self._respawns = [0] * num_workers
        self._pool = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="openfront-worker")
        self.batch = self._empty_batch()
        self._next = np.zeros(self.num_rows, dtype=bool)
        self.clients: list[WorkerClient | None] = list(self._pool.map(self._factory, range(num_workers)))

    def _empty_batch(self) -> dict:
        n = self.num_rows
        return {
            "raster": np.zeros((n, L.RASTER_PLANES, L.RASTER_HEIGHT, L.RASTER_WIDTH), dtype=np.uint8),
            "features": np.zeros((n, L.FEATURE_SIZE), dtype=np.float32),
            "masks": np.zeros((n, L.MASK_SIZE), dtype=bool),
            "reward": np.zeros(n, dtype=np.float32),
            "stats": np.zeros((n, len(L.STATS)), dtype=np.float32),
            "terminal": np.zeros(n, dtype=np.uint8),
            "flags": np.zeros(n, dtype=np.uint8),
        }

    def game_of_row(self, row: int) -> int:
        return row // self.seats

    def rows_of_game(self, game: int) -> range:
        return range(game * self.seats, (game + 1) * self.seats)

    def _slice(self, w: int) -> slice:
        return slice(w * self.rows_per_worker, (w + 1) * self.rows_per_worker)

    def _seeds(self, w: int) -> list[str]:
        suffix = f"-r{self._respawns[w]}" if self._respawns[w] else ""
        return [f"{self.seed}-{w * self.games_per_worker + g}{suffix}" for g in range(self.games_per_worker)]

    def _store(self, w: int, rows: dict) -> None:
        sl = self._slice(w)
        for key in BATCH_FIELDS:
            self.batch[key][sl] = rows[key]

    def reset(self) -> dict:
        def one(w: int):
            return w, self._client(w).reset(self._seeds(w))

        for w, rows in self._pool.map(one, range(self.num_workers)):
            self._store(w, rows)
        self._next = (self.batch["flags"] & L.FLAG_NEXT) != 0
        return self.batch

    def step(self, actions: np.ndarray, abort_games: set[int] | None = None) -> dict:
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_rows, 5)
        abort_games = abort_games or set()

        def one(w: int):
            sl = self._slice(w)
            local_abort = [g - w * self.games_per_worker for g in abort_games
                           if w * self.games_per_worker <= g < (w + 1) * self.games_per_worker]
            try:
                return w, self._client(w).step(actions[sl], abort=sorted(local_abort)), None
            except (ConnectionError, OSError, ProtocolError) as exc:
                return w, None, exc

        for w, rows, exc in list(self._pool.map(one, range(self.num_workers))):
            if exc is None:
                self._store(w, rows)
            else:
                self._respawn(w, exc)
        self._next = (self.batch["flags"] & L.FLAG_NEXT) != 0
        return self.batch

    def _respawn(self, w: int, exc: Exception) -> None:
        """Replace a dead worker, reset its games, and end its rows' episodes."""
        old = self.clients[w]
        log = old.log_path if old is not None else None
        warnings.warn(f"worker {w} failed ({exc}); respawning. log: {log}", RuntimeWarning, stacklevel=3)
        if old is not None:
            try:
                old.close()
            except OSError:
                pass
        self.clients[w] = None
        self._respawns[w] += 1
        rows = self._client(w).reset(self._seeds(w))
        sl = self._slice(w)
        was_active = self._next[sl].copy()
        self._store(w, rows)
        self.batch["reward"][sl] = 0.0
        self.batch["terminal"][sl] = np.where(was_active, L.WORKER_DEATH, 0)
        self.batch["flags"][sl] = (
            np.where(was_active, L.FLAG_VALID | L.FLAG_DONE, 0) | L.FLAG_NEXT | L.FLAG_FRESH
        ).astype(np.uint8)

    def _client(self, w: int) -> WorkerClient:
        client = self.clients[w]
        if client is None or client.closed:
            client = self._factory(w)
            self.clients[w] = client
        return client

    def worker_logs(self) -> list[str | None]:
        return [c.log_path if c is not None else None for c in self.clients]

    def close(self) -> None:
        for client in self.clients:
            if client is not None:
                try:
                    client.close()
                except OSError:
                    pass
        self.clients = [None] * self.num_workers
        self._pool.shutdown(wait=False)
