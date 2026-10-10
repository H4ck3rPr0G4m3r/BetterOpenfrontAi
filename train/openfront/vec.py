"""W worker processes x G games x S seats behind one batched interface.

Rows are worker-major, then game, then seat. Workers may run different
configs (scenarios: seat counts, spawns, bots), so row and game offsets are
tables, not arithmetic; ``seats`` is only defined when every worker agrees.
Every call returns the same preallocated batch dict (raster, features,
masks, reward, stats, terminal, flags), so callers must copy what they keep.
Workers step in parallel threads (socket waits release the GIL). A worker
that dies is respawned; its rows come back DONE with ``worker_death``.
"""

from __future__ import annotations

import dataclasses
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
        config: WorkerConfig | list[WorkerConfig],
        num_workers: int,
        seed: str = "train",
        worker_factory: Callable[[int], WorkerClient] | None = None,
    ):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        configs = list(config) if isinstance(config, (list, tuple)) else [config] * num_workers
        if len(configs) != num_workers:
            raise ValueError(f"{len(configs)} worker configs for {num_workers} workers")
        self.configs = configs
        self.config = configs[0]
        self.num_workers = num_workers
        seat_counts = {c.seats for c in configs}
        self.seats = configs[0].seats if len(seat_counts) == 1 else None
        self.max_seats = max(seat_counts)
        # Per-worker game and row offsets, plus row -> game and game -> rows tables.
        self._game_base, self._row_base, game_rows, row_game = [], [], [], []
        for c in configs:
            self._game_base.append(len(game_rows))
            self._row_base.append(len(row_game))
            for _g in range(c.games):
                start = len(row_game)
                game_rows.append(range(start, start + c.seats))
                row_game.extend([len(game_rows) - 1] * c.seats)
        self._game_rows = game_rows
        self._row_game = row_game
        self.num_games = len(game_rows)
        # Replays rotate over all of the run's games, not per worker.
        self.configs = configs = [dataclasses.replace(c, replay_slot=self._game_base[i], replay_stride=self.num_games)
                                  for i, c in enumerate(configs)]
        self.config = configs[0]
        self.num_rows = len(row_game)
        self.seed = seed
        self._factory = worker_factory or (lambda i: start_worker(self.configs[i]))
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
        return self._row_game[row]

    def rows_of_game(self, game: int) -> range:
        return self._game_rows[game]

    def worker_of_game(self, game: int) -> int:
        for w in range(self.num_workers - 1, -1, -1):
            if game >= self._game_base[w]:
                return w
        raise IndexError(game)

    def _slice(self, w: int) -> slice:
        c = self.configs[w]
        return slice(self._row_base[w], self._row_base[w] + c.games * c.seats)

    def _seeds(self, w: int) -> list[str]:
        suffix = f"-r{self._respawns[w]}" if self._respawns[w] else ""
        return [f"{self.seed}-{self._game_base[w] + g}{suffix}" for g in range(self.configs[w].games)]

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
            base, count = self._game_base[w], self.configs[w].games
            local_abort = [g - base for g in abort_games if base <= g < base + count]
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
