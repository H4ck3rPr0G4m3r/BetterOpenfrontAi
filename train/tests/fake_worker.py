"""In-process fake bridge worker speaking protocol v2, for tests without Node.

Each game lasts ``game_length`` steps; seat ``s`` dies at step
``death_at[s]`` (None = survives to the cap). Masks keep noop and
attack-on-slot-0 open so policies have real choices.
"""

from __future__ import annotations

import json
import socket
import struct
import threading

import numpy as np

from openfront import layout as L
from openfront import protocol as P
from openfront.worker import WorkerClient, WorkerConfig


class FakeServer:
    def __init__(self, games: int, seats: int, game_length: int = 8, death_at: dict | None = None, reward: float = 0.1):
        self.games, self.seats = games, seats
        self.rows = games * seats
        self.game_length = game_length
        self.death_at = death_at or {}
        self.reward = reward
        self.steps = [0] * games
        self.alive = np.ones((games, seats), bool)
        self.resets: list[str] = []
        self.aborts: list[int] = []
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _masks(self, active: bool) -> np.ndarray:
        m = np.zeros(L.MASK_SIZE, np.uint8)
        m[L.MASK_INTENT + L.NOOP] = 1
        if active:
            m[L.MASK_INTENT + L.ATTACK] = 1
            m[L.MASK_TARGET + L.ATTACK * L.NUM_SLOTS + 0] = 1
            m[L.MASK_RATIO : L.MASK_SPATIAL] = 1
        return m

    def _rows(self, rewards, terminals, flags) -> bytes:
        n = self.rows
        raster = np.full((n, L.RASTER_SIZE), 7, np.uint8)
        feats = np.full((n, L.FEATURE_SIZE), 0.5, "<f4")
        masks = np.stack([self._masks(bool(f & L.FLAG_NEXT)) for f in flags])
        stats = np.tile(np.array([100, 1000, 0, 0.5, 0], "<f4"), (n, 1))
        return (struct.pack("<I", n) + raster.tobytes() + feats.tobytes() + masks.tobytes()
                + np.asarray(rewards, "<f4").tobytes() + stats.tobytes()
                + np.asarray(terminals, np.uint8).tobytes() + np.asarray(flags, np.uint8).tobytes())

    def _serve(self):
        conn, _ = self.listener.accept()
        hello = {"layout": L.layout_dict(), "games": self.games, "seats": self.seats, "rows": self.rows}
        P.send_frame(conn, 0, P.MSG_HELLO, P.encode_json(hello))
        try:
            while True:
                seq, kind, body = P.recv_frame(conn)
                if kind == P.MSG_CLOSE:
                    return
                if kind == P.MSG_RESET:
                    self.resets += json.loads(bytes(body))["seeds"]
                    self.steps = [0] * self.games
                    self.alive[:] = True
                    reply = self._rows([0] * self.rows, [0] * self.rows, [L.FLAG_NEXT | L.FLAG_FRESH] * self.rows)
                else:
                    (n,) = struct.unpack_from("<I", body, 0)
                    jlen = struct.unpack_from("<I", body, 4 + n * 10)[0]
                    extra = json.loads(bytes(body[8 + n * 10 : 8 + n * 10 + jlen]))
                    reply = self._step(set(extra.get("abort", [])))
                P.send_frame(conn, seq, P.MSG_ROWS, reply)
        except (ConnectionError, OSError):
            return

    def _step(self, abort: set) -> bytes:
        rewards, terminals, flags = [0.0] * self.rows, [0] * self.rows, [0] * self.rows
        for g in range(self.games):
            self.steps[g] += 1
            over = g in abort or self.steps[g] >= self.game_length
            for s in range(self.seats):
                r = g * self.seats + s
                if not self.alive[g, s]:
                    continue
                f = L.FLAG_VALID
                rewards[r] = self.reward
                if g in abort:
                    terminals[r], f, rewards[r] = 5, f | L.FLAG_DONE, 0.0
                elif self.death_at.get(s) == self.steps[g]:
                    terminals[r], f = 2, f | L.FLAG_DONE
                    self.alive[g, s] = False
                elif over:
                    terminals[r], f = 3, f | L.FLAG_DONE
                else:
                    f |= L.FLAG_NEXT
                flags[r] = f
            if over or not self.alive[g].any():
                if g in abort:
                    self.aborts.append(g)
                self.steps[g] = 0
                self.alive[g] = True
                for s in range(self.seats):
                    flags[g * self.seats + s] |= L.FLAG_NEXT | L.FLAG_FRESH
        return self._rows(rewards, terminals, flags)


def fake_factory(servers: list, **kwargs):
    """worker_factory for VecGames/Trainer: one FakeServer per worker."""
    def make(index: int, config: WorkerConfig | None = None):
        cfg = config or WorkerConfig()
        server = FakeServer(cfg.games, cfg.seats, **kwargs)
        servers.append(server)
        return WorkerClient.connect(server.port, cfg)
    return make
