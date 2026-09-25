"""Self-play opponent pool: checkpoint snapshots as sparring partners.

Gating (user requirement): self-play is useful once the policy can
consistently beat the engine bots. The pool stays dormant until eval win
rate vs the scripted baseline crosses ``SELFPLAY_WIN_RATE_GATE``
(``selfplay_ready()``); before that, training runs the bot curriculum in
``curriculum.py``. This module is pure stdlib so it is unit-testable
without torch, gymnasium, or a live worker.

Same-game note: today's bridge seats exactly one learner per env
(``GameEnv`` creates a single Human seat; opponents are engine bots), so
there is no protocol path to inject a pool checkpoint as an in-game
opponent yet. Until that multi-seat bridge lands, "self-play" here means
league-style training: the learner trains against bots while the pool
tracks snapshots, every segment is evaluated against sampled pool
checkpoints on the SAME fixed seeds (``evaluate_vs_pool()`` in
``train_sp.py``), and promotion requires beating both bots and the pool's
best. The pool file format is forward-compatible with a future in-game
opponent seat: snapshots are plain checkpoint zips with step metadata.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

SELFPLAY_WIN_RATE_GATE = 0.5
"""Eval win rate vs bots at which pool sparring activates."""


def selfplay_ready(eval_win_rate: float, threshold: float = SELFPLAY_WIN_RATE_GATE) -> bool:
    """True once the policy beats bots consistently enough for pool play."""
    return float(eval_win_rate) >= float(threshold)


@dataclass
class Snapshot:
    path: str
    step: int
    wins: int = 0
    games: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def to_dict(self) -> dict:
        return {"path": self.path, "step": self.step, "wins": self.wins, "games": self.games}

    @staticmethod
    def from_dict(raw: dict) -> "Snapshot":
        return Snapshot(
            path=str(raw["path"]),
            step=int(raw["step"]),
            wins=int(raw.get("wins", 0)),
            games=int(raw.get("games", 0)),
        )


@dataclass
class OpponentPool:
    """Checkpoint snapshot store with sampling strategies and an index file."""

    root: Path
    snapshots: list[Snapshot] = field(default_factory=list)

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def add_snapshot(self, checkpoint: Path | str, step: int) -> Snapshot:
        """Copy a checkpoint into the pool and register it. Idempotent per step."""
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"snapshot source missing: {checkpoint}")
        for existing in self.snapshots:
            if existing.step == int(step):
                return existing
        dest = self.root / f"snapshot_step{int(step)}.zip"
        shutil.copyfile(checkpoint, dest)
        snap = Snapshot(path=str(dest), step=int(step))
        self.snapshots.append(snap)
        self.snapshots.sort(key=lambda s: s.step)
        self.save()
        return snap

    def sample(
        self,
        strategy: str = "uniform",
        last_k: int = 3,
        rng: random.Random | None = None,
    ) -> Snapshot | None:
        """Draw one opponent snapshot. None when the pool is empty."""
        if not self.snapshots:
            return None
        rng = rng or random.Random()
        if strategy == "latest":
            return self.snapshots[-1]
        if strategy == "last_k":
            window = self.snapshots[-max(int(last_k), 1) :]
            return rng.choice(window)
        if strategy == "uniform":
            return rng.choice(self.snapshots)
        raise ValueError(f"unknown pool sampling strategy {strategy!r}")

    def best(self) -> Snapshot | None:
        """Highest-step snapshot with at least one recorded game, else latest."""
        if not self.snapshots:
            return None
        played = [s for s in self.snapshots if s.games > 0]
        if not played:
            return self.snapshots[-1]
        return max(played, key=lambda s: (s.win_rate, s.step))

    def record_result(self, snapshot: Snapshot, won: bool) -> None:
        for existing in self.snapshots:
            if existing.step == snapshot.step:
                existing.games += 1
                existing.wins += 1 if won else 0
                break
        self.save()

    def save(self) -> None:
        self.index_path.write_text(
            json.dumps({"snapshots": [s.to_dict() for s in self.snapshots]}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def load(root: Path | str) -> "OpponentPool":
        pool = OpponentPool(Path(root))
        if pool.index_path.is_file():
            raw = json.loads(pool.index_path.read_text(encoding="utf-8"))
            pool.snapshots = [Snapshot.from_dict(item) for item in raw.get("snapshots", [])]
            pool.snapshots.sort(key=lambda s: s.step)
        return pool
