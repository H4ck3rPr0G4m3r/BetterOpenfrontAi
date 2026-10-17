"""Self-play opponent pool: frozen snapshots of the learner that play in the
same game as the live learner (multi-seat GameEnv), next to engine bots.

Each game's seat 0 is the learner; every other seat is the live learner
with probability ``learner_seat_prob`` (pure self-play, trained on) or a
frozen snapshot from the pool (not trained on; keeps the learner honest
against its past selves and stops strategy cycling). Snapshots are plain
v2 checkpoints plus ``index.json`` with per-snapshot head-to-head results.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

LEARNER = -1


@dataclass
class Snapshot:
    path: str
    step: int
    games: int = 0
    learner_wins: int = 0

    @property
    def learner_win_rate(self) -> float:
        return self.learner_wins / self.games if self.games else 0.0


@dataclass
class OpponentPool:
    root: Path
    max_size: int = 24
    snapshots: list[Snapshot] = field(default_factory=list)

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        index = self.root / "index.json"
        if index.is_file() and not self.snapshots:
            raw = json.loads(index.read_text(encoding="utf-8"))
            self.snapshots = [Snapshot(**s) for s in raw.get("snapshots", []) if Path(s["path"]).is_file()]

    def add(self, model, step: int) -> Snapshot:
        from .model import save_checkpoint

        path = self.root / f"snapshot_{int(step):012d}.pt"
        save_checkpoint(path, model, meta={"step": int(step), "snapshot": True})
        snap = Snapshot(str(path), int(step))
        self.snapshots.append(snap)
        # Keep the newest half and an even spread of the older ones.
        while len(self.snapshots) > self.max_size:
            older = self.snapshots[: len(self.snapshots) // 2]
            victim = older[len(older) // 2]
            self.snapshots.remove(victim)
            Path(victim.path).unlink(missing_ok=True)
        self.save()
        return snap

    def sample(self, k: int, latest_bias: float, rng: random.Random) -> list[Snapshot]:
        """k opponents: ``latest_bias`` of them from the 3 newest, the rest uniform."""
        if not self.snapshots:
            return []
        out = []
        for _ in range(k):
            pool = self.snapshots[-3:] if rng.random() < latest_bias else self.snapshots
            out.append(rng.choice(pool))
        return out

    def record(self, snapshot_path: str, learner_won: bool) -> None:
        for snap in self.snapshots:
            if snap.path == snapshot_path:
                snap.games += 1
                snap.learner_wins += int(learner_won)
                return

    def save(self) -> None:
        payload = {"snapshots": [s.__dict__ for s in self.snapshots]}
        (self.root / "index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def assign_seats(seats: int, opponents: int, learner_seat_prob: float, rng: random.Random) -> list[int]:
    """Controller per seat: LEARNER or an index into the active opponents."""
    out = [LEARNER]
    for _ in range(1, seats):
        if opponents == 0 or rng.random() < learner_seat_prob:
            out.append(LEARNER)
        else:
            out.append(rng.randrange(opponents))
    return out
