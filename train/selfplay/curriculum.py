"""Bot curriculum: staged opponent difficulty with promotion rules.

The 2M-step plateau diagnosis: training starts on full World / 20 bots
from step 0, so win/kill rewards almost never fire and the policy learns
to turtle. The curriculum stages difficulty instead:

* stage1 (easy): 5 bots, nations disabled — wins and kills fire early.
* stage2 (medium): 10 bots, nations default.
* stage3 (full): 20 bots, nations default — the reference task.

Promotion requires ``promote_win_rate`` eval win rate over at least
``min_episodes`` eval episodes, against BOTH the stage's bots and (once
gated) the pool best. Pure stdlib, unit-tested without a worker.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    config: str
    bots: int
    nations: str
    promote_win_rate: float = 0.5
    min_episodes: int = 10


STAGES: tuple[Stage, ...] = (
    Stage(name="stage1", config="configs/stage1easy.yaml", bots=5, nations="disabled"),
    Stage(name="stage2", config="configs/stage2medium.yaml", bots=10, nations="default"),
    Stage(name="stage3", config="configs/stage3full.yaml", bots=20, nations="default",
          promote_win_rate=1.0 + 1e-9, min_episodes=10),
)


def stage_by_name(name: str) -> Stage:
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise ValueError(f"unknown curriculum stage {name!r}")


def should_promote(stage: Stage, win_rate: float, episodes: int, pool_win_rate: float | None = None) -> bool:
    """Promote when bots are beaten over enough episodes, and the pool best
    (when pool play is active) is beaten too."""
    if episodes < stage.min_episodes:
        return False
    if float(win_rate) < stage.promote_win_rate:
        return False
    if pool_win_rate is not None and float(pool_win_rate) < stage.promote_win_rate:
        return False
    return True


def next_stage(current: str) -> str | None:
    """Name of the stage after ``current``, or None when already at full."""
    names = [s.name for s in STAGES]
    index = names.index(current)
    if index + 1 >= len(names):
        return None
    return names[index + 1]
