"""Self-play package: opponent pool plus bot curriculum."""

from .curriculum import STAGES, next_stage, should_promote, stage_by_name
from .pool import SELFPLAY_WIN_RATE_GATE, OpponentPool, Snapshot, selfplay_ready

__all__ = [
    "OpponentPool",
    "SELFPLAY_WIN_RATE_GATE",
    "STAGES",
    "Snapshot",
    "next_stage",
    "selfplay_ready",
    "should_promote",
    "stage_by_name",
]
