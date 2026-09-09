"""Action and observation shapes shared by the gym wrapper and the agents.

The numbers mirror ``env/src/bridge/Protocol.ts`` and, through it, the
action codec and observation encoder. The bridge size test pins the
TypeScript side; the mask order here must stay identical: intent type,
target, ratio bucket, spatial cell, build site.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from .protocol import (
    MASK_BUILD_SITES,
    MASK_INTENT_TYPES,
    MASK_RATIO_BUCKETS,
    MASK_SPATIAL_CELLS,
    MASK_TARGET_SLOTS,
    MASK_TOTAL,
    OBSERVATION_SIZE,
    TerminalCode,
)

HEAD_NAMES = ("intent_type", "target", "ratio_bucket", "spatial_cell", "build_site")
HEAD_SIZES = (
    MASK_INTENT_TYPES,
    MASK_TARGET_SLOTS,
    MASK_RATIO_BUCKETS,
    MASK_SPATIAL_CELLS,
    MASK_BUILD_SITES,
)
HEAD_OFFSETS = (0, 14, 31, 39, 8231)

assert HEAD_OFFSETS[-1] + HEAD_SIZES[-1] == MASK_TOTAL  # noqa: S101


class IntentType:
    NOOP = 0
    ATTACK = 1
    BOAT = 2
    BUILD_UNIT = 3
    UPGRADE_STRUCTURE = 4
    MOVE_WARSHIP = 5
    TARGET_PLAYER = 6
    ALLIANCE_REQUEST = 7
    ALLIANCE_REJECT = 8
    BREAK_ALLIANCE = 9
    EMBARGO = 10
    DONATE_GOLD = 11
    DONATE_TROOPS = 12
    CANCEL_ATTACK = 13


TERRA_NULLIUS_SLOT = 0
MID_RATIO_BUCKET = 4

TERMINAL_NAMES = {
    TerminalCode.CONTINUING: None,
    TerminalCode.WIN: "win",
    TerminalCode.DEATH: "death",
    TerminalCode.TICK_CAP: "tick_cap",
    TerminalCode.SIM_ERROR: "sim_error",
}

WORKER_DEATH_REASON = "worker_death"


def make_action_space() -> gymnasium.spaces.MultiDiscrete:
    """One integer per head, in HEAD_NAMES order."""
    return gymnasium.spaces.MultiDiscrete(list(HEAD_SIZES), dtype=np.int64)


def make_observation_space(obs_size: int = OBSERVATION_SIZE) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
    )


def action_array_to_dict(action) -> dict:
    """A (5,) array-like into the mapping the wire encoder takes."""
    values = [int(v) for v in action]
    if len(values) != len(HEAD_NAMES):
        raise ValueError(f"action must hold 5 heads, got {len(values)}")
    return dict(zip(HEAD_NAMES, values))


def flatten_mask_dict(mask: dict) -> np.ndarray:
    """Per-head mask arrays into one flat bool vector in schema order."""
    parts = [np.asarray(mask[name], dtype=bool).reshape(-1) for name in HEAD_NAMES]
    for part, want in zip(parts, HEAD_SIZES):
        if part.shape[0] != want:
            raise ValueError(f"mask head must hold {want} entries, got {part.shape[0]}")
    return np.concatenate(parts)


def split_flat_mask(flat) -> dict:
    """One flat bool vector back into per-head views."""
    flat = np.asarray(flat, dtype=bool).reshape(-1)
    if flat.shape[0] != MASK_TOTAL:
        raise ValueError(f"flat mask must hold {MASK_TOTAL} entries, got {flat.shape[0]}")
    return {
        name: flat[offset : offset + size] for name, offset, size in zip(HEAD_NAMES, HEAD_OFFSETS, HEAD_SIZES)
    }


def open_indices(head_mask) -> np.ndarray:
    """Indices of the open entries of one head mask."""
    return np.flatnonzero(np.asarray(head_mask, dtype=bool))


def mask_bytes_to_array(raw: bytes) -> np.ndarray:
    """Wire mask bytes into a flat bool vector.

    Plain ``np.asarray`` on bytes gives a scalar truth value, not a
    vector, so this goes through uint8 explicitly.
    """
    arr = np.frombuffer(bytes(raw), dtype=np.uint8).astype(bool)
    if arr.shape[0] != MASK_TOTAL:
        raise ValueError(f"mask must hold {MASK_TOTAL} bytes, got {arr.shape[0]}")
    return arr
