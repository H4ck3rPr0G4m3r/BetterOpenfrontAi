"""Observation / action layout, schema v2. Mirrors ``env/src/Layout.ts``.

The worker's HELLO carries its layout; ``check_layout`` refuses to run when
the two sides disagree, so they cannot drift silently.
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = 3

RASTER_PLANES = 11
RASTER_HEIGHT = 64
RASTER_WIDTH = 128
RASTER_CELLS = RASTER_HEIGHT * RASTER_WIDTH
RASTER_SIZE = RASTER_PLANES * RASTER_CELLS
PLANE_NAMES = (
    "land", "self", "ally", "enemy", "unowned", "self_troops", "enemy_troops",
    "threat", "targets", "own_structures", "enemy_structures",
)

NUM_SLOTS = 17
SLOT_FIELDS = 16
MAX_BUILD_SITES = 32
SITE_FIELDS = 12
NUM_GLOBALS = 28
SLOT_TABLE = NUM_SLOTS * SLOT_FIELDS
SITE_TABLE = MAX_BUILD_SITES * SITE_FIELDS
FEATURE_SIZE = SLOT_TABLE + SITE_TABLE + NUM_GLOBALS

# Globals (index into the globals block).
G_TIME, G_ALIVE_FRACTION, G_RANK, G_TILES, G_LAND_SHARE, G_TROOPS, G_TROOP_RATIO = range(7)
G_ALIVE = 21

INTENT_NAMES = (
    "noop", "attack", "boat", "build", "upgrade", "move_warship", "target_player",
    "alliance_request", "alliance_reject", "break_alliance", "embargo",
    "donate_gold", "donate_troops", "cancel_attack",
)
NUM_INTENTS = len(INTENT_NAMES)
NOOP, ATTACK, BOAT, BUILD, UPGRADE, MOVE_WARSHIP = range(6)
RATIO_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
NUM_RATIOS = len(RATIO_BUCKETS)
SPATIAL_KINDS = 2
SITE_KINDS = 2
BUILD_TYPES = ("City", "Defense Post", "Port", "SAM Launcher", "Missile Silo", "Factory", "Warship")

MASK_INTENT = 0
MASK_TARGET = MASK_INTENT + NUM_INTENTS
MASK_RATIO = MASK_TARGET + NUM_INTENTS * NUM_SLOTS
MASK_SPATIAL = MASK_RATIO + NUM_RATIOS
MASK_SITE = MASK_SPATIAL + SPATIAL_KINDS * RASTER_CELLS
MASK_SIZE = MASK_SITE + SITE_KINDS * MAX_BUILD_SITES

# Which conditional heads each intent uses: target?, ratio?, spatial kind (-1 none), site kind (-1 none).
USES_TARGET = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
USES_RATIO = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], dtype=bool)
SPATIAL_KIND = np.array([-1, -1, 0, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.int64)
SITE_KIND = np.array([-1, -1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.int64)

TERMINALS = {0: None, 1: "win", 2: "death", 3: "tick_cap", 4: "sim_error", 5: "aborted", 6: "worker_death"}
WORKER_DEATH = 6
TERMINAL_NAMES = ("win", "death", "tick_cap", "sim_error", "aborted", "worker_death")

FLAG_VALID, FLAG_DONE, FLAG_NEXT, FLAG_FRESH = 1, 2, 4, 8
STATS = ("tiles", "troops", "kills", "placement", "ticks")


def layout_dict() -> dict:
    """The layout the TS side advertises in HELLO, for comparison."""
    return {
        "version": SCHEMA_VERSION,
        "rasterPlanes": RASTER_PLANES,
        "rasterHeight": RASTER_HEIGHT,
        "rasterWidth": RASTER_WIDTH,
        "slots": NUM_SLOTS,
        "slotFields": SLOT_FIELDS,
        "buildSites": MAX_BUILD_SITES,
        "siteFields": SITE_FIELDS,
        "globals": NUM_GLOBALS,
        "featureSize": FEATURE_SIZE,
        "intents": NUM_INTENTS,
        "ratioBuckets": NUM_RATIOS,
        "spatialKinds": SPATIAL_KINDS,
        "siteKinds": SITE_KINDS,
        "maskSize": MASK_SIZE,
    }


def check_layout(remote: dict) -> None:
    local = layout_dict()
    diff = {k: (local[k], remote.get(k)) for k in local if remote.get(k) != local[k]}
    if diff:
        raise RuntimeError(f"worker layout differs from train/openfront/layout.py: {diff}")


def split_masks(masks: np.ndarray) -> dict:
    """[N, MASK_SIZE] bool -> per-head views with conditional rows unpacked."""
    n = masks.shape[0]
    return {
        "intent": masks[:, MASK_INTENT:MASK_TARGET],
        "target": masks[:, MASK_TARGET:MASK_RATIO].reshape(n, NUM_INTENTS, NUM_SLOTS),
        "ratio": masks[:, MASK_RATIO:MASK_SPATIAL],
        "spatial": masks[:, MASK_SPATIAL:MASK_SITE].reshape(n, SPATIAL_KINDS, RASTER_CELLS),
        "site": masks[:, MASK_SITE:MASK_SIZE].reshape(n, SITE_KINDS, MAX_BUILD_SITES),
    }


def describe_action(action) -> str:
    """One-line readout of a 5-head action, for logs and the dashboard."""
    intent, target, ratio, cell, site = (int(v) for v in action)
    name = INTENT_NAMES[intent] if 0 <= intent < NUM_INTENTS else f"intent{intent}"
    if intent == ATTACK:
        where = "unowned land" if target == 0 else f"player slot {target}"
        return f"attack {where} with {int(RATIO_BUCKETS[ratio] * 100)}%"
    if intent == BOAT:
        return f"boat to cell ({cell % RASTER_WIDTH},{cell // RASTER_WIDTH}) with {int(RATIO_BUCKETS[ratio] * 100)}%"
    if intent == MOVE_WARSHIP:
        return f"warship to cell ({cell % RASTER_WIDTH},{cell // RASTER_WIDTH})"
    if intent in (BUILD, UPGRADE):
        return f"{name} site {site}"
    if USES_TARGET[intent]:
        return f"{name} slot {target}"
    return name
