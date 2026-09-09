"""Split the flat bridge observation into raster, table, and globals.

The bridge sends one flat float32 vector per env. Layout, matching
``env/src/ObservationEncoder.ts`` and ``env/src/bridge/Protocol.ts``:

* raster first: 8 planes over a 128 by 64 grid, 65536 entries
* player table next: 17 slots times 8 fields, 136 entries, slot 0 terra nullius
* 12 global scalars last

Total is 65684. This module only reshapes and validates. Normalization, if
ever added, lives here and must document its ranges.
"""

from __future__ import annotations

import numpy as np

from .protocol import OBSERVATION_SIZE

RASTER_WIDTH = 128
RASTER_HEIGHT = 64
RASTER_PLANES = 8
RASTER_SIZE = RASTER_WIDTH * RASTER_HEIGHT * RASTER_PLANES

NUM_SLOTS = 17
SLOT_FIELDS = 8
TABLE_SIZE = NUM_SLOTS * SLOT_FIELDS

NUM_GLOBALS = 12

SPATIAL_SHAPE = (RASTER_PLANES, RASTER_HEIGHT, RASTER_WIDTH)
TABLE_SHAPE = (NUM_SLOTS, SLOT_FIELDS)

assert RASTER_SIZE + TABLE_SIZE + NUM_GLOBALS == OBSERVATION_SIZE  # noqa: S101


def split_observation(obs) -> dict[str, np.ndarray]:
    """One flat observation into spatial, table, and global views."""
    flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    if flat.shape[0] != OBSERVATION_SIZE:
        raise ValueError(f"obs must hold {OBSERVATION_SIZE} floats, got {flat.shape[0]}")
    return {
        "spatial": flat[:RASTER_SIZE].reshape(SPATIAL_SHAPE),
        "table": flat[RASTER_SIZE : RASTER_SIZE + TABLE_SIZE].reshape(TABLE_SHAPE),
        "globals": flat[RASTER_SIZE + TABLE_SIZE :].reshape((NUM_GLOBALS,)),
    }


def split_batch(obs_batch) -> dict[str, np.ndarray]:
    """Batched flat observations into batched views, axis 0 is the env."""
    batch = np.asarray(obs_batch, dtype=np.float32)
    if batch.ndim != 2 or batch.shape[1] != OBSERVATION_SIZE:
        raise ValueError(f"obs batch must be (N, {OBSERVATION_SIZE}), got {batch.shape}")
    return {
        "spatial": batch[:, :RASTER_SIZE].reshape((-1,) + SPATIAL_SHAPE),
        "table": batch[:, RASTER_SIZE : RASTER_SIZE + TABLE_SIZE].reshape((-1,) + TABLE_SHAPE),
        "globals": batch[:, RASTER_SIZE + TABLE_SIZE :].reshape((-1, NUM_GLOBALS)),
    }
