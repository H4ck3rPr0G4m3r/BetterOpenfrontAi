"""Uniform sampling over the open mask entries, one head at a time."""

from __future__ import annotations

import numpy as np

from openfront_gym import spaces


class MaskedRandomAgent:
    """The phase 1 random baseline. Never picks a masked-out entry."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, action_masks, info=None) -> np.ndarray:
        del obs, info
        heads = spaces.split_flat_mask(action_masks)
        action = np.zeros(len(spaces.HEAD_NAMES), dtype=np.int64)
        for index, name in enumerate(spaces.HEAD_NAMES):
            open_entries = spaces.open_indices(heads[name])
            action[index] = int(self.rng.choice(open_entries)) if len(open_entries) else 0
        return action

    def act_batch(self, obs_batch, masks_batch, infos=None) -> np.ndarray:
        infos = infos if infos is not None else [None] * len(obs_batch)
        return np.stack(
            [self.act(obs, masks, info) for obs, masks, info in zip(obs_batch, masks_batch, infos)]
        )
