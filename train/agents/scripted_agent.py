"""A deterministic heuristic baseline. Expands, then attacks, then builds.

Priority per step, all gated on the masks so every emitted action is
legal: expand into bordering unowned land, strike an open player target,
sail to an open cell, raise a structure on an open site, otherwise pass.
Troop and gold amounts always use the middle ratio bucket. Deterministic
on purpose: same trajectory in, same actions out, which makes it a
stable yardstick for the acceptance runs.
"""

from __future__ import annotations

import numpy as np

from openfront_gym import spaces


class ScriptedBaselineAgent:
    def act(self, obs, action_masks, info=None) -> np.ndarray:
        del obs, info
        heads = spaces.split_flat_mask(action_masks)
        intent_open = np.asarray(heads["intent_type"], dtype=bool)
        target_open = np.asarray(heads["target"], dtype=bool)
        spatial_open = np.asarray(heads["spatial_cell"], dtype=bool)
        site_open = np.asarray(heads["build_site"], dtype=bool)

        def first(mask, start=0):
            hits = spaces.open_indices(mask)
            hits = hits[hits >= start]
            return int(hits[0]) if len(hits) else None

        attack = spaces.IntentType.ATTACK
        if intent_open[attack] and target_open[spaces.TERRA_NULLIUS_SLOT]:
            return self._make(attack, spaces.TERRA_NULLIUS_SLOT, 0, 0)
        player_slot = first(target_open, start=1)
        if intent_open[attack] and player_slot is not None:
            return self._make(attack, player_slot, 0, 0)
        boat = spaces.IntentType.BOAT
        cell = first(spatial_open)
        if intent_open[boat] and cell is not None:
            action = self._make(boat, 0, cell, 0)
            action[0] = boat
            return action
        build = spaces.IntentType.BUILD_UNIT
        site = first(site_open)
        if intent_open[build] and site is not None:
            return self._make(build, 0, 0, site)
        return self._make(spaces.IntentType.NOOP, 0, 0, 0)

    @staticmethod
    def _make(intent: int, target: int, cell: int, site: int) -> np.ndarray:
        return np.array(
            [intent, target, spaces.MID_RATIO_BUCKET, cell, site], dtype=np.int64
        )

    def act_batch(self, obs_batch, masks_batch, infos=None) -> np.ndarray:
        infos = infos if infos is not None else [None] * len(obs_batch)
        return np.stack(
            [self.act(obs, masks, info) for obs, masks, info in zip(obs_batch, masks_batch, infos)]
        )
