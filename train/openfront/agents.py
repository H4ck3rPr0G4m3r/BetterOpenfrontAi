"""Non-learning agents over the v2 observation and masks (numpy only).

* ``RandomAgent`` — conditional masked-uniform: what an untrained policy does.
* ``ScriptedAgent`` — a readable heuristic yardstick: expand at 20% while
  the army is above 45% of max, counter-attack whoever is attacking us,
  punish much weaker neighbours, build cities when affordable. The v1
  scripted baseline expanded at 50% every second, which starves the army
  (measured: ~1.2k troops per attack, dead by step 88-425).
* ``NoopAgent`` — never acts.

All take a batch (raster, features, masks) and return int64 actions [N, 5].
"""

from __future__ import annotations

import numpy as np

from . import layout as L

SLOT = L.SLOT_FIELDS
F_VALID, F_STRENGTH, F_BORDER, F_ATTACKING_US = 0, 4, 7, 8
G0 = L.SLOT_TABLE + L.SITE_TABLE
CITY = 0  # BUILD_TYPES index


class NoopAgent:
    def act(self, raster, features, masks) -> np.ndarray:
        return np.zeros((len(features), 5), dtype=np.int64)


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def _pick(self, mask) -> int:
        open_ = np.flatnonzero(mask)
        return int(self.rng.choice(open_)) if len(open_) else 0

    def act(self, raster, features, masks) -> np.ndarray:
        heads = L.split_masks(np.asarray(masks, dtype=bool))
        out = np.zeros((len(features), 5), dtype=np.int64)
        for i in range(len(features)):
            intent = self._pick(heads["intent"][i])
            out[i, 0] = intent
            if L.USES_TARGET[intent]:
                out[i, 1] = self._pick(heads["target"][i, intent])
            if L.USES_RATIO[intent]:
                out[i, 2] = self._pick(heads["ratio"][i])
            if L.SPATIAL_KIND[intent] >= 0:
                out[i, 3] = self._pick(heads["spatial"][i, L.SPATIAL_KIND[intent]])
            if L.SITE_KIND[intent] >= 0:
                out[i, 4] = self._pick(heads["site"][i, L.SITE_KIND[intent]])
        return out


class ScriptedAgent:
    def act(self, raster, features, masks) -> np.ndarray:
        heads = L.split_masks(np.asarray(masks, dtype=bool))
        out = np.zeros((len(features), 5), dtype=np.int64)
        for i, f in enumerate(np.asarray(features)):
            out[i] = self._one(f, {k: v[i] for k, v in heads.items()})
        return out

    @staticmethod
    def _one(f: np.ndarray, m: dict) -> list[int]:
        ratio = f[G0 + L.G_TROOP_RATIO]
        slots = f[: L.SLOT_TABLE].reshape(L.NUM_SLOTS, SLOT)
        attack = m["target"][L.ATTACK] if m["intent"][L.ATTACK] else np.zeros(L.NUM_SLOTS, bool)
        # 1. Someone is attacking us: hit back (opposing attacks cancel out).
        threats = [s for s in range(1, L.NUM_SLOTS) if attack[s] and slots[s, F_ATTACKING_US] > 0]
        if threats and ratio > 0.25:
            worst = max(threats, key=lambda s: slots[s, F_ATTACKING_US])
            return [L.ATTACK, worst, 3, 0, 0]
        # 2. Expand into free land at a sustainable pace.
        if attack[0] and ratio > 0.45:
            return [L.ATTACK, 0, 2, 0, 0]
        # 3. Punish a much weaker neighbour when the army is nearly full.
        weak = [s for s in range(1, L.NUM_SLOTS) if attack[s] and slots[s, F_STRENGTH] < 0.3]
        if weak and ratio > 0.8:
            return [L.ATTACK, min(weak, key=lambda s: slots[s, F_STRENGTH]), 3, 0, 0]
        # 4. Cities raise max troops and gold.
        if m["intent"][L.BUILD]:
            sites = f[L.SLOT_TABLE : L.SLOT_TABLE + L.SITE_TABLE].reshape(L.MAX_BUILD_SITES, L.SITE_FIELDS)
            for site in np.flatnonzero(m["site"][0]):
                if sites[site, 1 + CITY] > 0:
                    return [L.BUILD, 0, 0, 0, int(site)]
        return [L.NOOP, 0, 0, 0, 0]


def make_agent(kind: str, seed: int = 0):
    if kind == "random":
        return RandomAgent(seed)
    if kind == "scripted":
        return ScriptedAgent()
    if kind == "noop":
        return NoopAgent()
    raise ValueError(f"unknown agent {kind!r} (random | scripted | noop)")
