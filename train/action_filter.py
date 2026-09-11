"""Side-quest AI control UI: behavior presets and inference aggression.

Capability limits (disabled intents/builds) and reward weights are enforced
engine-side through the worker config, so training and eval share them with
zero bridge change. This module owns the rest:

* ``BehaviorPreset``: a validated preset (JSON on disk) mapping to worker
  kwargs plus predictor tweaks. One preset drives training, eval, and the
  live view, which is what makes a challenge setup repeatable.
* Inference aggression: a minimum ratio bucket on ATTACK actions, applied
  to any predictor's output. No retraining needed, works on any checkpoint.
* ``mask_stats`` and ``describe_action``: small readouts for the dashboard
  so a toggle or slider visibly does something.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

INTENT_NAMES = (
    "noop",
    "attack",
    "boat",
    "build_unit",
    "upgrade_structure",
    "move_warship",
    "target_player",
    "alliance_request",
    "alliance_reject",
    "break_alliance",
    "embargo",
    "donate_gold",
    "donate_troops",
    "cancel_attack",
)

ATTACK_INTENT = 1

BUILD_TYPES = (
    "City",
    "Defense Post",
    "Port",
    "SAM Launcher",
    "Missile Silo",
    "Factory",
    "Warship",
)

RATIO_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)

DEFAULT_REWARD_WEIGHTS = (1.0, 0.1, 0.1, 1.0, -1.0)

HEAD_NAMES = ("intent_type", "target", "ratio_bucket", "spatial_cell", "build_site")
HEAD_SIZES = (14, 17, 8, 8192, 32)


def aggression_to_ratio_floor(level: int) -> int:
    """Aggression 0..100 into a minimum ATTACK ratio bucket index.

    Mapping is printed in the UI next to the slider, no hidden math:
    0..33 keeps every bucket open, 34..66 floors attacks at 20%,
    67..100 floors them at 50%.
    """
    if not 0 <= level <= 100:
        raise ValueError(f"aggression must be 0..100, got {level}")
    if level <= 33:
        return 0
    if level <= 66:
        return 2
    return 4


def aggression_label(level: int) -> str:
    if level <= 33:
        return "turtle"
    if level <= 66:
        return "balanced"
    return "berserker"


@dataclass
class BehaviorPreset:
    """One named behavior + training setup for the control UI."""

    name: str = "custom"
    description: str = ""
    aggression: int = 50
    min_attack_ratio: int | None = None
    disabled_intents: tuple = ()
    disabled_builds: tuple = ()
    reward_weights: tuple | None = None
    env_overrides: dict = field(default_factory=dict)
    ppo_overrides: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(raw: dict) -> "BehaviorPreset":
        if not isinstance(raw, dict):
            raise ValueError("preset must be a mapping")
        aggression = int(raw.get("aggression", 50))
        if not 0 <= aggression <= 100:
            raise ValueError(f"aggression must be 0..100, got {aggression}")
        min_ratio = raw.get("min_attack_ratio", None)
        if min_ratio is None:
            min_ratio = aggression_to_ratio_floor(aggression)
        min_ratio = int(min_ratio)
        if not 0 <= min_ratio < len(RATIO_BUCKETS):
            raise ValueError(f"min_attack_ratio must be 0..7, got {min_ratio}")
        disabled_intents = tuple(int(v) for v in raw.get("disabled_intents", ()))
        for value in disabled_intents:
            if not 1 <= value <= 13:
                raise ValueError(f"disabled intent must be 1..13, got {value}")
        disabled_builds = tuple(str(v) for v in raw.get("disabled_builds", ()))
        for name in disabled_builds:
            if name not in BUILD_TYPES:
                raise ValueError(f"unknown build type {name!r}")
        weights = raw.get("reward_weights", None)
        if weights is not None:
            weights = tuple(float(v) for v in weights)
            if len(weights) not in (5, 6, 13):
                raise ValueError(
                    "reward_weights needs 5 values: land,troop,gold,win,death, "
                    "plus an optional 6th per-kill bonus, or the full 13 with "
                    "landEarly,landLoss,city,factory,troopHold,survive,rich"
                )
            for value in weights:
                if not math.isfinite(value):
                    raise ValueError(f"reward_weights must be finite, got {value}")
        env_overrides = dict(raw.get("env_overrides", {}))
        ppo_overrides = dict(raw.get("ppo_overrides", {}))
        return BehaviorPreset(
            name=str(raw.get("name", "custom")),
            description=str(raw.get("description", "")),
            aggression=aggression,
            min_attack_ratio=min_ratio,
            disabled_intents=disabled_intents,
            disabled_builds=disabled_builds,
            reward_weights=weights,
            env_overrides=env_overrides,
            ppo_overrides=ppo_overrides,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "aggression": self.aggression,
            "aggression_label": aggression_label(self.aggression),
            "min_attack_ratio": self.min_attack_ratio,
            "min_attack_ratio_pct": int(RATIO_BUCKETS[self.min_attack_ratio] * 100),
            "disabled_intents": list(self.disabled_intents),
            "disabled_build_names": list(self.disabled_builds),
            "reward_weights": list(self.reward_weights) if self.reward_weights else None,
            "env_overrides": dict(self.env_overrides),
            "ppo_overrides": dict(self.ppo_overrides),
        }

    def to_worker_kwargs(self) -> dict:
        """Disabled sets plus reward weights for OpenFrontVecEnv."""
        return {
            "disabled_intents": self.disabled_intents,
            "disabled_builds": self.disabled_builds,
            "reward_weights": self.reward_weights,
        }


def apply_aggression(action, min_attack_ratio: int):
    """Floor the ratio bucket of ATTACK actions. Passes everything else through."""
    action = np.asarray(action, dtype=np.int64).copy()
    if action[0] == ATTACK_INTENT and action[2] < min_attack_ratio:
        action[2] = min_attack_ratio
    return action


def apply_aggression_batch(actions, min_attack_ratio: int):
    return np.stack([apply_aggression(a, min_attack_ratio) for a in np.asarray(actions)])


def mask_stats(flat_mask) -> dict:
    """Open counts per head, for the dashboard behavior panel."""
    flat = np.asarray(flat_mask, dtype=bool).reshape(-1)
    out = {}
    offset = 0
    for name, size in zip(HEAD_NAMES, HEAD_SIZES):
        head = flat[offset : offset + size]
        out[name] = {"open": int(head.sum()), "total": int(size)}
        offset += size
    return out


def describe_action(action) -> str:
    """One-line human readout of a 5-head action for the live view."""
    intent, target, ratio, cell, site = (int(v) for v in action)
    name = INTENT_NAMES[intent] if 0 <= intent < len(INTENT_NAMES) else f"intent{intent}"
    detail = ""
    if intent == ATTACK_INTENT:
        where = "unowned land" if target == 0 else f"player slot {target}"
        detail = f" {where} @ {int(RATIO_BUCKETS[ratio] * 100)}%"
    elif name in ("boat", "move_warship"):
        detail = f" cell {cell}"
    elif name in ("build_unit", "upgrade_structure"):
        detail = f" site {site}"
    elif name in ("donate_gold", "donate_troops"):
        detail = f" slot {target} @ {int(RATIO_BUCKETS[ratio] * 100)}%"
    elif intent not in (0, 13):
        detail = f" slot {target}"
    return f"{name}{detail}"


def behavior_from_config(cfg: dict) -> dict:
    """Read the ``behavior`` YAML section into OpenFrontVecEnv kwargs."""
    raw = cfg.get("behavior", {}) or {}
    preset = BehaviorPreset.from_dict(
        {
            "aggression": raw.get("aggression", 50),
            "min_attack_ratio": raw.get("min_attack_ratio", None),
            "disabled_intents": raw.get("disabled_intents", ()),
            "disabled_builds": raw.get("disabled_builds", ()),
            "reward_weights": raw.get("reward_weights", None),
        }
    )
    return preset.to_worker_kwargs()
