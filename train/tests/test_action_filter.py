"""Tests for the side-quest behavior controls (action_filter)."""

import numpy as np
import pytest

from action_filter import (
    ATTACK_INTENT,
    BUILD_TYPES,
    BehaviorPreset,
    aggression_to_ratio_floor,
    apply_aggression,
    behavior_from_config,
    describe_action,
    mask_stats,
)
from openfront_gym import spaces
from openfront_gym.worker import WorkerConfig, worker_argv


def test_aggression_mapping_is_monotone_and_bounded():
    assert aggression_to_ratio_floor(0) == 0
    assert aggression_to_ratio_floor(50) == 2
    assert aggression_to_ratio_floor(100) == 4
    with pytest.raises(ValueError):
        aggression_to_ratio_floor(101)


def test_preset_validation():
    preset = BehaviorPreset.from_dict(
        {"name": "no-factories", "aggression": 80, "disabled_builds": ["Factory"]}
    )
    assert preset.min_attack_ratio == 4
    kwargs = preset.to_worker_kwargs()
    assert kwargs["disabled_builds"] == ("Factory",)
    assert kwargs["reward_weights"] is None

    with pytest.raises(ValueError):
        BehaviorPreset.from_dict({"disabled_intents": [0]})
    with pytest.raises(ValueError):
        BehaviorPreset.from_dict({"disabled_builds": ["DeathStar"]})
    with pytest.raises(ValueError):
        BehaviorPreset.from_dict({"reward_weights": [1, 2, 3]})
    with pytest.raises(ValueError):
        BehaviorPreset.from_dict({"aggression": -1})


def test_preset_build_names_match_engine_candidates():
    # The TypeScript side pins KNOWN_BUILD_TYPES to BUILD_CANDIDATE_TYPES;
    # this pins the Python copy to the same seven names.
    assert set(BUILD_TYPES) == {
        "City",
        "Defense Post",
        "Port",
        "SAM Launcher",
        "Missile Silo",
        "Factory",
        "Warship",
    }


def test_apply_aggression_floors_only_attacks():
    attack = np.array([ATTACK_INTENT, 1, 0, 0, 0])
    assert apply_aggression(attack, 4)[2] == 4
    assert apply_aggression(attack, 0)[2] == 0
    noop = np.array([0, 0, 0, 0, 0])
    assert apply_aggression(noop, 4)[2] == 0


def test_mask_stats_and_describe():
    flat = np.zeros(spaces.MASK_TOTAL, dtype=bool)
    flat[spaces.IntentType.ATTACK] = True
    stats = mask_stats(flat)
    assert stats["intent_type"] == {"open": 1, "total": 14}
    assert "attack" in describe_action(np.array([1, 0, 4, 0, 0]))
    assert "unowned land" in describe_action(np.array([1, 0, 4, 0, 0]))
    assert describe_action(np.array([0, 0, 0, 0, 0])) == "noop"


def test_behavior_from_config_defaults_to_no_limits():
    kwargs = behavior_from_config({"env": {}})
    assert kwargs == {
        "disabled_intents": (),
        "disabled_builds": (),
        "reward_weights": None,
    }


def test_worker_argv_carries_behavior_flags():
    argv = worker_argv(
        WorkerConfig(
            num_envs=1,
            disabled_intents=(1, 10),
            disabled_builds=("Factory",),
            reward_weights=(2.0, 0.0, 0.0, 5.0, -5.0),
        ),
        5001,
    )
    assert ["--disable-intent", "1"] in [argv[i : i + 2] for i in range(len(argv))]
    assert ["--disable-build", "Factory"] in [argv[i : i + 2] for i in range(len(argv))]
    assert "--reward-weights" in argv
    assert "2.0,0.0,0.0,5.0,-5.0" in argv
