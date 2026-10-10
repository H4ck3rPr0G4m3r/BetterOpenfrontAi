"""Training configuration: defaults, YAML loading, validation.

A config file only needs the keys it changes; everything else comes from
DEFAULTS. Unknown keys are errors (a typo must not silently train the
wrong thing).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "run": {
        "name": "openfront",
        "total_steps": 10_000_000,  # learner transitions
        "device": "auto",
        "seed": 0,
        "checkpoint_dir": "checkpoints",
        "log_dir": "runs",
        "checkpoint_every": 20,  # updates
        "replay": None,  # {"dir": "replays/<name>", "every": 200}
    },
    "env": {
        "map": "World",
        "bots": 20,
        "nations": "default",
        "ticks_per_step": 10,
        "tick_cap": 6000,
        "workers": 8,
        "games_per_worker": 1,
        "seats": 4,
        "disabled_intents": [],
        "disabled_builds": [],
        "maps_dir": None,
    },
    "reward": {
        "land": 1.0,
        "structures": 0.2,
        "kill": 0.5,
        "win": 5.0,
        "death": -2.0,
        "placement": 1.0,
        "survive": 0.0,
    },
    "ppo": {
        "n_steps": 256,
        "epochs": 4,
        "minibatch": 512,
        "lr": 3e-4,
        "lr_final_fraction": 0.2,
        "gamma": 0.997,
        "gae_lambda": 0.95,
        "clip": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.005,
        "ent_weights": {"intent": 1.0, "target": 0.5, "ratio": 0.5, "spatial": 0.1, "site": 0.5},
        "max_grad_norm": 0.5,
        "target_kl": 0.05,
        "amp": True,
    },
    "selfplay": {
        # Seat 0 of every game is always the learner. Each other seat is the
        # live learner with this probability, else a frozen pool snapshot.
        "learner_seat_prob": 0.5,
        "snapshot_every": 10,  # updates
        "pool_size": 24,
        "active_opponents": 4,
        "latest_bias": 0.5,  # share of opponents drawn from the 3 newest snapshots
        "pool_dir": None,  # default: <checkpoint_dir>/<name>/pool
    },
    "curriculum": [],
    "model": {"trunk": 256, "slot_dim": 64, "site_dim": 32, "context": 128},
}

CURRICULUM_ENV_KEYS = {"bots", "nations", "seats", "tick_cap", "map"}


def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key not in base:
            raise ValueError(f"unknown config key {path}{key}")
        if isinstance(base[key], dict) and isinstance(value, dict) and key != "ent_weights":
            out[key] = _merge(base[key], value, f"{path}{key}.")
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    raw = {}
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    cfg = _merge(DEFAULTS, raw)
    if overrides:
        cfg = _merge(cfg, overrides)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    env, ppo = cfg["env"], cfg["ppo"]
    if not 1 <= int(env["seats"]) <= 8:
        raise ValueError("env.seats must be 1..8")
    if env["nations"] not in ("default", "disabled"):
        raise ValueError("env.nations must be default or disabled")
    for key in ("workers", "games_per_worker", "ticks_per_step", "tick_cap"):
        if int(env[key]) < 1:
            raise ValueError(f"env.{key} must be >= 1")
    if not 0.0 <= float(cfg["selfplay"]["learner_seat_prob"]) <= 1.0:
        raise ValueError("selfplay.learner_seat_prob must be 0..1")
    if int(ppo["minibatch"]) < 1 or int(ppo["n_steps"]) < 1 or int(ppo["epochs"]) < 1:
        raise ValueError("ppo.minibatch, n_steps, epochs must be >= 1")
    for i, stage in enumerate(cfg["curriculum"]):
        extra = set(stage) - CURRICULUM_ENV_KEYS - {"promote", "name"}
        if extra:
            raise ValueError(f"curriculum[{i}] has unknown keys {sorted(extra)}")


def stage_env(cfg: dict, stage: int) -> dict:
    """The env section with curriculum stage ``stage`` applied."""
    env = copy.deepcopy(cfg["env"])
    if cfg["curriculum"]:
        stage_cfg = cfg["curriculum"][min(stage, len(cfg["curriculum"]) - 1)]
        env.update({k: v for k, v in stage_cfg.items() if k in CURRICULUM_ENV_KEYS})
    return env


def worker_config(env: dict, reward: dict):
    from .worker import WorkerConfig

    return WorkerConfig(
        games=int(env["games_per_worker"]),
        seats=int(env["seats"]),
        map=str(env["map"]),
        bots=int(env["bots"]),
        nations=str(env["nations"]),
        ticks_per_step=int(env["ticks_per_step"]),
        tick_cap=int(env["tick_cap"]),
        maps_dir=env.get("maps_dir"),
        disabled_intents=tuple(int(v) for v in env.get("disabled_intents", [])),
        disabled_builds=tuple(str(v) for v in env.get("disabled_builds", [])),
        reward={k: float(v) for k, v in reward.items()},
    )
