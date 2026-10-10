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
        # Donations are off: real games rarely need them, and the v2 policy
        # gave its troops away to allied nations. Re-enable for fine-tuning.
        "allow_donations": False,
        "allow_betrayal": False,  # breaking alliances: off, like donations
        "troop_reserve": 0.1,  # attacks must leave this share of max troops home
        "close_spawn": 0.3,  # share of multi-seat games with seats spawned close
        "maps_dir": None,
    },
    "reward": {
        "land": 1.0,
        "structures": 0.2,
        "cities": 0.5,
        "rail": 0.5,
        "popCap": 0.3,
        "income": 1.0,
        "kill": 0.5,
        "win": 5.0,
        "death": -2.0,
        "placement": 1.0,
        "survive": 0.0,
        "lowTroops": -0.05,
        "lowTroopsBelow": 0.3,
        "betrayal": -1.0,
        "diplomacy": -0.005,
        "advantage": 0.5,
        "defend": 0.3,
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
    # Mixed training: each scenario gets a share of the workers (by weight)
    # and its own env overrides; see configs/main.yaml. Exclusive with
    # curriculum.
    "scenarios": [],
    "model": {"trunk": 256, "slot_dim": 64, "site_dim": 32, "context": 128},
}

CURRICULUM_ENV_KEYS = {"bots", "nations", "seats", "tick_cap", "map"}
SCENARIO_ENV_KEYS = {"bots", "nations", "seats", "tick_cap", "map", "close_spawn"}


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
    for key in ("troop_reserve", "close_spawn"):
        if not 0.0 <= float(env[key]) <= 1.0:
            raise ValueError(f"env.{key} must be 0..1")
    if not 0.0 <= float(cfg["selfplay"]["learner_seat_prob"]) <= 1.0:
        raise ValueError("selfplay.learner_seat_prob must be 0..1")
    if int(ppo["minibatch"]) < 1 or int(ppo["n_steps"]) < 1 or int(ppo["epochs"]) < 1:
        raise ValueError("ppo.minibatch, n_steps, epochs must be >= 1")
    if cfg["scenarios"] and cfg["curriculum"]:
        raise ValueError("use scenarios or curriculum, not both")
    for i, sc in enumerate(cfg["scenarios"]):
        extra = set(sc) - SCENARIO_ENV_KEYS - {"name", "weight"}
        if extra or "name" not in sc:
            raise ValueError(f"scenarios[{i}] needs a name and only {sorted(SCENARIO_ENV_KEYS | {'weight'})}")
        if float(sc.get("weight", 1)) <= 0:
            raise ValueError(f"scenarios[{i}].weight must be > 0")
        if not 1 <= int(sc.get("seats", env["seats"])) <= 8:
            raise ValueError(f"scenarios[{i}].seats must be 1..8")
    if cfg["scenarios"] and int(env["workers"]) < len(cfg["scenarios"]):
        raise ValueError("need at least one worker per scenario")
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


def scenario_workers(cfg: dict) -> list[tuple[str, dict]]:
    """(scenario name, env) per worker: workers split by weight, each
    scenario getting at least one (largest remainder)."""
    env = cfg["env"]
    scenarios = cfg["scenarios"]
    workers = int(env["workers"])
    if not scenarios:
        return [("default", copy.deepcopy(env))] * workers
    weights = [float(s.get("weight", 1)) for s in scenarios]
    total = sum(weights)
    share = [1 + (workers - len(scenarios)) * w / total for w in weights]
    counts = [int(x) for x in share]
    for i in sorted(range(len(share)), key=lambda i: share[i] - counts[i], reverse=True)[: workers - sum(counts)]:
        counts[i] += 1
    out = []
    for sc, n in zip(scenarios, counts):
        merged = {**copy.deepcopy(env), **{k: v for k, v in sc.items() if k in SCENARIO_ENV_KEYS}}
        out += [(str(sc["name"]), merged)] * n
    return out


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
        allow_donations=bool(env.get("allow_donations", False)),
        allow_betrayal=bool(env.get("allow_betrayal", False)),
        troop_reserve=float(env.get("troop_reserve", 0.1)),
        close_spawn=float(env.get("close_spawn", 0.0)),
        reward={k: float(v) for k, v in reward.items()},
    )
