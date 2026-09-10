"""Train MaskablePPO against the headless OpenFront bridge.

Run from ``train/``::

    python train_ppo.py --config configs/default.yaml

Reward is the phase 1 default and stays untouched here. Reward tuning is
a later phase. What this script owns is honest accounting: every run logs
return, episode length, losses, steps per second wall clock, and the full
terminal reason histogram with sim error and worker death next to return,
so a rising return built on a rising error rate cannot pass unnoticed.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch as th
import yaml
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from openfront_gym.vec import OpenFrontVecEnv
from action_filter import behavior_from_config
from openfront_policy import OpenFrontPolicy
from sb3_adapter import SB3OpenFrontVec

TERMINAL_KEYS = ("win", "death", "tick_cap", "sim_error", "worker_death")


class TerminalCounterCallback(BaseCallback):
    """Count terminal reasons per rollout and log steps per second."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.totals: Counter[str] = Counter()
        self._rollout_counts: Counter[str] = Counter()
        self._rollout_start = time.monotonic()
        self._rollout_steps = 0

    def _on_rollout_start(self) -> None:
        self._rollout_counts = Counter()
        self._rollout_start = time.monotonic()
        self._rollout_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            reason = info.get("terminal_reason") or "continuing"
            self._rollout_counts[reason] += 1
            self.totals[str(reason)] += 1
        self._rollout_steps += self.training_env.num_envs
        return True

    def _on_rollout_end(self) -> None:
        elapsed = max(time.monotonic() - self._rollout_start, 1e-6)
        total = sum(self._rollout_counts.values())
        self.logger.record("time/steps_per_second", self._rollout_steps / elapsed)
        for key in TERMINAL_KEYS:
            count = self._rollout_counts.get(key, 0)
            self.logger.record(f"terminals/{key}", count / total if total else 0.0)
        self.logger.record("terminals/episodes", total)


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if th.cuda.is_available() else "cpu"
    return name


def build_vec(cfg: dict) -> SB3OpenFrontVec:
    env_cfg = cfg["env"]
    behavior = behavior_from_config(cfg)
    inner = OpenFrontVecEnv(
        num_workers=int(env_cfg.get("num_workers", 1)),
        envs_per_worker=int(env_cfg.get("envs_per_worker", 1)),
        map=str(env_cfg.get("map", "World")),
        bots=int(env_cfg.get("bots", 20)),
        ticks_per_step=int(env_cfg.get("ticks_per_step", 10)),
        tick_cap=int(env_cfg.get("tick_cap", 6000)),
        maps_dir=env_cfg.get("maps_dir"),
        strict_illegal_actions=bool(env_cfg.get("strict_illegal_actions", True)),
        socket_timeout=float(env_cfg.get("socket_timeout", 180.0)),
        seeds=None,
        parallel=bool(env_cfg.get("parallel_workers", True)),
        **behavior,
    )
    vec = SB3OpenFrontVec(inner)
    if env_cfg.get("seed", None) is not None:
        vec.seed(int(env_cfg["seed"]))
    return vec


def build_model(vec: SB3OpenFrontVec, cfg: dict, tensorboard_log: str | None):
    ppo = cfg["ppo"]
    n_steps = int(ppo.get("n_steps", 128))
    batch_size = int(ppo.get("batch_size", 512))
    if (n_steps * vec.num_envs) % batch_size != 0:
        raise ValueError(
            f"n_steps * num_envs ({n_steps * vec.num_envs}) must divide "
            f"batch_size ({batch_size}) or SB3 fails cryptically"
        )
    policy_cfg = cfg.get("policy", {})
    policy_kwargs = {
        "features_extractor_kwargs": {
            "features_dim": int(policy_cfg.get("features_dim", 512)),
        },
        "net_arch": dict(
            pi=[int(v) for v in policy_cfg.get("net_arch_pi", [256])],
            vf=[int(v) for v in policy_cfg.get("net_arch_vf", [256])],
        ),
    }
    return MaskablePPO(
        OpenFrontPolicy,
        vec,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=int(cfg["env"].get("seed", 0)),
        device=resolve_device(str(cfg["run"].get("device", "auto"))),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(ppo.get("n_epochs", 4)),
        gamma=float(ppo.get("gamma", 0.99)),
        gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        clip_range=float(ppo.get("clip_range", 0.2)),
        ent_coef=float(ppo.get("ent_coef", 0.01)),
        vf_coef=float(ppo.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        learning_rate=float(ppo.get("learning_rate", 0.0003)),
        policy_kwargs=policy_kwargs,
    )


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on OpenFront.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run = cfg["run"]
    total = args.total_timesteps if args.total_timesteps is not None else int(run.get("total_timesteps", 1_000_000))
    run_name = args.run_name or str(run.get("run_name", "ppo"))
    if args.device is not None:
        run["device"] = args.device
    checkpoint_dir = Path(str(run.get("checkpoint_dir", "checkpoints/openfront_ppo")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tb_log = run.get("tensorboard_log")

    vec = build_vec(cfg)
    print(f"envs: {vec.num_envs}, obs: {vec.observation_space.shape}, device: {resolve_device(str(run.get('device', 'auto')))}")
    model = build_model(vec, cfg, tb_log)
    terminal_cb = TerminalCounterCallback()
    checkpoint_cb = CheckpointCallback(
        save_freq=max(int(run.get("checkpoint_freq", 50000)) // vec.num_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix=run_name,
    )
    started = time.monotonic()
    model.learn(
        total_timesteps=total,
        callback=[terminal_cb, checkpoint_cb],
        tb_log_name=run_name,
    )
    elapsed = time.monotonic() - started
    final_path = checkpoint_dir / f"{run_name}_final.zip"
    model.save(str(final_path.with_suffix("")))
    print(f"saved {final_path}")
    print(f"timesteps: {total} in {elapsed:.1f}s = {total / max(elapsed, 1e-6):.0f} steps/sec")
    print(f"terminals: {dict(terminal_cb.totals)}")
    vec.close()
    return {"model_path": str(final_path), "terminals": dict(terminal_cb.totals)}


if __name__ == "__main__":
    main()
