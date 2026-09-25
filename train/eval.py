"""Evaluate a MaskablePPO checkpoint against scripted reference behavior.

Run from ``train/``::

    python eval.py --model checkpoints/openfront_ppo/ppo_default_final.zip --episodes 10

Eval is deterministic by default on fixed seeds so two runs compare
directly. With ``--baseline`` the scripted agent plays the same seeds and
both rows print side by side. Terminal reasons are always reported, not
just return, because a policy that dies fast with high shaping reward
reads well on return alone and badly everywhere else.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from sb3_contrib import MaskablePPO

from agents.scripted_agent import ScriptedBaselineAgent
from action_filter import apply_aggression_batch, behavior_from_config
from openfront_gym.vec import OpenFrontVecEnv
from openfront_gym.normalize import ObservationNormalizer
from openfront_policy import OpenFrontPolicy  # noqa: F401, registers the custom policy for load


def run_episodes(
    predict,
    vec: OpenFrontVecEnv,
    episodes: int,
    seeds: list[str],
    deterministic: bool = True,
    max_steps: int = 1500,
) -> dict:
    if vec.num_envs != 1:
        raise ValueError("run_episodes tracks one env, pass a 1x1 vec")
    from action_filter import INTENT_NAMES
    from openfront_gym import spaces

    returns: list[float] = []
    lengths: list[int] = []
    reasons: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    attack_ratios: Counter[float] = Counter()
    attack_open_steps = 0
    total_steps = 0
    seed_cycle = [seeds[i % len(seeds)] for i in range(episodes)]
    for episode in range(episodes):
        obs, infos = vec.reset(seed=[seed_cycle[episode]] * vec.num_envs)
        masks = infos["action_masks"]
        done = np.zeros(vec.num_envs, dtype=bool)
        ep_return = np.zeros(vec.num_envs)
        ep_len = np.zeros(vec.num_envs, dtype=int)
        steps = 0
        while not done[0]:
            if steps >= max_steps:
                reasons["timeout"] += 1
                break
            actions = predict(obs, masks, deterministic)
            act = np.asarray(actions)[0]
            intent_counts[INTENT_NAMES[int(act[0])]] += 1
            if int(act[0]) == spaces.IntentType.ATTACK:
                from action_filter import RATIO_BUCKETS

                attack_ratios[float(RATIO_BUCKETS[int(act[2])])] += 1
            heads = spaces.split_flat_mask(np.asarray(masks)[0])
            if heads["intent_type"][spaces.IntentType.ATTACK]:
                attack_open_steps += 1
            total_steps += 1
            obs, rewards, terminated, truncated, infos = vec.step(actions)
            masks = infos["action_masks"]
            steps += 1
            active = ~done
            ep_return[active] += np.asarray(rewards)[active]
            ep_len[active] += 1
            finished = np.asarray(terminated) | np.asarray(truncated)
            for index in np.flatnonzero(finished & active):
                reason = infos["terminal_reason"][index]
                reasons[str(reason)] += 1
            done = done | finished
        returns.append(float(ep_return[0]))
        lengths.append(int(ep_len[0]))
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "win_rate": reasons.get("win", 0) / episodes,
        "terminals": dict(reasons),
        "intent_share": {
            name: intent_counts.get(name, 0) / total_steps for name in INTENT_NAMES
        },
        "attack_ratio_mix": dict(attack_ratios),
        "attack_open_share": attack_open_steps / total_steps if total_steps else 0.0,
    }


def policy_predictor(model: MaskablePPO, min_attack_ratio: int = 0):
    def predict(obs, masks, deterministic: bool):
        actions, _ = model.predict(np.asarray(obs), deterministic=deterministic, action_masks=np.asarray(masks))
        return apply_aggression_batch(np.asarray(actions), min_attack_ratio)

    return predict


def baseline_predictor(min_attack_ratio: int = 0):
    agent = ScriptedBaselineAgent()
    return lambda obs, masks, deterministic: apply_aggression_batch(
        agent.act_batch(np.asarray(obs), np.asarray(masks)), min_attack_ratio
    )


def build_vec(cfg: dict) -> OpenFrontVecEnv:
    env_cfg = cfg.get("env", {})
    behavior = behavior_from_config(cfg)
    vec = OpenFrontVecEnv(
        1,
        1,
        map=str(env_cfg.get("map", "World")),
        bots=int(env_cfg.get("bots", 20)),
        nations=str(env_cfg.get("nations", "default")),
        ticks_per_step=int(env_cfg.get("ticks_per_step", 10)),
        tick_cap=int(env_cfg.get("tick_cap", 6000)),
        maps_dir=env_cfg.get("maps_dir"),
        strict_illegal_actions=bool(env_cfg.get("strict_illegal_actions", True)),
        socket_timeout=float(env_cfg.get("socket_timeout", 180.0)),
        seeds=None,
        parallel=bool(env_cfg.get("parallel_workers", True)),
        **behavior,
    )
    obs_norm = (cfg.get("run", {}) or {}).get("obs_norm", None)
    if obs_norm:
        frozen = ObservationNormalizer(vec, training=False)
        frozen.load(str(obs_norm), vec.single_observation_space.shape[0])
        return frozen
    return vec


def evaluate(
    model_path: str,
    episodes: int,
    seeds: list[str],
    env_cfg: dict,
    device: str = "cpu",
    with_baseline: bool = False,
    deterministic: bool = True,
    min_attack_ratio: int = 0,
    behavior_cfg: dict | None = None,
    obs_norm: str | None = None,
) -> dict:
    model = MaskablePPO.load(model_path, device=device)
    full_cfg = {"env": env_cfg, "behavior": behavior_cfg or {}}
    if obs_norm is not None:
        full_cfg["run"] = {"obs_norm": obs_norm}
    vec = build_vec(full_cfg)
    try:
        stats = run_episodes(
            policy_predictor(model, min_attack_ratio), vec, episodes, seeds, deterministic
        )
    finally:
        vec.close()
    result = {"policy": stats}
    if with_baseline:
        baseline_vec = build_vec(full_cfg)
        try:
            result["baseline"] = run_episodes(
                baseline_predictor(min_attack_ratio), baseline_vec, episodes, seeds, True
            )
        finally:
            baseline_vec.close()
    return result


def main(argv: list[str] | None = None) -> dict:
    import os
    import yaml

    parser = argparse.ArgumentParser(description="Evaluate an OpenFront PPO checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", default="eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--min-attack-ratio", type=int, default=None)
    parser.add_argument(
        "--obs-norm",
        default=None,
        help="frozen observation stats (.npz saved next to a normalized checkpoint)",
    )
    parser.add_argument(
        "--replay-dir",
        default=None,
        help="record every episode here as a phase-8 replay file",
    )
    parser.add_argument("--replay-every", type=int, default=1)
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    raw_behavior = cfg.get("behavior", {}) or {}
    min_ratio = args.min_attack_ratio
    if min_ratio is None:
        # A preset may pin its own floor; otherwise fall back to the
        # aggression mapping when the config names one.
        min_ratio = raw_behavior.get("min_attack_ratio", None)
    if min_ratio is None:
        from action_filter import aggression_to_ratio_floor

        min_ratio = aggression_to_ratio_floor(int(raw_behavior.get("aggression", 50)))
    min_ratio = int(min_ratio)
    if not 0 <= min_ratio < 8:
        parser.error("--min-attack-ratio must be 0..7")
    if args.replay_dir is not None:
        if args.replay_every < 1:
            parser.error("--replay-every must be positive")
        # Workers run with env/ as cwd, so resolve here or files land there.
        from pathlib import Path

        replay_dir = Path(args.replay_dir)
        if not replay_dir.is_absolute():
            replay_dir = Path(__file__).resolve().parent / replay_dir
        os.environ["OPENFRONT_REPLAY_DIR"] = str(replay_dir)
        os.environ["OPENFRONT_REPLAY_EVERY"] = str(args.replay_every)
    else:
        os.environ.pop("OPENFRONT_REPLAY_DIR", None)
        os.environ.pop("OPENFRONT_REPLAY_EVERY", None)
    seeds = [f"{args.seed}-{i}" for i in range(args.episodes)]
    result = evaluate(
        args.model,
        args.episodes,
        seeds,
        cfg.get("env", {}),
        device=args.device,
        with_baseline=args.baseline,
        deterministic=not args.stochastic,
        min_attack_ratio=min_ratio,
        behavior_cfg=raw_behavior,
        obs_norm=args.obs_norm,
    )
    for name, stats in result.items():
        print(
            f"{name}: return {stats['mean_return']:.3f}, "
            f"length {stats['mean_length']:.0f}, "
            f"win rate {stats['win_rate']:.2f}, "
            f"terminals {stats['terminals']}"
        )
        top = sorted(stats["intent_share"].items(), key=lambda kv: -kv[1])[:5]
        print(
            f"{name}: top intents "
            + ", ".join(f"{k} {v:.2f}" for k, v in top)
            + f", attack open {stats['attack_open_share']:.2f}"
        )
        if stats["attack_ratio_mix"]:
            print(f"{name}: attack ratios {stats['attack_ratio_mix']}")
    return result


if __name__ == "__main__":
    main()
