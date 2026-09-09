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
from openfront_gym.vec import OpenFrontVecEnv
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
    returns: list[float] = []
    lengths: list[int] = []
    reasons: Counter[str] = Counter()
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
    }


def policy_predictor(model: MaskablePPO):
    def predict(obs, masks, deterministic: bool):
        actions, _ = model.predict(np.asarray(obs), deterministic=deterministic, action_masks=np.asarray(masks))
        return np.asarray(actions)

    return predict


def baseline_predictor() -> ScriptedBaselineAgent:
    agent = ScriptedBaselineAgent()
    return lambda obs, masks, deterministic: agent.act_batch(np.asarray(obs), np.asarray(masks))


def build_vec(cfg: dict) -> OpenFrontVecEnv:
    env_cfg = cfg.get("env", {})
    return OpenFrontVecEnv(
        1,
        1,
        map=str(env_cfg.get("map", "World")),
        bots=int(env_cfg.get("bots", 20)),
        ticks_per_step=int(env_cfg.get("ticks_per_step", 10)),
        tick_cap=int(env_cfg.get("tick_cap", 6000)),
        maps_dir=env_cfg.get("maps_dir"),
        strict_illegal_actions=bool(env_cfg.get("strict_illegal_actions", True)),
        socket_timeout=float(env_cfg.get("socket_timeout", 180.0)),
        seeds=None,
    )


def evaluate(
    model_path: str,
    episodes: int,
    seeds: list[str],
    env_cfg: dict,
    device: str = "cpu",
    with_baseline: bool = False,
    deterministic: bool = True,
) -> dict:
    model = MaskablePPO.load(model_path, device=device)
    vec = build_vec({"env": env_cfg})
    try:
        stats = run_episodes(policy_predictor(model), vec, episodes, seeds, deterministic)
    finally:
        vec.close()
    result = {"policy": stats}
    if with_baseline:
        baseline_vec = build_vec({"env": env_cfg})
        try:
            result["baseline"] = run_episodes(
                baseline_predictor(), baseline_vec, episodes, seeds, True
            )
        finally:
            baseline_vec.close()
    return result


def main(argv: list[str] | None = None) -> dict:
    import yaml

    parser = argparse.ArgumentParser(description="Evaluate an OpenFront PPO checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", default="eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    seeds = [f"{args.seed}-{i}" for i in range(args.episodes)]
    result = evaluate(
        args.model,
        args.episodes,
        seeds,
        cfg.get("env", {}),
        device=args.device,
        with_baseline=args.baseline,
        deterministic=not args.stochastic,
    )
    for name, stats in result.items():
        print(
            f"{name}: return {stats['mean_return']:.3f}, "
            f"length {stats['mean_length']:.0f}, "
            f"win rate {stats['win_rate']:.2f}, "
            f"terminals {stats['terminals']}"
        )
    return result


if __name__ == "__main__":
    main()
