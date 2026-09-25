"""League-style self-play driver: curriculum stages plus an opponent pool.

Runs training in segments, snapshots every segment's checkpoint into the
pool, and evaluates the learner against both stage bots and sampled pool
snapshots on the SAME fixed seeds. Promotion to the next curriculum stage
requires beating both (see ``selfplay/curriculum.py``).

Honest scope note: the bridge seats one learner per env, so
learner-vs-pool comparison is same-seed (separate episodes, identical
seeds), not same-game. The pool file format (plain checkpoint zips plus
``index.json``) is forward-compatible with a future in-game opponent
seat; when that lands, ``sample()`` feeds the opponent seat directly.

No long runs happen here by default: ``--segments 0`` validates configs
and exits. Real training runs leave this machine for hours; this driver
only orchestrates them.

Run from ``train/``::

    python train_sp.py --stage stage1 --segments 5 --steps-per-segment 100000
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from selfplay.curriculum import next_stage, should_promote, stage_by_name
from selfplay.pool import OpponentPool, selfplay_ready


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_segment_config(stage_cfg: dict, workdir: Path, stage: str, segment: int, steps: int) -> Path:
    """Stage config with run dirs pinned under the self-play workdir."""
    cfg = copy.deepcopy(stage_cfg)
    run = cfg.setdefault("run", {})
    run["total_timesteps"] = int(steps)
    run["checkpoint_dir"] = str(workdir / "checkpoints" / stage / f"seg{segment}")
    run["tensorboard_log"] = str(workdir / "runs" / stage / f"seg{segment}")
    run["run_name"] = f"{stage}_seg{segment}"
    out = workdir / "segment_configs" / f"{stage}_seg{segment}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    return out


def matchup_won(learner: dict, opponent: dict) -> bool:
    """Same-seed matchup: learner beats a pool snapshot on win rate first,
    mean return as the tiebreak. Both rows come from identical seeds, so the
    comparison is apples to apples even though the episodes are separate."""
    if float(learner.get("win_rate", 0.0)) != float(opponent.get("win_rate", 0.0)):
        return float(learner.get("win_rate", 0.0)) > float(opponent.get("win_rate", 0.0))
    return float(learner.get("mean_return", 0.0)) > float(opponent.get("mean_return", 0.0))


def run_segment(
    stage_cfg: dict,
    workdir: Path,
    pool: OpponentPool,
    stage: str,
    segment: int,
    steps: int,
    episodes: int,
    seed: str,
    device: str,
    resume: str | None,
    sample_strategy: str,
    pool_matches: int,
) -> dict:
    """Train one segment, snapshot it, evaluate learner vs bots and pool."""
    from eval import evaluate
    from train_ppo import main as train_main

    seg_config = write_segment_config(stage_cfg, workdir, stage, segment, steps)
    args = ["--config", str(seg_config)]
    if resume is not None:
        args += ["--resume", str(resume)]
    if device is not None:
        args += ["--device", str(device)]
    trained = train_main(args)
    model_path = trained["model_path"]

    total_steps = (segment + 1) * steps
    pool.add_snapshot(model_path, total_steps)

    seeds = [f"{seed}-{i}" for i in range(episodes)]
    env_cfg = stage_cfg.get("env", {})
    behavior_cfg = stage_cfg.get("behavior", {}) or {}
    learner = evaluate(
        model_path, episodes, seeds, env_cfg, device="cpu",
        deterministic=True, behavior_cfg=behavior_cfg,
    )["policy"]

    pool_win_rate: float | None = None
    pool_results: list[dict] = []
    if selfplay_ready(learner.get("win_rate", 0.0)):
        import random

        rng = random.Random(f"{seed}-{stage}-{segment}")
        wins = 0
        for _ in range(max(int(pool_matches), 1)):
            opponent = pool.sample(strategy=sample_strategy, rng=rng)
            if opponent is None:
                break
            opp_stats = evaluate(
                opponent.path, episodes, seeds, env_cfg, device="cpu",
                deterministic=True, behavior_cfg=behavior_cfg,
            )["policy"]
            won = matchup_won(learner, opp_stats)
            pool.record_result(opponent, won)
            wins += 1 if won else 0
            pool_results.append(
                {"opponent_step": opponent.step, "won": won,
                 "learner_win_rate": learner.get("win_rate", 0.0),
                 "opponent_win_rate": opp_stats.get("win_rate", 0.0)}
            )
        if pool_results:
            pool_win_rate = wins / len(pool_results)

    return {
        "model_path": model_path,
        "learner": learner,
        "pool_active": pool_win_rate is not None,
        "pool_win_rate": pool_win_rate,
        "pool_results": pool_results,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Curriculum + pool self-play driver.")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--segments", type=int, default=0)
    parser.add_argument("--steps-per-segment", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", default="sp")
    parser.add_argument("--device", default=None)
    parser.add_argument("--pool-dir", default="selfplay/pool")
    parser.add_argument("--workdir", default="selfplay")
    parser.add_argument("--sample-strategy", default="last_k")
    parser.add_argument("--pool-matches", type=int, default=2)
    args = parser.parse_args(argv)

    stage = stage_by_name(args.stage)
    stage_cfg = load_yaml(Path(stage.config))
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pool = OpponentPool.load(Path(args.pool_dir))

    state_path = workdir / "selfplay_state.json"
    state = {"stage": stage.name, "segments": [], "resume": None}
    if args.segments <= 0:
        print(f"validated stage {stage.name} ({stage.config}); segments=0, nothing to run")
        return state

    current = stage.name
    resume = None
    for segment in range(args.segments):
        current_stage = stage_by_name(current)
        cfg = load_yaml(Path(current_stage.config)) if current != stage.name else stage_cfg
        result = run_segment(
            cfg, workdir, pool, current, segment, args.steps_per_segment,
            args.episodes, args.seed, args.device, resume,
            args.sample_strategy, args.pool_matches,
        )
        resume = result["model_path"]
        entry = {"stage": current, "segment": segment,
                 "learner_win_rate": result["learner"].get("win_rate", 0.0),
                 "pool_active": result["pool_active"],
                 "pool_win_rate": result["pool_win_rate"]}
        state["segments"].append(entry)
        state["resume"] = resume
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[{current} seg{segment}] learner win {entry['learner_win_rate']:.2f} "
              f"pool_active={entry['pool_active']} pool_win={entry['pool_win_rate']}")
        if should_promote(current_stage, result["learner"].get("win_rate", 0.0),
                           args.episodes, result["pool_win_rate"]):
            nxt = next_stage(current)
            if nxt is None:
                print(f"[{current}] final stage cleared")
                break
            print(f"[{current}] promoted to {nxt}")
            current = nxt
    state["stage"] = current
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


if __name__ == "__main__":
    main()
