"""Evaluate policies, alone against bots or against each other in the same game.

Each ``--seats`` entry is a predictor (a v2 checkpoint ``.pt``, ``scripted``,
``random`` or ``noop``); they share every game, next to engine bots and
nations. Seat order rotates across games so nobody keeps the best spawn.

Run from ``train/``::

    # how does a checkpoint do alone against bots?
    python evaluate.py --seats checkpoints/selfplay/selfplay_latest.pt --games 10
    # AI vs AI: two checkpoints and the scripted yardstick in the same games
    python evaluate.py --seats a.pt b.pt scripted --games 12 --replay-dir replays/eval

Recorded replays play back in the real game client (see README).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from openfront import layout as L
from openfront.inference import Predictor
from openfront.vec import VecGames
from openfront.worker import WorkerConfig


def _survived(rec: dict) -> bool:
    return rec["terminal"] in ("tick_cap", "win")


def _rank(rec: dict) -> tuple:
    """Head-to-head order: survived, then lasted longer, then earned more."""
    return (_survived(rec), rec["length"], round(rec["return"], 3))


def evaluate(
    seat_specs: list[str],
    games: int = 10,
    bots: int = 20,
    nations: str = "default",
    tick_cap: int = 6000,
    map_name: str = "World",
    workers: int | None = None,
    seed: str = "eval",
    device: str = "cpu",
    deterministic: bool = False,
    reward: dict | None = None,
    replay_dir: str | None = None,
    close_spawn: float = 0.0,
    log=print,
    worker_factory=None,
) -> dict:
    seats = len(seat_specs)
    if not 1 <= seats <= 8:
        raise ValueError("need 1..8 seats")
    predictors = {spec: Predictor(spec, device=device, deterministic=deterministic) for spec in dict.fromkeys(seat_specs)}
    if replay_dir:
        os.environ["OPENFRONT_REPLAY_DIR"] = str(Path(replay_dir).resolve())
        os.environ["OPENFRONT_REPLAY_EVERY"] = "1"
    else:
        os.environ.pop("OPENFRONT_REPLAY_DIR", None)
    workers = max(1, min(games, workers or max(1, (os.cpu_count() or 2) // 2)))
    config = WorkerConfig(games=1, seats=seats, map=map_name, bots=bots, nations=nations, tick_cap=tick_cap,
                          close_spawn=close_spawn, reward=reward or {})
    factory = None if worker_factory is None else (lambda i: worker_factory(i, config))
    vec = VecGames(config, workers, seed=seed, worker_factory=factory)
    results: dict[str, list[dict]] = defaultdict(list)
    pairwise: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    game_index = list(range(workers))  # which evaluation game each vec game is playing
    seat_spec = lambda row: seat_specs[(row % seats + game_index[vec.game_of_row(row)]) % seats]
    try:
        batch = vec.reset()
        started, finished = workers, 0
        ep_ret = np.zeros(vec.num_rows)
        ep_len = np.zeros(vec.num_rows, np.int64)
        pending: dict[int, dict] = {}
        retired: set[int] = set()
        while finished < games:
            acting = (batch["flags"] & L.FLAG_NEXT) != 0
            live = [r for r in np.flatnonzero(acting) if vec.game_of_row(r) not in retired]
            actions = np.zeros((vec.num_rows, 5), np.int64)
            for spec, predictor in predictors.items():
                idx = [r for r in live if seat_spec(r) == spec]
                if idx:
                    actions[idx] = predictor(batch["raster"][idx], batch["features"][idx], batch["masks"][idx])
            batch = vec.step(actions)  # retired games idle on noops until the last game ends
            flags = batch["flags"]
            ep_ret += np.where(flags & L.FLAG_VALID, batch["reward"], 0.0)
            ep_len += (flags & L.FLAG_VALID) != 0
            for r in np.flatnonzero(((flags & L.FLAG_DONE) != 0) & ((flags & L.FLAG_VALID) != 0)):
                if vec.game_of_row(r) in retired:
                    continue
                stats = batch["stats"][r]
                pending[r] = {
                    "predictor": seat_spec(r),
                    "terminal": L.TERMINALS.get(int(batch["terminal"][r])),
                    "return": float(ep_ret[r]),
                    "length": int(ep_len[r]),
                    "tiles": float(stats[0]),
                    "kills": float(stats[2]),
                    "placement": float(stats[3]),
                }
            for g in sorted({vec.game_of_row(r) for r in np.flatnonzero(flags & L.FLAG_FRESH)} - retired):
                rows = list(vec.rows_of_game(g))
                game = [pending.pop(r) for r in rows if r in pending]
                for rec in game:
                    results[rec["predictor"]].append(rec)
                for a in game:
                    for b in game:
                        if a["predictor"] != b["predictor"]:
                            key = (a["predictor"], b["predictor"])
                            if _rank(a) != _rank(b):
                                pairwise[key][0] += int(_rank(a) > _rank(b))
                                pairwise[key][1] += 1
                finished += 1
                log(f"game {finished}/{games}: " + ", ".join(
                    f"{Path(rec['predictor']).stem} {rec['terminal']} ({rec['tiles']:.0f} tiles)" for rec in game))
                ep_ret[rows] = 0.0
                ep_len[rows] = 0
                if started < games:
                    game_index[g] = started
                    started += 1
                else:
                    retired.add(g)
    finally:
        vec.close()
    summary = {}
    for spec, recs in results.items():
        n = len(recs)
        summary[spec] = {
            "episodes": n,
            "survival_rate": sum(_survived(r) for r in recs) / n,
            "win_rate": sum(r["terminal"] == "win" for r in recs) / n,
            "death_rate": sum(r["terminal"] == "death" for r in recs) / n,
            "mean_return": float(np.mean([r["return"] for r in recs])),
            "mean_tiles": float(np.mean([r["tiles"] for r in recs])),
            "mean_placement": float(np.mean([r["placement"] for r in recs])),
            "mean_kills": float(np.mean([r["kills"] for r in recs])),
        }
    head_to_head = {f"{Path(a).stem} > {Path(b).stem}": w / n for (a, b), (w, n) in pairwise.items() if n}
    return {"predictors": summary, "head_to_head": head_to_head}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seats", nargs="+", required=True, help="predictor per seat: checkpoint .pt | scripted | random | noop")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--bots", type=int, default=20)
    parser.add_argument("--nations", default="default", choices=("default", "disabled"))
    parser.add_argument("--tick-cap", type=int, default=6000)
    parser.add_argument("--map", default="World")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", default="eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true", help="argmax instead of sampling")
    parser.add_argument("--replay-dir", default=None, help="record every game for playback in the client")
    parser.add_argument("--close-spawn", type=float, default=0.0, help="chance a game spawns its seats close together")
    parser.add_argument("--json", default=None, help="also write the result as JSON here")
    args = parser.parse_args(argv)
    result = evaluate(
        args.seats, games=args.games, bots=args.bots, nations=args.nations, tick_cap=args.tick_cap,
        map_name=args.map, workers=args.workers, seed=args.seed, device=args.device,
        deterministic=args.deterministic, replay_dir=args.replay_dir, close_spawn=args.close_spawn,
    )
    print(f"\n{'predictor':30} games  survive  win   return   tiles  place  kills")
    for spec, s in result["predictors"].items():
        print(f"{Path(spec).stem[:30]:30} {s['episodes']:5}  {s['survival_rate']:7.2f}  {s['win_rate']:4.2f}  "
              f"{s['mean_return']:6.2f}  {s['mean_tiles']:6.0f}  {s['mean_placement']:5.2f}  {s['mean_kills']:5.2f}")
    for pair, rate in result["head_to_head"].items():
        print(f"  {pair}: {rate:.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
