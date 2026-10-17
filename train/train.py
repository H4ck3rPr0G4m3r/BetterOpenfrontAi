"""Train the OpenFront policy with PPO and in-game self-play.

Run from ``train/``::

    python train.py --config configs/selfplay.yaml
    python train.py --config configs/selfplay.yaml --resume checkpoints/selfplay/selfplay_latest.pt
    python train.py --config configs/smoke.yaml --set run.total_steps=2000

Stop gracefully (checkpoint saved) by creating ``checkpoints/<name>/STOP``
or pressing Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from openfront.config import load_config
from openfront.ppo import Trainer


def parse_set(items: list[str]) -> dict:
    """``a.b=value`` pairs into a nested override dict (values parsed as YAML)."""
    out: dict = {}
    for item in items:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set needs key=value, got {item!r}")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/selfplay.yaml")
    parser.add_argument("--resume", nargs="?", const="latest", default=None,
                        help="checkpoint (.pt) to continue from; bare --resume = this run's latest")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, parse_set(args.set))
    if args.resume == "latest":
        run = cfg["run"]
        latest = Path(run["checkpoint_dir"]) / run["name"] / f"{run['name']}_latest.pt"
        if not latest.is_file():
            raise SystemExit(f"--resume: no checkpoint at {latest} to continue from")
        args.resume = str(latest)
        print(f"resuming from {latest}", flush=True)
    trainer = Trainer(cfg, resume=args.resume, log=lambda msg: print(msg, flush=True))
    try:
        result = trainer.train()
    except KeyboardInterrupt:
        print("interrupted; final checkpoint saved", file=sys.stderr)
        return {}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
