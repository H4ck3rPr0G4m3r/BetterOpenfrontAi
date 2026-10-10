"""Put an AI into an OpenFront lobby by ID: AI vs Human locally, or a real openfront.io lobby.

Run from ``train/``::

    # AI vs Human: start the local server (OpenFrontIO: npm run dev), open
    # http://localhost:9000, create a private lobby, copy its ID, then:
    python play_lobby.py AbCd1234 --model checkpoints/selfplay_v3/selfplay_v3_latest.pt
    # ...start the game from your lobby once the AI shows up in the player list.

    # A real openfront.io lobby someone shared with you (ID or link):
    python play_lobby.py https://openfront.io/w2/game/AbCd1234 --model <checkpoint.pt>

A visible browser window opens and joins as a named player. Leave it in the
foreground area of your screen; answer any check the site shows yourself.
Several AIs in one lobby: run the command again with another ``--profile``.
"""

from __future__ import annotations

import argparse

from openfront.inference import Predictor
from openfront.lobby import HOSTS, LobbyPlayer, resolve_lobby


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lobby", help="lobby ID (8 letters/digits) or a lobby link")
    parser.add_argument("--host", default="local", help=f"{' | '.join(HOSTS)} | a base URL (ignored for links)")
    parser.add_argument("--model", default="scripted", help="checkpoint .pt | scripted | random | noop")
    parser.add_argument("--name", default="OpenFrontAI", help="player name shown in the lobby")
    parser.add_argument("--profile", default="default", help="browser profile; use one per simultaneous AI")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between decisions (training: 1.0)")
    parser.add_argument("--browser", default="msedge", choices=("msedge", "chrome", "chromium"))
    parser.add_argument("--deterministic", action="store_true", help="argmax instead of sampling")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true", help="no window (local testing only)")
    parser.add_argument("--max-seconds", type=float, default=None, help="leave the game after this long")
    args = parser.parse_args(argv)

    base, lobby = resolve_lobby(args.lobby, args.host)
    predictor = Predictor(args.model, device=args.device, deterministic=args.deterministic)
    print(f"AI: {predictor.label} -> {base} lobby {lobby}")
    LobbyPlayer(predictor, base, lobby, name=args.name, profile=args.profile,
                interval=args.interval, browser=args.browser, headless=args.headless,
                max_seconds=args.max_seconds).run()


if __name__ == "__main__":
    main()
