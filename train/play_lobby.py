"""Put an AI into an OpenFront lobby by ID: AI vs Human locally, or a real openfront.io lobby.

Run from ``train/``::

    # AI vs Human: start the local server (OpenFrontIO: npm run dev), open
    # http://localhost:9000, create a private lobby, copy its ID, then:
    python play_lobby.py AbCd1234 --model checkpoints/selfplay_v3/selfplay_v3_latest.pt
    # ...start the game from your lobby once the AI shows up in the player list.

    # A real openfront.io lobby someone shared with you (ID or link):
    python play_lobby.py https://openfront.io/w2/game/AbCd1234Ef --model <checkpoint.pt>

    # The next public openfront.io lobby (ffa | team | special | any), or list them:
    python play_lobby.py --public ffa --model <checkpoint.pt>
    python play_lobby.py --list

A visible browser window opens and joins as a named player. Leave it in the
foreground area of your screen; answer any check the site shows yourself.
Several AIs in one lobby: run the command again with another ``--profile``.
"""

from __future__ import annotations

import argparse
import json
import sys

from openfront.lobby import HOSTS, PUBLIC_KINDS, LobbyPlayer, list_public_lobbies, resolve_lobby


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lobby", nargs="?", help="lobby ID (6-16 letters/digits) or a lobby link")
    parser.add_argument("--host", default=None,
                        help=f"{' | '.join(HOSTS)} | a base URL (ignored for links; default local, "
                             "openfront with --public/--list)")
    parser.add_argument("--public", choices=(*PUBLIC_KINDS, "any"),
                        help="join the next public lobby of this kind instead of a given ID")
    parser.add_argument("--list", action="store_true", help="print the public lobbies as JSON and exit")
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

    if args.list or args.public:
        host = args.host or "openfront"
        base = HOSTS.get(host, host).rstrip("/")
    if args.list:
        rows = list_public_lobbies(base, profile=args.profile, browser=args.browser,
                                   log=lambda m: print(m, file=sys.stderr))
        print(json.dumps({"lobbies": rows}))
        return
    if args.public:
        lobby = None
    elif args.lobby:
        base, lobby = resolve_lobby(args.lobby, args.host or "local")
    else:
        parser.error("give a lobby ID or link, --public KIND, or --list")

    from openfront.inference import Predictor

    predictor = Predictor(args.model, device=args.device, deterministic=args.deterministic)
    target = f"lobby {lobby}" if lobby else f"next public {args.public} lobby"
    print(f"AI: {predictor.label} -> {base} {target}")
    LobbyPlayer(predictor, base, lobby, name=args.name, profile=args.profile,
                interval=args.interval, browser=args.browser, headless=args.headless,
                max_seconds=args.max_seconds, public=args.public).run()


if __name__ == "__main__":
    main()
