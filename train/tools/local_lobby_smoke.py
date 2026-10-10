"""End-to-end check of the lobby driver against the local OpenFront server.

Hosts a private lobby in a headless browser (standing in for the human),
starts ``play_lobby.py`` against it, starts the game once the AI is in the
player list, and lets the AI play for ``--seconds``. Passes when the AI
joined, spawned and sent actions. Needs ``npm run dev`` in OpenFrontIO/.

Run from ``train/``::

    python tools/local_lobby_smoke.py --model scripted --seconds 60
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="scripted")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--base", default="http://localhost:9000")
    parser.add_argument("--browser", default="msedge")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=args.browser, headless=True)
        host = browser.new_page()
        host.goto(args.base, wait_until="domcontentloaded")
        host.wait_for_function("() => customElements.get('host-lobby-modal') !== undefined", timeout=60_000)
        host.evaluate("() => document.querySelector('host-lobby-modal').open()")
        host.wait_for_function("() => document.querySelector('host-lobby-modal').lobbyId", timeout=30_000)
        lobby = host.evaluate("() => document.querySelector('host-lobby-modal').lobbyId")
        print(f"hosted lobby {lobby}")

        ai = subprocess.Popen(
            [sys.executable, "-u", "play_lobby.py", lobby, "--host", args.base, "--model", args.model,
             "--headless", "--max-seconds", str(args.seconds), "--browser", args.browser, "--profile", "smoke"],
            cwd=TRAIN, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        host.wait_for_function("() => document.querySelector('host-lobby-modal').clients.length >= 2", timeout=120_000)
        print("AI is in the lobby; starting the game")
        host.evaluate("() => document.querySelector('host-lobby-modal').toggleGameStartTimer()")
        output = []
        for line in ai.stdout:
            print(f"  ai | {line.rstrip()}")
            output.append(line)
        ai.wait()
        browser.close()

    text = "".join(output)
    checks = {
        "joined the game": "game started" in text,
        "spawned": "spawn at" in text,
        "sent actions": re.search(r"-> (?!nothing)", text) is not None,
        "exited cleanly": ai.returncode == 0,
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
