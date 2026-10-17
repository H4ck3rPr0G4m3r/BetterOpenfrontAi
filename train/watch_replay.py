"""Open a recorded game (train/replays/...) in the real game client.

Plays in the local client (``npm run dev`` in OpenFrontIO/); if nothing is
serving http://localhost:9000 yet, it is started in its own window and left
running for the next replay. A browser window opens with the replay loaded;
use the client's own replay speed controls. Closing the window ends the
command.

Run from ``train/``::

    python watch_replay.py replays/selfplay_v3/game0-...-alive.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from openfront.lobby import PROFILES

CLIENT_DIR = Path(__file__).resolve().parents[1] / "OpenFrontIO"
START_SECONDS = 240


def client_up(base: str) -> bool:
    try:
        with urllib.request.urlopen(base, timeout=3):
            return True
    except Exception:
        return False


def ensure_client(base: str) -> bool:
    """Start the local game client when nothing serves ``base``. True when
    it had to be started (its first page load is slow)."""
    if client_up(base):
        return False
    if urllib.parse.urlparse(base).hostname not in ("localhost", "127.0.0.1"):
        raise SystemExit(f"cannot reach {base}")
    if not (CLIENT_DIR / "package.json").is_file():
        raise SystemExit(f"no game client at {CLIENT_DIR}")
    print("the local game server is not running; starting it in its own window (npm run dev)...", flush=True)
    if sys.platform == "win32":
        # Through `start`, so it outlives this command like start.bat's does.
        subprocess.Popen(f'start "OpenFront game server" /d "{CLIENT_DIR}" cmd /k npm run dev', shell=True)
    else:
        subprocess.Popen(["npm", "run", "dev"], cwd=CLIENT_DIR, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + START_SECONDS
    while not client_up(base):
        if time.monotonic() > deadline:
            raise SystemExit(f"the game server did not come up on {base} within {START_SECONDS}s; "
                             "check its window for errors")
        time.sleep(2)
    print("game server is up", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", help="a replay JSON written by training or evaluate.py --replay-dir")
    parser.add_argument("--base", default="http://localhost:9000")
    parser.add_argument("--browser", default="msedge", choices=("msedge", "chrome", "chromium"))
    args = parser.parse_args()

    record = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or "info" not in record or not isinstance(record.get("turns"), list):
        raise SystemExit(f"{args.replay} is not a replay file (needs info and turns)")

    started = ensure_client(args.base)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        channel = None if args.browser == "chromium" else args.browser
        context = pw.chromium.launch_persistent_context(
            str(PROFILES / "replay"), channel=channel, headless=False,
            viewport={"width": 1400, "height": 860}, args=["--mute-audio"],
            chromium_sandbox=True)  # Playwright adds --no-sandbox otherwise
        page = context.pages[0] if context.pages else context.new_page()
        try:
            page.goto(args.base, wait_until="domcontentloaded")
        except Exception as err:
            raise SystemExit(f"cannot open {args.base} ({err})")
        # A dev server's first load compiles the client on demand: give it longer.
        page.wait_for_function("() => customElements.get('join-lobby-modal') !== undefined",
                               timeout=(180 if started else 60) * 1000)
        page.wait_for_timeout(1500)  # let the client attach its join-lobby listener
        page.evaluate(
            """(record) => document.dispatchEvent(new CustomEvent("join-lobby", {
                 detail: { gameID: record.info.gameID, gameRecord: record, source: "private" } }))""",
            record,
        )
        print(f"playing {Path(args.replay).name}; close the window to finish")
        page.wait_for_event("close", timeout=0)
        context.close()


if __name__ == "__main__":
    main()
