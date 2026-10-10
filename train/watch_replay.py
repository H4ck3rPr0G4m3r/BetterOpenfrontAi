"""Open a recorded game (train/replays/...) in the real game client.

Needs the local client running (``npm run dev`` in OpenFrontIO/). A browser
window opens on http://localhost:9000 with the replay loaded; use the
client's own replay speed controls. Closing the window ends the command.

Run from ``train/``::

    python watch_replay.py replays/selfplay_v3/game0-...-alive.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openfront.lobby import PROFILES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", help="a replay JSON written by training or evaluate.py --replay-dir")
    parser.add_argument("--base", default="http://localhost:9000")
    parser.add_argument("--browser", default="msedge", choices=("msedge", "chrome", "chromium"))
    args = parser.parse_args()

    record = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or "info" not in record or not isinstance(record.get("turns"), list):
        raise SystemExit(f"{args.replay} is not a replay file (needs info and turns)")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        channel = None if args.browser == "chromium" else args.browser
        context = pw.chromium.launch_persistent_context(
            str(PROFILES / "replay"), channel=channel, headless=False,
            viewport={"width": 1400, "height": 860}, args=["--mute-audio"])
        page = context.pages[0] if context.pages else context.new_page()
        try:
            page.goto(args.base, wait_until="domcontentloaded")
        except Exception as err:
            raise SystemExit(f"cannot open {args.base} ({err}); start the client with `npm run dev` in OpenFrontIO/")
        page.wait_for_function("() => customElements.get('join-lobby-modal') !== undefined", timeout=60_000)
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
