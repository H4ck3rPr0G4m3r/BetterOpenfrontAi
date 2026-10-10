"""Drive a policy inside a real OpenFront client through Playwright.

One driver serves two modes:

* AI vs Human (local): run the OpenFront dev server (``npm run dev`` in
  OpenFrontIO/, http://localhost:9000), create a private lobby in your own
  browser, and give the lobby ID to ``play_lobby.py --host local``.
* AI in a real openfront.io lobby: ``play_lobby.py --host openfront <id>``
  with a lobby ID (or link) you were given.

The browser opens the lobby link like any player would, the in-page agent
(``env/dist/page_agent.js``, built by ``npm run build:page``) reads the
client's own game state through the training encoder, and intents leave
through the client's own handlers. The window is visible on purpose: the
bot plays as an ordinary, named player, and any human check the site shows
(Cloudflare Turnstile, consent dialogs) is for you to answer — nothing here
solves or hides from it.
"""

from __future__ import annotations

import base64
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import layout as L

ENV_ROOT = Path(__file__).resolve().parents[2] / "env"
BUNDLE = ENV_ROOT / "dist" / "page_agent.js"
PROFILES = Path(__file__).resolve().parents[1] / ".lobby_profiles"
HOSTS = {"local": "http://localhost:9000", "openfront": "https://openfront.io"}
GAME_ID = re.compile(r"^[A-Za-z0-9]{8}$")
PAGE_AGENT_VERSION = 2


def resolve_lobby(lobby: str, host: str = "local") -> tuple[str, str]:
    """(base URL, lobby ID) from an ID or a pasted lobby link.

    Accepts ``AbCd1234``, ``https://openfront.io/w3/game/AbCd1234`` or
    ``localhost:9000/game/AbCd1234``; a link's own origin wins over ``host``.
    """
    lobby = lobby.strip()
    link = re.match(r"^(?:(https?)://)?([^/]+)/(?:w\d+/)?game/([A-Za-z0-9]{8})(?:[/?#].*)?$", lobby)
    if link:
        scheme = link.group(1) or ("http" if link.group(2).startswith(("localhost", "127.")) else "https")
        return f"{scheme}://{link.group(2)}", link.group(3)
    if not GAME_ID.match(lobby):
        raise ValueError(f"not a lobby ID or lobby link: {lobby!r} (IDs are 8 letters/digits)")
    base = HOSTS.get(host, host).rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ValueError(f"host must be local, openfront, or a URL, got {host!r}")
    return base, lobby


def decode_observation(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The page agent's observe() payload as one-row model inputs."""
    raster = np.frombuffer(base64.b64decode(obs["raster"]), np.uint8)
    masks = np.frombuffer(base64.b64decode(obs["masks"]), np.uint8)
    features = np.asarray(obs["features"], np.float32)
    if raster.size != L.RASTER_SIZE or features.size != L.FEATURE_SIZE or masks.size != L.MASK_SIZE:
        raise ValueError(
            f"page agent layout mismatch: raster {raster.size}/{L.RASTER_SIZE}, "
            f"features {features.size}/{L.FEATURE_SIZE}, masks {masks.size}/{L.MASK_SIZE} (rebuild: npm run build:page)")
    return (
        raster.reshape(1, L.RASTER_PLANES, L.RASTER_HEIGHT, L.RASTER_WIDTH).copy(),
        features.reshape(1, L.FEATURE_SIZE),
        masks.reshape(1, L.MASK_SIZE).astype(bool),
    )


def action_payload(action) -> dict:
    intent, target, ratio, cell, site = (int(v) for v in action)
    return {"intentType": intent, "target": target, "ratioBucket": ratio, "spatialCell": cell, "buildSite": site}


def ensure_bundle(rebuild: bool = False) -> Path:
    """Build env/dist/page_agent.js when missing or older than its sources."""
    sources = [*(ENV_ROOT / "src").rglob("*.ts")]
    stale = not BUNDLE.exists() or any(p.stat().st_mtime > BUNDLE.stat().st_mtime for p in sources)
    if rebuild or stale:
        subprocess.run("npm run build:page", cwd=ENV_ROOT, shell=True, check=True,
                       stdout=subprocess.DEVNULL)
    return BUNDLE


@dataclass
class LobbyRun:
    """What happened in one lobby game, for the CLI summary."""
    lobby: str
    actions: int = 0
    sent: dict = field(default_factory=dict)
    outcome: str = "unknown"
    peak_tiles: int = 0


class LobbyPlayer:
    """Join one lobby in a visible browser and play it with ``predictor``.

    ``interval`` is seconds between decisions; 1.0 matches training (10
    ticks of 100 ms per step).
    """

    def __init__(self, predictor, base: str, lobby: str, name: str = "OpenFrontAI",
                 profile: str = "default", interval: float = 1.0, browser: str = "msedge",
                 join_timeout: float = 1800.0, headless: bool = False, max_seconds: float | None = None,
                 log=print):
        self.predictor = predictor
        self.base = base
        self.lobby = lobby
        self.name = name
        self.profile = PROFILES / profile
        self.interval = interval
        self.browser = browser
        self.join_timeout = join_timeout
        self.headless = headless
        self.max_seconds = max_seconds
        self.log = log

    def run(self) -> LobbyRun:
        from playwright.sync_api import sync_playwright

        bundle = ensure_bundle().read_text(encoding="utf-8")
        self.profile.mkdir(parents=True, exist_ok=True)
        result = LobbyRun(self.lobby)
        with sync_playwright() as pw:
            context = self._launch(pw)
            try:
                page = context.pages[0] if context.pages else context.new_page()
                # Name the player before the client reads its settings.
                page.add_init_script(f"try {{ localStorage.setItem('username', {self.name!r}); }} catch (e) {{}}")
                page.add_init_script(bundle)
                url = f"{self.base}/game/{self.lobby}"
                self.log(f"opening {url} as {self.name!r}")
                page.goto(url, wait_until="domcontentloaded")
                self._play(page, result)
            finally:
                context.close()
        return result

    def _launch(self, pw):
        kwargs = dict(headless=self.headless, viewport={"width": 1400, "height": 860}, args=["--mute-audio"])
        channels = [self.browser] if self.browser != "chromium" else []
        for channel in [*channels, None]:
            try:
                return pw.chromium.launch_persistent_context(str(self.profile), channel=channel, **kwargs)
            except Exception as err:  # channel not installed: try the next
                last = err
        raise RuntimeError(f"no usable browser (tried {channels + ['bundled chromium']}): {last}"
                           " — install Edge/Chrome or run `python -m playwright install chromium`")

    def _status(self, page) -> dict:
        return page.evaluate("() => window.__openfrontAgent ? window.__openfrontAgent.status() : {missing: true}")

    def _play(self, page, result: LobbyRun) -> None:
        deadline = time.monotonic() + self.join_timeout
        last_note = ""
        spawned_at = None
        while True:
            status = self._status(page)
            if status.get("missing"):
                note = "waiting for the page agent to load"
            elif not status.get("inGame"):
                note = "in the lobby, waiting for the game to start (answer any check in the browser window)"
            else:
                break
            if note != last_note:
                self.log(note)
                last_note = note
            if time.monotonic() > deadline:
                raise TimeoutError(f"game did not start within {self.join_timeout:.0f}s")
            if page.is_closed():
                raise RuntimeError("browser window closed")
            time.sleep(1.0)
        version = page.evaluate("() => window.__openfrontAgent.version")
        if version != PAGE_AGENT_VERSION:
            raise RuntimeError(f"page agent v{version}, driver expects v{PAGE_AGENT_VERSION} (npm run build:page)")
        self.log(f"game started ({status.get('players')} players)")
        started = time.monotonic()

        while True:
            t0 = time.monotonic()
            status = self._status(page)
            if not status.get("inGame"):
                result.outcome = "left game"
                break
            me = status.get("me") or {}
            result.peak_tiles = max(result.peak_tiles, int(me.get("tiles") or 0))
            if self.max_seconds is not None and time.monotonic() - started > self.max_seconds:
                result.outcome = "time limit, alive" if status.get("alive") else "time limit"
                break
            if status.get("gameOver"):
                result.outcome = "game over, alive" if status.get("alive") else "game over, eliminated"
                break
            if status.get("inSpawnPhase"):
                if not status.get("spawned") and (spawned_at is None or time.monotonic() - spawned_at > 5):
                    spot = page.evaluate("(seed) => window.__openfrontAgent.spawn(seed)", int(time.time()))
                    self.log(f"spawn at ({spot['x']}, {spot['y']})")
                    spawned_at = time.monotonic()
            elif status.get("hasPlayer") and status.get("spawned") and not status.get("alive"):
                result.outcome = "eliminated"
                break
            elif status.get("hasPlayer") and status.get("alive"):
                self._decide(page, result, status)
            time.sleep(max(0.0, self.interval - (time.monotonic() - t0)))
        self.log(f"{result.outcome}; peak {result.peak_tiles} tiles, {result.actions} actions {result.sent}")

    def _decide(self, page, result: LobbyRun, status: dict) -> None:
        try:
            obs = page.evaluate("() => window.__openfrontAgent.observe()")
        except Exception as err:
            self.log(f"observe failed (retrying): {err}")
            return
        raster, features, masks = decode_observation(obs)
        action = self.predictor(raster, features, masks)[0]
        if int(action[0]) == L.NOOP:
            return
        try:
            out = page.evaluate("(a) => window.__openfrontAgent.act(a)", action_payload(action))
        except Exception as err:
            self.log(f"act failed: {L.describe_action(action)}: {err}")
            return
        result.actions += 1
        for kind in out.get("sent", []):
            result.sent[kind] = result.sent.get(kind, 0) + 1
        me = status.get("me") or {}
        self.log(f"t{obs['tick']:>6} tiles {me.get('tiles', 0):>6} troops {int(me.get('troops', 0)):>8} | "
                 f"{L.describe_action(action)} -> {', '.join(out.get('sent', [])) or 'nothing sent'}")
