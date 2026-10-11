"""Drive a policy inside a real OpenFront client through Playwright.

One driver serves two modes:

* AI vs Human (local): run the OpenFront dev server (``npm run dev`` in
  OpenFrontIO/, http://localhost:9000), create a private lobby in your own
  browser, and give the lobby ID to ``play_lobby.py --host local``.
* AI in a real openfront.io lobby: ``play_lobby.py --host openfront <id>``
  with a lobby ID (or link) you were given, or ``--public ffa`` to take the
  next public lobby the site itself offers (``--list`` shows them).

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
# Local dev servers mint 8-character IDs, openfront.io currently 10.
GAME_ID = re.compile(r"^[A-Za-z0-9]{6,16}$")
PAGE_AGENT_VERSION = 3
# Public lobbies the site schedules itself; "hosted" ones are player-run.
PUBLIC_KINDS = ("ffa", "team", "special")

# The homepage announces its public lobby list with this event.
WATCH_LOBBIES = """
window.__ofPublicLobbies = null;
document.addEventListener('public-lobbies-update', (e) => { window.__ofPublicLobbies = e.detail.payload; });
"""
# The client's own link shape: /w<simpleHash(id) % numWorkers>/game/<id>.
JOIN_URL = """(id) => {
  const n = (window.BOOTSTRAP_CONFIG && window.BOOTSTRAP_CONFIG.numWorkers) || 0;
  let h = 0;
  for (let i = 0; i < id.length; i++) { h = (h << 5) - h + id.charCodeAt(i); h = h & h; }
  return `${location.origin}${n ? `/w${Math.abs(h) % n}` : ""}/game/${id}`;
}"""


def resolve_lobby(lobby: str, host: str = "local") -> tuple[str, str]:
    """(base URL, lobby ID) from an ID or a pasted lobby link.

    Accepts ``AbCd1234``, ``https://openfront.io/w3/game/AbCd1234`` or
    ``localhost:9000/game/AbCd1234``; a link's own origin wins over ``host``.
    """
    lobby = lobby.strip()
    link = re.match(r"^(?:(https?)://)?([^/]+)/(?:w\d+/)?game/([A-Za-z0-9]{6,16})(?:[/?#].*)?$", lobby)
    if link:
        scheme = link.group(1) or ("http" if link.group(2).startswith(("localhost", "127.")) else "https")
        return f"{scheme}://{link.group(2)}", link.group(3)
    if not GAME_ID.match(lobby):
        raise ValueError(f"not a lobby ID or lobby link: {lobby!r} (IDs are 6-16 letters/digits)")
    base = HOSTS.get(host, host).rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ValueError(f"host must be local, openfront, or a URL, got {host!r}")
    return base, lobby


def summarize_lobbies(payload: dict) -> list[dict]:
    """The homepage's public lobby payload as flat rows, soonest start first."""
    rows = []
    now = payload.get("serverTime")
    for kind, games in (payload.get("games") or {}).items():
        for game in games or []:
            config = game.get("gameConfig") or {}
            starts = game.get("startsAt")
            rows.append({
                "id": game.get("gameID"), "kind": kind, "map": config.get("gameMap"),
                "mode": config.get("gameMode"), "players": int(game.get("numClients") or 0),
                "max": config.get("maxPlayers"),
                "starts_in": max(0, round((starts - now) / 1000)) if starts and now else None,
            })
    rows.sort(key=lambda r: (r["starts_in"] is None, r["starts_in"] or 0, -r["players"]))
    return rows


def pick_public(rows: list[dict], kind: str = "ffa", min_lead: int = 10) -> dict | None:
    """The public lobby to join: one with room that starts soonest but not
    within ``min_lead`` seconds (joining takes a moment), else the fullest
    one still waiting for its countdown."""
    kinds = PUBLIC_KINDS if kind == "any" else (kind,)
    room = [r for r in rows if r["kind"] in kinds and r["id"]
            and (not r["max"] or r["players"] < r["max"])]
    timed = [r for r in room if r["starts_in"] is not None and r["starts_in"] >= min_lead]
    if timed:
        return min(timed, key=lambda r: r["starts_in"])
    waiting = [r for r in room if r["starts_in"] is None]
    return max(waiting, key=lambda r: r["players"]) if waiting else None


def launch_context(pw, profile: Path, browser: str = "msedge", headless: bool = False):
    """A persistent browser context on ``profile``, falling back to bundled Chromium."""
    profile.mkdir(parents=True, exist_ok=True)
    # Playwright turns the browser's sandbox off by default (--no-sandbox);
    # this browser visits other people's sites, so it stays on.
    kwargs = dict(headless=headless, viewport={"width": 1400, "height": 860}, args=["--mute-audio"],
                  chromium_sandbox=True)
    channels = [browser] if browser != "chromium" else []
    for channel in [*channels, None]:
        try:
            return pw.chromium.launch_persistent_context(str(profile), channel=channel, **kwargs)
        except Exception as err:  # channel not installed: try the next
            last = err
    raise RuntimeError(f"no usable browser (tried {channels + ['bundled chromium']}): {last}"
                       " — install Edge/Chrome or run `python -m playwright install chromium`")


def open_home(page, base: str, want_lobbies: bool, timeout: float, log=print) -> list[dict]:
    """Load the site's homepage and wait until its client is up (and, with
    ``want_lobbies``, has the public lobby list). Returns that list."""
    # Whether the main document is Cloudflare's check page, read from the
    # responses alone: nothing is run inside the page while the check is up.
    challenged = {"on": False}

    def on_response(response):
        if response.request.is_navigation_request() and response.frame == page.main_frame:
            challenged["on"] = response.headers.get("cf-mitigated") == "challenge"

    page.on("response", on_response)
    page.goto(f"{base}/", wait_until="domcontentloaded")
    start = time.monotonic()
    noted = ""
    while True:
        config, lobbies = False, None
        if not challenged["on"]:
            try:
                config, lobbies = page.evaluate("() => [!!window.BOOTSTRAP_CONFIG, window.__ofPublicLobbies]")
            except Exception:  # mid-navigation
                pass
        if config and (lobbies or not want_lobbies):
            page.remove_listener("response", on_response)
            return summarize_lobbies(lobbies) if lobbies else []
        waited = time.monotonic() - start
        note = ("Cloudflare check: click it in the browser window" if challenged["on"]
                else "waiting for the site to load" if waited > 5 else "")
        if note and note != noted:
            log(note)
            noted = note
        if waited > timeout:
            raise TimeoutError(f"{base} did not load within {timeout:.0f}s")
        time.sleep(0.5)


def list_public_lobbies(base: str = HOSTS["openfront"], profile: str = "default", browser: str = "msedge",
                        timeout: float = 180.0, log=print) -> list[dict]:
    """The public lobbies the site's homepage shows right now, read in a
    short-lived visible window (the site's bot check blocks headless)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        context = launch_context(pw, PROFILES / profile, browser)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.add_init_script(WATCH_LOBBIES)
            return open_home(page, base, True, timeout, log)
        finally:
            context.close()


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

    def __init__(self, predictor, base: str, lobby: str | None, name: str = "OpenFrontAI",
                 profile: str = "default", interval: float = 1.0, browser: str = "msedge",
                 join_timeout: float = 1800.0, headless: bool = False, max_seconds: float | None = None,
                 public: str | None = None, log=print):
        # lobby: an ID, or None with public = ffa | team | special | any to
        # take the next public lobby of that kind.
        if lobby is None and public is None:
            raise ValueError("give a lobby ID or a public lobby kind")
        self.predictor = predictor
        self.base = base
        self.lobby = lobby
        self.public = public
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
        with sync_playwright() as pw:
            context = launch_context(pw, self.profile, self.browser, self.headless)
            try:
                page = context.pages[0] if context.pages else context.new_page()
                # Name the player before the client reads its settings.
                page.add_init_script(f"try {{ localStorage.setItem('username', {self.name!r}); }} catch (e) {{}}")
                page.add_init_script(WATCH_LOBBIES)
                page.add_init_script(bundle)
                # The homepage first: it knows which worker serves a lobby
                # and, for public play, which lobbies are open.
                rows = open_home(page, self.base, self.public is not None, 120.0, self.log)
                if self.public is not None:
                    pick = pick_public(rows, self.public)
                    if pick is None:
                        raise RuntimeError(f"no open public {self.public} lobby right now ({len(rows)} listed)")
                    self.lobby = pick["id"]
                    when = f", starts in {pick['starts_in']}s" if pick["starts_in"] is not None else ""
                    self.log(f"public {pick['kind']} lobby {pick['id']}: {pick['map']}, "
                             f"{pick['players']}/{pick['max'] or '?'} players{when}")
                url = page.evaluate(JOIN_URL, self.lobby)
                self.log(f"opening {url} as {self.name!r}")
                page.goto(url, wait_until="domcontentloaded")
                result = LobbyRun(self.lobby)
                self._play(page, result)
            finally:
                context.close()
        return result

    def _status(self, page) -> dict:
        try:
            return page.evaluate("() => window.__openfrontAgent ? window.__openfrontAgent.status() : {missing: true}")
        except Exception:
            if page.is_closed():
                raise
            return {"missing": True}  # mid-navigation

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
