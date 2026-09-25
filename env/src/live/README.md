# Live lobby play (openfront.io) + local versus play

Two ways to play real games. Local versus is built and tested; official
lobbies follow the staging below.

## A. Local: play the AI yourself (works now)

One paired move per click — you and the AI share one local sim, no
browser automation, no accounts.

```bash
# terminal 1: control server (serves the dashboard + play API)
cd train && python control_server.py --port 8765

# browser: http://127.0.0.1:8765/play.html
```

1. **Start a match**: seed, bots (start with 2–5), nations, AI predictor
   (`scripted` needs no checkpoint; `checkpoint` loads a `.zip` from
   `train/`; `idle` lets you practice alone).
2. **Move**: pick an intent, a target slot (0 = unowned land: expand),
   a ratio, click the board to aim, Make move. The AI answers instantly.
3. The board polls every 2s. Blue is you, red is the AI.

Plumbing: `env/src/play/TwoSeatGame.ts` (two human seats + bots, no
bridge/protocol change) behind `env/src/play/play_server.ts` (newline
JSON over TCP), driven by `train/play_session.py` and mounted on the
control server as `/api/play/{start,move,state,stop}`.
Tests: `env/tests/TwoSeatGame.test.ts`, `env/tests/PlayServer.test.ts`
(end to end over a real socket), `train/tests/test_play_session.py`.

## B. Official openfront.io lobbies (staged, spec §10)

Transport: headless Chromium loads the REAL client and runs the REAL
join flow (Turnstile included); a `window.WebSocket` shim installed via
Playwright `addInitScript` captures the game socket (`src/live/ws_shim.js`).
Server frames (full Turns) pump into a Node shadow sim
(`src/live/shadow_sim.ts`); the agent loop (`src/live/agent_loop.ts`)
encodes, masks, predicts, and emits intents inside one ~100ms turn, with
defined drops (stale turn, shadow error, hash mismatch, overrun,
rate limit) instead of late actions. Intent budget is self-capped below
the server's 10/s + 150/min (`src/live/rate_limiter.ts`).

Staging (in order, each gates the next):

1. **Self-hosted server + real client + headless browser.** Whole loop,
   no strangers, no policy questions.
2. **Turnstile probe.** `node src/live/probe_turnstile.mjs` (needs
   `playwright`). Cheap; gates everything after it. No token → stop and
   ask the operators for an `ADMIN_BOT_API_KEY` seat instead of fighting
   the challenge.
3. **Official private lobby** (agent + invited humans).
4. **Official public lobby.** Raises the operators conversation first —
   steps 3–4 put the agent in games with people who did not opt into
   playing a bot.

Hard invariants (never relaxed):

- **Version parity first**: `window.BOOTSTRAP_CONFIG.gitCommit` must equal
  the pinned training commit (`shadow_sim.ts: PINNED_ENGINE_COMMIT`).
  `GET /desktop/version.json` is the wrong source (Steam-asset version).
- **Hash agreement every hash turn**: shadow hash must equal the wire
  hash or the episode aborts and the agent stops acting. A mismatch means
  the bot is playing a different game, not playing badly.
