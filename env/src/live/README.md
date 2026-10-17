# Live games: the in-page agent

`PageAgent.ts` is bundled into `env/dist/page_agent.js` (`npm run build:page`;
`train/openfront/lobby.py` rebuilds it automatically when sources change)
and injected into the real game client by Playwright. It exposes
`window.__openfrontAgent`:

| Call | Does |
| --- | --- |
| `status()` | in game? spawned? alive? ticks, own tiles/troops |
| `observe()` | the training observation and masks for this player (base64) |
| `act(action)` | decode a 5-head action with the training codec and send it |
| `spawn(seed)` | pick free land with room to grow and send the spawn |

`GameViewAdapter.ts` makes the client's `GameView` look like the engine's
`Game` to the encoder and codec, so live play uses exactly the code the
policy trained with. Engine queries the view answers asynchronously
(buildables, attackability, boat spawns) go through a two-pass cache.

Intents leave through the client's own Transport handlers, the same path
a click takes. Nothing here hides that a bot is playing or answers human
checks for it.

Driver: `train/play_lobby.py` (CLI) and the dashboard's Play panel. End to
end check against the local server: `train/tools/local_lobby_smoke.py`.
