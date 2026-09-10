# Phase 8: data replay (record training episodes, watch them in the game)

**Date:** 2026-09-10
**Status:** Proposed scope, investigation done, not started
**Scope:** One JSON file per recorded episode plus one-click playback in the real game client. No protocol change, no bridge schema change, no vendored-tree changes.

## 1. Goal

Any training or eval episode can be watched back later inside the actual OpenFrontIO client, driven by data rather than video. The file is small because the sim is deterministic: seed, config, and the agent's turns are the whole game. Bots, nations, and tribes act through internal seeded executions, never through turns, so agent intents plus the seed reproduce everything.

## 2. Facts the design depends on

* Upstream replays run from an archived `GameRecord` (`src/client/LocalServer.ts`): recorded turns become the tick intents, empty turns are skipped, missing hashes warn but do not stop playback.
* A record is `{info, version: "v0.0.2", gitCommit, turns}` (`src/core/Schemas.ts:1199`). Empty turns are filtered at write time by the engine's own `createPartialGameRecord` (`src/core/Util.ts:289`), which this phase reuses instead of reimplementing.
* The dev client skips the commit check (`gitCommit !== "DEV"` in `JoinLobbyModal.ts:1277`), so locally built clients play our records with no version dance.
* `Main.ts:436` listens for `join-lobby` on `document`, and `joinIsGateable` passes anything carrying a `gameRecord`. A bookmarklet can therefore load a record file and dispatch the event with zero changes to the vendored tree.
* Core `GameRunner` already retains every fed turn; `TurnFeeder` builds them. Capture is one array, not new machinery.
* Turn intents may carry bigints (gold amounts). Records must be JSON-safe, so the writer normalizes bigints to numbers. The acceptance test parses the output with `GameRecordSchema.safeParse`.

## 3. Work items

### 3.1 `env/src/Replay.ts`: record builder

`buildGameRecord({game, turns, seed, config, winner, startedAt, endedAt})` returns a JSON-safe object: `createPartialGameRecord` for info plus turns, `gitCommit` pinned to the training engine commit, single-element `players` mirroring a singleplayer lobby (the agent as humans[0] with clientID `client0`; bots and nations respawn from config deterministically). Stats synthesized as zeros; the replay sim regenerates the end screen. Winner names the agent on a win and stays undefined otherwise; playback ends when turns run out regardless.

### 3.2 Capture and cadence, Node side

`TurnFeeder` exposes the turns it fed (a fresh feeder already exists per episode, so boundaries are free). `GameEnv` counts episodes and, when replay is enabled, writes every Nth episode to disk as `<dir>/<seed>.json`. Enablement arrives via `OPENFRONT_REPLAY_DIR` and `OPENFRONT_REPLAY_EVERY` environment variables read in `worker_main.ts`, threaded into `EnvConfig`. No protocol change: Python only sets environment before spawning workers.

### 3.3 Python threading and cadence rule

`train_ppo.py` reads an optional `replay: {dir, every}` config section and exports the two variables before building the vec. `eval.py` gains `--replay-dir`, which implies every 1. Cadence default follows the size rule, measured 2026-09-10: a 248-step
eval episode recorded 13 KB (2782 turns, 128 non-empty). Full-length
episodes scale to the low tens of kilobytes. Small by the stated rule,
so the default is every 10,000th episode per env.

### 3.4 Playback: bookmarklet plus runbook

A bookmarklet file served next to the dashboard: opens a file picker, parses the JSON, dispatches `join-lobby` with `{gameID, gameRecord}` on `document`. Runbook: boot the vendored client dev server, open the page, click the bookmarklet, pick a file, drive with the built-in replay speed controls. Verified live before close: one recorded training episode plays end to end with matching final ownership.

## 4. Success criteria

1. A recorded episode file parses under `GameRecordSchema.safeParse` with zero errors.
2. The file plays in a dev client with no desync errors and visibly matching play (spot check final ownership against the training log).
3. A short training run with replay enabled writes files at the configured cadence and stays within 5 percent of the 137 steps per second baseline.
4. All new and existing TS and Python tests pass. No bridge schema change, no vendored-tree change.

## 6. Implementation record (2026-09-10)

Recording works: `TurnFeeder` keeps fed turns, `GameEnv` writes every Nth episode via `OPENFRONT_REPLAY_DIR/EVERY`, Python threads them from YAML and eval flags. A 248-step episode is 13 KB, so the every-10,000th default stands.

Two findings from verifying with the upstream `tests/replay/ReplayGame.ts` harness, both fixed:

* `ActionCodec.decode()` never stamped `clientID` on intents, and the engine resolves every intent through `playerByClientID`. Every policy action ever trained was silently dropped as a noop — the only intent that ever acted was the hand-written spawn. The worker logs said so all along (`player with clientID undefined not found`). `decode()` now stamps the agent's clientID with regression tests, including an end-to-end attack that grows tiles. The 100k learning curves are therefore void as learning data (see the phase-6 correction): the policy trained disconnected, which is also why the first connected eval instantly survived to tick cap with positive return.
* Replay fidelity needed three alignments on top of determinism: game ID, executor seed, and nation stream all derive from `replayGameID(seed)`; the agent's player ID is the stream's first draw (it feeds the state hash); state hashes are collected from HashUpdates into the record. Verified: 631/631 hash checkpoints match on a 600-step episode.

Playback has two paths. The in-page Watch Replay button opens a browser over the training server's replay folder (`GET /api/replays` index plus `/replays/<file>` download, both with CORS open; path traversal fails closed like the checkpoint endpoints): recorded episodes load with one click, newest last, with the disk file picker as fallback when the server is down. Covered by index/traversal tests server-side and parser, fetch, plus button-to-dispatch wiring tests client-side. The dashboard bookmarklet remains as the no-build alternative. Criterion 2's browser click-through is still open: background servers do not survive in this environment, so the dev-client watch still wants a manual pass. Criteria 1, 3, and 4 hold, with one amendment to 4: the button lives in the tracked vendored tree (one component, three tests, six strings; no `src/core` engine change, no bridge change).

## 5. Non-goals

Video export, hash capture per turn (warnings accepted), dashboard-embedded renderer, serving records from an archive API, multi-map or team-game records. Spawn stays heuristic, so the record starts post-spawn like training does.
