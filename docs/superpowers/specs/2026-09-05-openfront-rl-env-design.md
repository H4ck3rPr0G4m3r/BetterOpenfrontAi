# OpenFront RL Environment — Phase 1 Design

**Date:** 2026-09-05
**Status:** Approved design, pending implementation plan
**Scope:** Sub-project 1 of 4 — the headless environment and Python bridge. No learning.
§10 additionally specifies phase 3 (live openfront.io lobbies), because two phase-1
decisions only make sense against it.

---

## 1. Goal

Build a Gymnasium-compatible environment that drives the real OpenFront simulation
headlessly, fast enough and reproducibly enough to train a reinforcement learning agent
on later. Phase 1 succeeds when 1000 games run end to end, reproducibly, at a measured
and recorded steps/sec, driven by a random agent and a scripted baseline.

Phase 1 deliberately contains **no learning code**. All the technical risk lives in the
environment; coupling env bugs to training bugs would make both undiagnosable.

### Non-goals for this phase

- Training loops, reward tuning, curricula, self-play (phase 2)
- Connecting to live multiplayer servers (phase 3 — designed in §10, not built here)
- Any user-facing way to play against a checkpoint (phase 4)

---

## 2. Context and measured facts

The upstream engine is [OpenFrontIO](https://github.com/openfrontio/OpenFrontIO),
pinned at commit `f51f165b947a92cc683ade3a72c3800300e86e61` (2026-09-04).

### 2.1 The simulation is already headless

`tests/perf/fullgame/FullGamePerf.ts` runs the real pipeline — `createGame` +
`GameRunner` + `Executor` + real `Config`, production maps via `NodeGameMapLoader`,
nations from the map manifest — with no browser, no worker, and no server. This is the
foundation. It is **not** the same as `tests/util/Setup.ts`, which uses `TestConfig` and
downscaled test maps; training against those would produce a policy that does not
transfer.

### 2.2 Throughput (measured 2026-09-05, 16-core machine, RTX 3060 Ti)

Nations enabled, real combat occurring:

| Map | Players | ticks/sec | mean tick | Land tiles |
|---|---|---|---|---|
| Baltics | 35 | 1545 | 0.65ms | 1,039,982 |
| **World** | 92 | **1075** | 0.93ms | 651,569 |
| Europe | 72 | 566 | 1.77ms | 2,345,907 |

World is the fastest of the large maps and is the default training map.

An earlier measurement of 2503 ticks/sec was discarded: it used `--nations disabled`,
which skips `execManager.nationExecutions()` entirely, leaving tribes that barely act
(0 units, 0 deaths after 1800 ticks). It was measuring an idle simulation.

**Derived budget.** At 10 sim ticks per agent decision, World gives ~107 decisions/sec
per environment. With 12–14 worker processes on 16 cores that is ~1000–1700 raw
steps/sec, and an estimated 800–1200 after observation encoding. 10M steps lands around
3 hours. Encoding cost, not simulation cost, is the thing to optimize.

That 800–1200 figure was an estimate; part of it is no longer one. Observation encoding
is now measured directly (§5.3): ~7ms/call on World with 20 bots, which is added cost on
top of the ~9.3ms/10-tick simulation budget, not a rounding error inside it — closer to
+75% than the "cheap encoding" the estimate assumed. Action-mask computation is
**still excluded** — quantified in §11.1 as tens-of-microseconds-per-candidate for a
bounded candidate list, but not yet measured end to end inside a real `GameEnv.step()`.
The combined encode+mask+sim cost per step must not be quoted as measured until that
integration measurement exists; treat the 800–1200 figure as superseded, not confirmed.

### 2.3 Determinism is real

Baltics, seed `rl-probe`, run twice: identical final state hash
(`12599198748988756`), identical unit counts. The engine's seeded PRNG and
float-free arithmetic hold up. This is the load-bearing assumption of reproducible
training and it has been verified, not assumed.

### 2.4 The action space is player-indexed, not spatial

From `src/core/Schemas.ts`, most intents carry a player ID rather than a tile:

- `attack` → `{targetID, troops}` — no tile
- `targetPlayer`, `allianceRequest`, `breakAlliance`, `embargo`, `donate_gold`,
  `donate_troops`, `emoji` → player-indexed
- Only `spawn`, `boat.dst`, `build_unit.tile`, `move_warship.tile` need a tile

`AttackIntentSchema.targetID` is `MappedID.nullable()`, and `AttackExecution.ts:73`
resolves a null target to `mg.terraNullius()`. **A null target is how a player expands
into unowned land** — the dominant action of the entire early game, not an edge case.
The action space must be able to express it (§5.4).

This makes a structured action space tractable. A naive flat space over 651k tiles
would not be.

### 2.5 The engine computes its own legality

`GameRunner` exposes, for the UI's benefit: `playerActions()`, `playerBuildables()`,
`attackClusteredPositions()`, `bestTransportShipSpawn()`. These are action masks and
candidate-target generators we get for free. Building the action space on top of them is
the difference between a tractable project and a stalled one.

### 2.6 Ownership is directly readable

`GameMap.tileStateBuffer(): Uint16Array` exposes the full tile-state plane, with bits
0–11 holding `ownerID`. Separately, `GameImpl.drainPackedTileUpdates()` returns
`Uint32Array` pairs of `(tileRef, state | terrainByte << 16)` for tiles that changed
this tick. The encoder uses the deltas for speed and the full buffer as a correctness
oracle in tests.

Note that `map4x.bin` is a downsampled *terrain* map. It does not carry live ownership
and cannot be used as an ownership raster.

### 2.7 A human seat works headlessly — verified

The measurements in §2.2 were all taken with `players: []`
(`FullGamePerf.ts:239`) — zero human players. They proved the simulation runs fast and
deterministically, but said nothing about whether an agent-controlled seat works at all,
which every component in §5 assumes.

A throwaway probe (`OpenFrontIO/tests/perf/fullgame/HumanSeatProbe.ts`) seats one
`PlayerType.Human` player and drives it entirely through Intents in Turns. Results on
World with 20 bots and default nations:

| Check | Result |
|---|---|
| Human seat exists after the spawn phase | pass |
| Agent alive with troops | pass — 25000 troops |
| `SpawnIntent` claims territory | pass — 28 tiles |
| `attack` with `targetID: null` expands into terra nullius | pass — 28 → 716 tiles |
| Per-tick cost with a human present | 0.97ms, 1026 ticks/sec |

`randomSpawn: false` is the default, so the agent must place itself via `SpawnIntent`
during the spawn phase — this works, and the spawn tile is a normal action-space
decision, not special-cased setup. The human seat costs roughly 5% throughput versus
§2.2's 1075 ticks/sec, which is within noise for planning purposes.

This probe is the natural basis for implementation task #1 and should be replaced by
`env/GameEnv.ts` plus its tests, not kept.

---

## 3. Repo layout

`BetterOpenfrontAi/` is a git repository. `OpenFrontIO/` is currently gitignored and
tracked by the pinned commit recorded in §2.

**The submodule-vs-fork decision is deliberately deferred** to the point where it can be
made on evidence: write the observation encoder against the public `Game` / `Player` /
`GameMap` interfaces first. If it works without reaching into engine internals,
`OpenFrontIO/` becomes a git submodule pinned upstream and our code stays a clean
sibling package. If the encoder needs to patch `src/core` (most likely to short-circuit
the packed-update drains for allocation reasons), it becomes an in-tree fork branch
instead.

```
BetterOpenfrontAi/
  OpenFrontIO/          # upstream engine (submodule or fork branch — decided per above)
  env/                  # TypeScript: environment + bridge server
    src/
      GameEnv.ts
      TurnFeeder.ts
      ObservationEncoder.ts
      ActionCodec.ts
      Reward.ts
      EnvConfig.ts
      bridge/WorkerServer.ts
      bridge/Protocol.ts
    tests/
  train/                # Python: gym wrapper, baselines, harnesses
    openfront_gym/
      env.py
      vec.py
      protocol.py
      spaces.py
    agents/
      random_agent.py
      scripted_agent.py
    tests/
  docs/superpowers/specs/
```

### Licensing

The engine is **AGPL-3.0**. `env/` links `src/core` directly and is therefore a derived
work inheriting AGPL. That is unproblematic for private local use; it becomes a
publication obligation if the environment or a trained-agent service is ever network-
hosted. `train/` communicates only over a socket and is arguably separate, but there is
no practical benefit to licensing it differently — the whole repo is AGPL-3.0.

---

## 4. Architecture

```
Python (train/)                    Node (env/)
┌────────────────────┐            ┌──────────────────────────────┐
│ PPO / random agent │            │ WorkerServer                 │
│        ↕           │            │  ├── GameEnv[0]              │
│ VecEnv (W workers) │◄──socket──►│  ├── GameEnv[1]   M per proc │
│        ↕           │  batched   │  └── GameEnv[M-1]            │
│ OpenFrontEnv       │  binary    │        ↓                     │
└────────────────────┘            │  GameRunner + Executor       │
                                  │  (real src/core simulation)  │
                                  └──────────────────────────────┘
```

Each of the W worker processes hosts M environments and speaks one socket to Python.
Python sends a batch of M actions and receives one packed observation buffer back. IPC
cost is amortized twice over: across M environments per round trip, and across the 10
simulation ticks each `step()` advances.

---

## 5. Components

Each component has one purpose, a defined interface, and is testable in isolation.

### 5.1 `GameEnv`

**Purpose:** own one episode's lifecycle.

**Interface:**
```ts
class GameEnv {
  constructor(config: EnvConfig);
  reset(seed: string): Observation;
  step(action: Action): { obs: Observation; reward: number; done: boolean; info: Info };
  close(): void;
}
```

**Depends on:** `Game`, `GameRunner`, `Executor` from `src/core`; `TurnFeeder`,
`ObservationEncoder`, `ActionCodec`, `Reward`.

Constructed the way `FullGamePerf.ts` constructs its game: real `Config`, production map
via `NodeGameMapLoader`, nations from the map manifest — except that the human seat is
passed to `createGame()` rather than left empty. The agent joins as a `PlayerType.Human`
player and its actions become ordinary `Intent`s inside a `Turn` — the exact path the
real browser client uses. This is what makes phase 3 transfer possible at all, and it is
verified working in §2.7.

**Episode boundaries.** The spawn phase runs first (~302 turns); the agent's `spawn`
intent is issued during it. Nations do not eliminate each other within 3300 ticks, so
episodes need a tick cap (default 6000 post-spawn) with value bootstrapping rather than
running to a natural winner. An episode also ends on agent death or on a `WinCheck`
result.

### 5.2 `TurnFeeder`

**Purpose:** correctly advance the simulation by N ticks.

`GameRunner.executeNextTick()` returns `false` when `currTurn >= turns.length` — the
simulation does not advance without a queued `Turn`. A 10-tick `step()` therefore pushes
**10 turns** (nine with empty intents, one carrying the agent's) and calls
`executeNextTick()` ten times. This is the loop `FullGamePerf.ts:325-333` already
implements; it is extracted here rather than reinvented.

Isolating this in its own unit keeps an easy off-by-one out of `GameEnv`.

### 5.3 `ObservationEncoder`

**Purpose:** turn game state into a fixed-shape `Float32Array`, cheaply.

Three parts, written into one preallocated buffer:

**Spatial raster** — a fixed 128×64 grid for every map, planes:

| Plane | Contents |
|---|---|
| 0 | land mask |
| 1 | self-owned fraction |
| 2 | ally-owned fraction |
| 3 | enemy-owned fraction |
| 4 | unowned/terra-nullius fraction |
| 5 | self troop density |
| 6 | enemy troop density |
| 7 | structure presence |

The grid shape is fixed across maps so the policy network has one input shape, but maps
have very different aspect ratios (World 2000×1000, Baltics 1364×1324, Europe
2904×1672). A single scale factor per map would distort geometry differently on each
one, so the raster is **letterboxed**: one scale factor `s = max(width/128,
height/64)` applied to both axes, the map centred in the grid, and unused border cells
zeroed with the land mask clear. Geometry stays isometric on every map at the cost of
some wasted cells on non-2:1 maps. The spatial action head shares this grid, so
letterbox padding cells are permanently masked out.

**Built as a full recompute, not the incremental design this section originally
specified** — every plane is rebuilt from `map.tileStateBuffer()` / `map.isLand()`
each call, no per-episode state, no dependency on `drainPackedTileUpdates()` (a
destructive queue that would need exactly one owner). Incremental maintenance was the
starting assumption; it was dropped once the full recompute was implemented and
measured, not on a promise it would turn out fast. It's genuinely not free: **~7ms per
call on World with 20 bots** (`ObservationEncoder.test.ts`'s throughput check), down
from an initial ~13ms after two rounds of profiling-driven optimization (flat
typed-array lookups instead of `Map`/string-keyed `Record`; hoisting the per-row raster
cell out of the per-tile loop). The remaining cost is dominated by `map.isLand(t)` — an
interface method called once per tile, up to ~2M times on World — which has no bulk
buffer to read around the way `ownerID` does via `tileStateBuffer()`. This adds to the
~9.3ms/10-tick simulation budget (§2.2) rather than replacing any of it; recorded here
so it isn't silently assumed away the way the buildableUnits() cost briefly was (§11.1).

**Player slot table** — 17 slots. **Slot 0 is reserved for terra nullius** (the null
attack target); slots 1–16 hold real players, filled bordering-players-first then by
strength. Per slot: troops, gold, tile count, alliance state, embargo state, traitor
flag, shared-border flag, and a validity bit. Terra nullius fills only the fields that
apply, with the rest zeroed. Fixed size so the network shape is stable; slot identity
for players is *not* stable across steps, which the policy must tolerate (this is
standard for entity-slot encodings and is why the sort key is deterministic).

**Global scalars** — 12 values, implemented as a real, justified set rather than padded
out to this section's original "~32" estimate: tick fraction, spawn-phase flag,
alive-player fraction, own rank among alive players, own troop/gold/tile fractions, own
alive/traitor/doomsday-clock flags, ally fraction, and embargoed-against-us fraction.
More can be appended later without breaking anything that reads by field name — the
interface's `size` is authoritative, not a number fixed in this document.

**Interface:** `encode(game, agentPlayerID, out: Float32Array): void` — implemented in
`env/src/ObservationEncoder.ts`, tests in `env/tests/ObservationEncoder.test.ts`
(shape/finiteness, the raster-correctness oracle §8 calls out, the ownership-fraction
partition invariant, agent-exclusion from its own opponent table, and the throughput
check above).

**A genuine pre-existing engine bug surfaced while writing these tests, unrelated to
this code**: at higher bot counts (30-50) and certain seeds, `GameImpl.ts`'s
`updateBorderStatus` throws `Cannot read properties of undefined (reading
'_borderTiles')` during the spawn phase — a dangling `ownerID` with no resolvable
`Player`. Reproducible at bots=20 too, just less often; not caused by anything in
`env/`, since it's deep in engine code we never touch. Worth reporting upstream. It's
also a concrete instance of exactly the failure mode §7 already designs for — a caught
simulation exception ends the episode as a non-scoring `sim_error` rather than crashing
the process — and a reason training-scale bot counts (§2.2 used 50) may see more
`sim_error` terminals than a quick read of §7 would suggest.

### 5.4 `ActionCodec`

**Purpose:** decode a multi-head action into `Intent`s, and produce legality masks.

Five heads, every one masked from engine queries:

| Head | Size | Mask source |
|---|---|---|
| Intent type | ~14 | `GameRunner.playerActions()` |
| Target | 17 (slot 0 = terra nullius) | slot occupancy + per-relation flags from `playerActions().interaction`; slot 0 masked on `player.canAttack()` against a neutral border tile |
| Ratio bucket | 8 (5/10/20/30/50/70/90/100%) | always legal |
| Spatial cell | 8192 (128×64) | terrain + ownership |
| Build site | ≤32 (bounded candidate list, not the raster) | see below |

The spatial head selects a coarse **cell**, not a tile, for attacks and spawns. A
concrete `TileRef` is resolved inside that cell by an engine query — best buildable tile
via the same candidate mechanism as the build-site head. This keeps the head at ~8k
logits instead of 651k while preserving full expressiveness, because the engine's
resolver is the same one the UI uses for human players.

**Build masking is a bounded candidate list, not a dense grid over the spatial
head — measured, not assumed.** `player.buildableUnits(tile, units)` was benchmarked
directly (`env/probes/BuildableUnitsBench.ts`, World map, 50 bots, ~4900 owned tiles):

- Calling it with the default 11-type list across the full 8192-cell raster costs
  ~1000ms — **~108× the ~9.3ms simulation budget of one 10-tick step.** Dense per-cell
  build masking is not viable at any reasonable candidate density; this closes open
  question §11.1 in favor of a bounded list.
- Per-type cost is bimodal. Calling with a single type is what actually matters, since
  each action head only needs to ask about the unit type it represents:
  - **Cheap** (City, Defense Post, SAM Launcher, Missile Silo, Atom Bomb, Hydrogen Bomb,
    MIRV, Warship): well under 100us/call even on a large (~4900-tile) territory.
    Affordable for a candidate list of tens of tiles.
  - **Port**: ~50-100us/call — affordable for a bounded list, not for the raster.
  - **Factory**: ~6us/call when illegal (unowned tile), but **~840us/call when the
    build is actually legal** — the cost is `RailNetwork.overlappingRailroads()` /
    `computeGhostRailPaths()`, gated on `buildNew`, not on territory size. A candidate
    list must cap Factory queries tightly (a handful of sites, not all border tiles).
  - **TransportShip via `buildableUnits()`**: 60-290us/call — too expensive to use for
    spatial masking. `GameRunner.bestTransportShipSpawn()` was benchmarked separately and
    is the correct replacement: ~11us/call, about 25× cheaper. It is a purpose-built
    query, not an assumption that it would be cheap because it exists.

**Design:** the build-site head is a bounded candidate list (≤32 slots) built from
`playerBorderTiles()` plus a small number of interior sites, masked with one
single-unit-type `buildableUnits()` call per candidate per relevant unit type — not the
default 11-type query. Boat/transport-ship destinations go through
`bestTransportShipSpawn()` instead of `buildableUnits()`. This keeps every mask
engine-verified and sound (§8); it is a change in which engine query is called and how
many candidates are considered, not a relaxation of correctness.

**Interface:**
```ts
decode(action: Action, game: Game, playerID: PlayerID): Intent[]
masks(game: Game, playerID: PlayerID, out: MaskBuffers): void
```

### 5.5 `Reward`

**Purpose:** map a state transition to a scalar. Pluggable via `EnvConfig`.

The phase-1 default, stated concretely so the implementation is unambiguous:

```
r = 1.0 * Δ(own_land_tiles / total_land_tiles)
  + 0.1 * Δ(own_troops / TROOP_SCALE)
  + 0.1 * Δ(own_gold   / GOLD_SCALE)
  + terminal: +1 win, -1 death, 0 tick-cap
```

`TROOP_SCALE` and `GOLD_SCALE` are **fixed constants**, not per-step maxima over living
players. A moving denominator would hand the agent negative reward whenever a rival's
troops spiked, which is not a signal about its own play.

These coefficients are a defensible starting point, not a tuned result — reward *design*
is a phase-2 concern. What phase 1 owes is one unambiguous default and a clean seam
(`Reward` selected by `EnvConfig`) to swap it without touching `GameEnv`.

### 5.6 `bridge/WorkerServer` and `Protocol`

**Purpose:** move batched observations and actions across the language boundary.

Length-prefixed binary framing over a local socket (named pipe on Windows). One message
carries actions for all M environments; the reply carries one packed observation buffer
plus rewards, done flags, and mask buffers. `Protocol.ts` and `protocol.py` are written
against a single shared schema description so the two ends cannot silently drift.

Auto-reset on episode end (standard `VecEnv` semantics) happens on the Node side, so a
`done` reply already carries the first observation of the next episode.

### 5.7 Python `openfront_gym`

**Purpose:** present all of the above as a standard Gymnasium API.

`OpenFrontEnv` implements `reset`/`step` for a single environment; `OpenFrontVecEnv`
fans out over W worker processes and presents the batched interface trainers expect.
Action masks are surfaced in `info` under the convention MaskablePPO already reads, so
phase 2 can use an off-the-shelf trainer.

---

## 6. Data flow for one `step()`

1. Python sends a batch of M encoded actions.
2. For each env, `ActionCodec.decode()` produces zero or more `Intent`s.
3. `TurnFeeder` pushes one `Turn` carrying those intents, then nine empty turns,
   calling `executeNextTick()` after each.
4. `ObservationEncoder` folds each tick's `drainPackedTileUpdates()` into the running
   raster, then writes slots and globals into the output buffer.
5. `Reward` computes the scalar from the pre/post state snapshot.
6. Terminal check: agent dead, win, or tick cap. If terminal, reset and encode the new
   initial observation in place.
7. `ActionCodec.masks()` writes the next step's masks.
8. The worker replies with one packed buffer.

---

## 7. Error handling

- **Simulation exception.** `GameRunner.executeNextTick()` catches tick errors and
  invokes the callback with an `ErrorUpdate` instead of throwing. `GameEnv` treats this
  as a terminal, non-scoring episode end: it logs the seed and the action history,
  ends the episode with reward 0, and resets. A crashed episode must never silently
  poison a training batch, and the seed plus action history must be enough to reproduce
  it exactly.
- **Illegal action submitted.** Should be impossible given masking, so it is treated as
  a bug rather than smoothed over: raise in test/dev builds, count and drop in
  production runs. A rising drop counter is a mask-correctness regression signal.
- **Worker death.** The Python `VecEnv` detects a closed socket, respawns the worker,
  and marks its M environments as done. Training continues with a logged warning.
- **Protocol desync.** Every frame carries a sequence number and a payload length.
  Mismatch is fatal on both sides — silent desync would corrupt training data in a way
  that is very hard to diagnose later.

---

## 8. Testing

**TypeScript (Vitest):**

| Test | Asserts |
|---|---|
| Determinism | Same seed + same action sequence → identical final game hash |
| Raster correctness | Incremental ownership raster matches a full pass over `tileStateBuffer()` at every checkpoint |
| Mask soundness | Every unmasked action executes without throwing and produces a legal intent |
| Mask completeness | A sample of masked-out actions is genuinely illegal per the engine |
| Observation invariants | Shape stable, all values finite and within documented bounds |
| Turn feeding | N-tick step advances `game.ticks()` by exactly N |
| Throughput floor | Steps/sec above a recorded threshold, so regressions surface in CI |
| Shadow-sim agreement (phase 3) | A second `GameRunner` fed the turn stream a first one produced reaches an identical hash at every hash turn |

The shadow-sim agreement test needs no browser: it is the §10.3 invariant proved against
a known-good local source, and it reuses the determinism test's machinery.

The raster correctness test is the single most important one — the incremental encoder
is the only place in phase 1 where we maintain state the engine also maintains, so it is
the only place the two can silently disagree.

**Python (pytest):** Gymnasium API compliance checker; protocol round-trip against a
fake Node server; worker-death recovery.

**Integration:** the phase-1 acceptance run — 1000 games, random and scripted agents,
recording steps/sec, crash count, and episode-length distribution.

---

## 9. Success criteria

1. 1000 games run end to end **with `sim_error` terminals (§7) counted and reported, not
   silently eliminated as a pass/fail gate.** This criterion originally read "without
   crashes or resource growth" — changed on evidence, not softened as a convenience. A
   pre-existing engine bug (§5.3's `_borderTiles` crash) was measured at roughly 7/17
   seeds (~40%) failing within spawn plus a 300-tick buffer on World with 20 bots
   (`env/probes/SeedScan.ts`) — a small sample, and **a floor, not the true rate**: a
   seed that passed that exact check still crashed after 10 more ticks in a later test
   (`ObservationEncoder.test.ts`'s `encoder-safe-10`), so full 6000-tick episodes should
   be expected to fail *more* often than 40%, not to match it. What phase 1 *can* commit
   to: every `sim_error` ends its episode cleanly as a non-scoring terminal (§7, verified
   by `GameEnv`'s tests) rather than crashing the worker process, and the acceptance run
   reports the resulting rate honestly. A rate even this high is itself a reason to
   prioritize the root-cause work in §11's new open question before scaling up training.
2. Byte-identical results for a fixed seed and action sequence, across two runs.
3. Measured and documented steps/sec for World, Baltics, and Europe.
4. Random agent and scripted baseline both play complete games.
5. All tests in §8 passing.

Phase 2 does not begin until these hold.

Phase 3 (§10) has no acceptance criteria yet, deliberately: the Turnstile probe
(§10.6, step 2) gates everything after it, and writing criteria before that answer
would be writing them for a phase that may need a different shape entirely. The
probe and the version-parity check are the two things answerable now, and both are
cheap — the same status `HumanSeatProbe.ts` had before §2.7 existed.

---

## 10. Phase 3 — real lobbies through a headless browser

Not phase-1 scope. It is specified here because two phase-1 decisions are only correct
if this is the target: the encoder's interface discipline (§3) and the action cadence
(§5.2).

**Goal:** the trained agent plays on the official openfront.io servers, in the same
public lobbies as everyone else — so it can be tested against strong human players, and
so you can play alongside or against it in a normal game.

### 10.1 Why a headless browser rather than a raw WebSocket client

Official joins are gated by Cloudflare Turnstile (`src/server/JoinVerify.ts`,
`src/server/Worker.ts:488`): the first join of a `persistentID` must carry a token or is
rejected. `Roster.ts` caches admitted persistentIDs so reconnects skip the re-check, but
the first one is unavoidable. A hand-rolled Node WebSocket client has no way to produce
that token. A real browser loading the real page does, through the ordinary widget the
site shows every player.

Everything else follows from that. The browser is not a UI convenience — it is the
authenticated transport. It loads openfront.io's own bundle and runs their join flow,
their auth, their Turnstile, their asset loading, and holds a legitimately admitted
session.

**Explicit non-goal: no anti-detection work.** No fingerprint spoofing, no stealth
plugins, no Turnstile solvers. Either headless Chromium passes the ordinary challenge or
it does not. If it does not, the correct next step is to ask the operators — OpenFront
already ships an `ADMIN_BOT_API_KEY` path for sanctioned bots
(`src/server/AdminBotRoutes.ts`), so a bot seat is a conversation they have had before,
not an unthinkable request.

### 10.2 Architecture

Three ways into a running page were considered:

| Approach | Verdict |
|---|---|
| Synthetic mouse/key events on the canvas | Most faithful, most fragile — needs screen↔tile mapping and UI-state tracking against a minified bundle |
| Hook the client `EventBus` (`SendAttackIntentEvent` etc.) | Clean, but requires serving our own bundle; only viable self-hosted |
| **Shim `window.WebSocket` before the bundle loads** | **Chosen** |

The shim is installed with Playwright's `addInitScript`, which runs before any page
script. It wraps `window.WebSocket`, captures the instance the client opens to the game
worker (`src/client/Transport.ts:393`), and exposes both directions to Node:

- **Inbound:** every server frame is forwarded out of the page. Turns arrive here — the
  full intent list of every player, which is all the state the simulation needs.
- **Outbound:** we `send()` zbin-encoded intents on that same socket.

Intents sent this way are indistinguishable from the client's own, because the server
stamps `clientID` onto the intent itself rather than trusting the payload
(`src/server/IntentAuthorization.ts`). The visible client also stays perfectly
consistent: the server relays our intent back inside a Turn like everyone else's, and
the page's own worker executes it. The rendered game is never a lie about what the agent
did. Set `binaryType = "arraybuffer"` on the captured socket and marshal frames across
the page boundary via `exposeBinding` — trivial at a 100ms cadence, but worth pinning
down rather than discovering.

Observation comes from a **shadow simulation** in Node: the same `GameRunner` +
`Executor` as `GameEnv`, constructed from the lobby's `GameStartInfo` and fed the
intercepted Turn stream. Determinism (§2.3) is what makes this work — same start info,
same turns, same state. `ObservationEncoder` and `ActionCodec` then run unchanged.

```
openfront.io  ──ws──►  headless Chromium (real client, real Turnstile)
                            │  WebSocket shim (addInitScript)
                            ▼
                       Node driver
                        ├── shadow GameRunner   ← intercepted Turns
                        ├── ObservationEncoder / ActionCodec   (phase 1, unchanged)
                        └── policy checkpoint → intent → shim → socket
```

### 10.3 The load-bearing risk: shadow-sim divergence

This is phase 3's equivalent of the human-seat assumption (§2.7), except it fails
silently rather than loudly. The shadow sim is only correct if our pinned engine matches
what openfront.io is actually running — same execution logic, same config constants,
same map data. Terrain does not come over the socket; it comes from their CDN, while we
load `resources/maps/` at our pinned commit. A mismatch produces a confidently wrong
observation, not a crash.

Two mechanisms that already exist turn this from silent into loud:

1. **Version parity, checked before joining.** The served page injects the server's own
   commit as `window.BOOTSTRAP_CONFIG.gitCommit` (`src/server/RenderHtml.ts:19`, read by
   `ClientEnv.gitCommit()`), so the shim reads it directly from the live page — the
   commit of the bundle we are actually running against. Refuse to join on mismatch. Note
   that `GET /desktop/version.json` is *not* the right source despite looking like one:
   its `coreVersion` comes from the desktop shell release descriptor
   (`src/server/DesktopRelease.ts:295`), which is a Steam-asset version with different
   semantics from the web bundle's commit.
2. **Hash agreement as a hard invariant.** Clients periodically send a state hash (`hash`
   intent, `src/core/Schemas.ts:1055`; turns carry one too), which the server uses for
   desync detection (`src/server/DesyncDetector.ts`). Our shim sees the real client's
   hashes on the wire. **Every hash turn, the shadow sim's hash must equal the client's,
   or the episode aborts.** The agent stops acting the moment it can no longer prove it
   is looking at the real game.

Getting this wrong is the difference between "the bot played badly" and "the bot was
playing a different game."

### 10.4 The clock is no longer ours

Phase 1's `TurnFeeder` owns the clock: `step()` blocks until 10 ticks are simulated. In
a real lobby the server owns it. Turns arrive at ~100ms and cannot be paused, so phase 3
is a deadline-bound reactive loop — decode turn, advance shadow sim, encode, mask, infer,
emit, all inside one turn interval, with defined behavior on overrun (drop the action;
never land it late against stale state).

This raises the priority of open question §11.1: masking cost that merely reduces
training throughput becomes a hard real-time deadline here.

### 10.5 Rate limits

`src/server/ClientMsgRateLimiter.ts` allows 10 intents/second, 150 intents/minute, 2000
bytes per intent, and 5MB total per client. The per-second and per-minute limits
throttle; the size limits kick. Phase 1's cadence of one action per 10 ticks is ~60
intents/minute at a 100ms tick — comfortable, but not so comfortable that a multi-intent
action or a burst-happy policy can be ignored. The driver enforces its own budget below
the server's rather than discovering the ceiling by being kicked mid-game.

### 10.6 Staging

1. **Self-hosted server, real client, real headless browser.** The whole loop, with no
   policy questions and no strangers involved. A development stepping stone, not the
   deliverable.
2. **Turnstile probe.** Load openfront.io headless and check whether a token materializes
   at all. Cheap, and it gates everything after it — phase 3's `HumanSeatProbe.ts`.
3. **Official private lobby**, agent plus invited humans. Covers "play with it myself
   with others" end to end.
4. **Official public lobby.** The real test against strong players.

Steps 3 and 4 put the agent in games with people who did not opt into playing a bot.
That is your call to make, and step 4 is worth raising with the operators first; this
design does not decide it.

### 10.7 What phase 3 needs from phase 1

- **The observation encoder written against the public `Game` / `Player` / `GameMap`
  interfaces.** The shadow sim runs a real `GameImpl`, so this is not strictly required —
  but `GameView implements GameMap` and exposes `tileStateBuffer()`
  (`src/client/view/GameView.ts:1357`), so interface discipline keeps page introspection
  available as a fallback if the shadow sim proves unmaintainable. Cheap insurance, and
  it is the same discipline §3 wants for the submodule-versus-fork decision.
- **Action-codec symmetry.** The codec must emit real `Intent` objects, not env-internal
  shorthand, so one encoder feeds `TurnFeeder` in training and the socket shim in a live
  game.
- **Determinism** (§2.3), on which the shadow sim entirely depends.

---

## 11. Open questions carried into implementation

1. ~~Cost of `buildableUnits()` per tile.~~ **Resolved** (`env/probes/BuildableUnitsBench.ts`,
   World map, ~4900 owned tiles): dense per-cell masking costs ~108× the per-step
   simulation budget and is not viable. §5.4 now specifies a bounded candidate-list
   build-site head instead, with Factory capped tightly (~840us/call when legal) and
   `TransportShip` masking replaced by `bestTransportShipSpawn()` (~11us/call, ~25×
   cheaper than the equivalent `buildableUnits()` query). Still open: the exact
   candidate-selection heuristic (how `playerBorderTiles()` output is thinned to ≤32
   sites) is an implementation detail, not a design gate, and phase 3's real-time
   deadline (§10.4) inherits this same bound.
2. Whether the encoder can be written against public interfaces only — decides submodule
   versus fork (§3), and keeps phase 3's page-introspection fallback available (§10.7).
3. Optimal M (envs per process) and W (processes) — to be tuned empirically against the
   16-core / 8GB-VRAM machine, not guessed.
4. Whether headless Chromium obtains a Turnstile token by ordinary means (§10.6, step 2).
   Gates all of phase 3, and is answered by a probe, not by argument.
5. **Root cause of the `_borderTiles` engine crash (§5.3, §9 criterion 1).** Traced to
   `GameImpl.owner()` → `playerBySmallID()` returning `undefined` for a smallID present
   in the tile-state plane — a dangling ownerID with no corresponding entry in
   `_playersBySmallID`. Confirmed nations register there regardless of later spawn
   failure (`this._nations.forEach((n) => this.addPlayer(n.playerInfo))` runs
   unconditionally), which rules out the simplest theory. `_playersBySmallID` only ever
   grows via `push()` and nothing in `GameImpl.ts` removes entries from it, which points
   at the write side, not removal: `GameMapImpl.setOwnerID` only rejects an ID above the
   4095 bit-width ceiling (`PLAYER_ID_MASK`), not one beyond the actual registered player
   count, so an out-of-range-but-under-4096 ownerID would write cleanly and only surface
   later as this crash. Where such a write would originate — bot code, nation code, or
   elsewhere — wasn't run down further, since fixing engine internals is out of scope
   while the submodule-vs-fork decision (#2) is still open. Worth an upstream bug report
   either way; blocks criterion 1 until resolved one way or the other.
