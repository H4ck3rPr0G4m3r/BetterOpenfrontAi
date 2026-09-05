# OpenFront RL Environment — Phase 1 Design

**Date:** 2026-09-05
**Status:** Approved design, pending implementation plan
**Scope:** Sub-project 1 of 4 — the headless environment and Python bridge. No learning.

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
- Connecting to live multiplayer servers (phase 3)
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

That 800–1200 figure is an estimate covering simulation plus observation encoding
only. It **excludes per-step action-mask computation**, which is open question §10.1 and
could be significant. It must not be quoted later as a measured end-to-end number.

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

Ownership planes are maintained **incrementally** from `drainPackedTileUpdates()`. A
full re-rasterization of 651k tiles every step is the encoding bottleneck to avoid; the
engine already hands us exactly the deltas we need.

**Player slot table** — 17 slots. **Slot 0 is reserved for terra nullius** (the null
attack target); slots 1–16 hold real players, filled bordering-players-first then by
strength. Per slot: troops, gold, tile count, alliance state, embargo state, traitor
flag, shared-border flag, and a validity bit. Terra nullius fills only the fields that
apply, with the rest zeroed. Fixed size so the network shape is stable; slot identity
for players is *not* stable across steps, which the policy must tolerate (this is
standard for entity-slot encodings and is why the sort key is deterministic).

**Global scalars** — ~32 values: own troops, gold, tile count and share, current tick,
phase flag, alive-player count, own rank.

**Interface:** `encode(game, agentPlayerID, out: Float32Array): void`

### 5.4 `ActionCodec`

**Purpose:** decode a multi-head action into `Intent`s, and produce legality masks.

Five heads, every one masked from engine queries:

| Head | Size | Mask source |
|---|---|---|
| Intent type | ~14 | `GameRunner.playerActions()` |
| Target | 17 (slot 0 = terra nullius) | slot occupancy + per-relation flags from `playerActions().interaction`; slot 0 masked on `player.canAttack()` against a neutral border tile |
| Ratio bucket | 8 (5/10/20/30/50/70/90/100%) | always legal |
| Spatial cell | 8192 (128×64) | terrain + ownership |
| Unit type | `UnitType` count | `playerBuildables()` |

The spatial head selects a coarse **cell**, not a tile. A concrete `TileRef` is then
resolved inside that cell by an engine query — `bestTransportShipSpawn()` for boats, best
buildable tile for structures. This keeps the head at ~8k logits instead of 651k while
preserving full expressiveness, because the engine's resolver is the same one the UI
uses for human players.

**Known risk to measure before committing:** `player.buildableUnits(tile, units)` is
per-tile, so masking builds across the full grid is O(cells) engine calls per step. The
implementation plan must microbenchmark a single call first. If it is expensive, build
actions switch to a candidate-list treatment (a short list of engine-proposed sites)
instead of a dense spatial mask.

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

The raster correctness test is the single most important one — the incremental encoder
is the only place in phase 1 where we maintain state the engine also maintains, so it is
the only place the two can silently disagree.

**Python (pytest):** Gymnasium API compliance checker; protocol round-trip against a
fake Node server; worker-death recovery.

**Integration:** the phase-1 acceptance run — 1000 games, random and scripted agents,
recording steps/sec, crash count, and episode-length distribution.

---

## 9. Success criteria

1. 1000 games run end to end without crashes or resource growth.
2. Byte-identical results for a fixed seed and action sequence, across two runs.
3. Measured and documented steps/sec for World, Baltics, and Europe.
4. Random agent and scripted baseline both play complete games.
5. All tests in §8 passing.

Phase 2 does not begin until these hold.

---

## 10. Open questions carried into implementation

1. Cost of `buildableUnits()` per tile. Decides dense spatial build masking versus a
   candidate list (§5.4), and gates whether the §2.2 throughput estimate survives once
   per-step masking is included.
2. Whether the encoder can be written against public interfaces only — decides submodule
   versus fork (§3).
3. Optimal M (envs per process) and W (processes) — to be tuned empirically against the
   16-core / 8GB-VRAM machine, not guessed.
