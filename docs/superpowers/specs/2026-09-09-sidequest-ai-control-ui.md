# Side quest: AI control UI

**Date:** 2026-09-09
**Status:** Proposed, awaiting go-ahead
**Relation to main track:** branches off phase 6 (training foundation). It changes no learning math, no bridge schema, no encoder or codec. It adds a local dashboard that drives the existing `train_ppo.py` / `eval.py` scripts and constrains what the policy may do at action-mask level.

## 1. Goal

One local page where I can shape how the AI behaves and watch what training does:

- set aggressivity (how often and how hard it attacks vs expands and builds)
- disable capabilities (per intent type, plus per build type, so "no factories" is one checkbox)
- start and stop training runs with those settings, from the browser instead of the terminal
- see live training output: stdout tail, return, losses, steps/sec, and the terminal reason histogram next to return, same honesty rule as phase 6

## 2. What already exists (facts this plan depends on)

- Training entry points are `train/train_ppo.py --config configs/<name>.yaml` and `train/eval.py --model <ckpt> --episodes N`. Configs split into `env`, `ppo`, `policy`, `run` sections. Defaults live in `train/configs/default.yaml`, smoke in `train/configs/smoke.yaml`.
- Action space is MultiDiscrete with 5 heads: 14 intent types, 17 target slots, 8 ratio buckets (5/10/20/30/50/70/90/100%), 8192 spatial cells, 32 build sites (`train/openfront_gym/spaces.py`, `IntentType` enum).
- Build-site candidates cover 7 types: City, DefensePost, Port, SAMLauncher, MissileSilo, Factory, Warship (`env/src/ActionCodec.ts`, `BUILD_CANDIDATE_TYPES`). There is no per-type index on the wire. A site slot is one (tile, type) pair, so "no factories" means masking out slots whose type is Factory.
- Masks reach Python as one flat bool vector in head order, exposed via `info["action_masks"]` and the SB3 mask protocol (`train/openfront_gym/vec.py`). MaskablePPO reads them. Anything still open after extra masking stays legal by construction, so an extra AND-mask cannot produce illegal actions.
- There is no aggressivity knob anywhere. Reward shaping is hardcoded in `env/src/Reward.ts` (1.0 land delta, 0.1 troop delta, 0.1 gold delta, +1 win / -1 death). The scripted baseline (`train/agents/scripted_agent.py`) always expands first, attacks second, uses the middle ratio bucket.
- Run telemetry already exists in two places: stdout prints from `train_ppo.py` (steps/sec, terminal totals) and TensorBoard scalars from SB3 plus the `TerminalCounterCallback` terminal histogram. The `tensorboard` package is already a dependency, so a server can read event files with `event_accumulator` and add nothing new.

## 3. Core idea

All behavior control happens Python-side as an extra mask ANDed onto the engine masks, in one new module. No TypeScript changes, bridge stays at version 1.

```
engine masks (8263 bools)  AND  capability mask  AND  aggression mask
      -> MaskablePPO predict / training rollout
```

- Capability filter: per intent type on/off (14 toggles), per build type on/off (7 toggles). Build-type filtering needs the (tile, type) behind each site slot. The flat mask has no type labels, so the filter resolves them once per step through a tiny Node query or, simpler first cut, through the candidate order already defined in the codec. Open question 1 below picks the mechanism. If the cheap path fails, the fallback is a `site_types` field on the worker reply, which is a protocol addition and needs versioning per the phase 6 rule.
- Aggression: three layers, all optional and combinable.
  1. Inference-time bias: minimum ratio bucket for ATTACK actions, expand-first vs attack-first ordering for the eval/play predictor, boat/naval on/off. No retraining needed, works on any checkpoint.
  2. Training-time shaping: land/troop/gold coefficients and win/death bonus as env config overrides, threaded through to `Reward` via the worker config. This touches `env/` config plumbing but not the wire format.
  3. PPO-level: expose `ent_coef` and `learning_rate` in the run form. They already exist in YAML, the UI only edits them.
- Presets: named JSON files under `train/control/presets/` holding one full behavior + training setup (e.g. `no-factories.json`, `berserker.json`, `turtle.json`). Selecting a preset fills the form. This is also what makes "a new challenge for it" repeatable: the same preset constrains both training and eval.

## 4. Architecture

Two new pieces, both outside the main track. No changes to the OpenFrontIO client bundle.

```
dashboard/                  static page, no build step
  index.html / app.js       sliders, toggles, run controls, log view, charts
train/control_server.py     stdlib HTTP server (no new deps)
  GET  /api/state           jobs, presets, checkpoints, recent metrics
  POST /api/jobs            start train or eval (writes generated YAML, spawns subprocess)
  DELETE /api/jobs/<id>     stop a run
  GET  /api/jobs/<id>/logs  SSE tail of the run stdout
  GET  /api/metrics?run=    scalars from TensorBoard event files + terminal histogram
  GET/PUT /api/presets      behavior presets
```

Why stdlib + static page: the machine already runs Python and a browser. No npm build, no pip additions, nothing to version-pin. If the page outgrows vanilla JS, graduate it to Vite later. That migration is explicitly not this quest.

Job model: the server spawns `train_ppo.py` / `eval.py` as subprocesses with a generated config (base YAML + UI overrides), captures stdout to `train/control/runs/<id>/stdout.log`, and reports liveness by polling the process. One training job at a time on the reference GPU. Eval may run alongside training since eval defaults to CPU.

## 5. UI panels (one page, four sections)

1. Behavior. Aggression slider (0-100, mapped to the three layers with the mapping printed next to it, no hidden math), min attack ratio bucket, expand/attack/boat priority order, intent-type toggles (14), build-type toggles (7, Factory included). Shows the resulting effective mask counts from the last step so a toggle visibly does something.
2. Training. Base config picker (default/smoke), map, bots, tick cap, workers/envs, total timesteps, checkpoint freq, device, PPO fields (`ent_coef`, `learning_rate`, `n_steps`, `batch_size`). Start/stop, run name. Writes the exact generated YAML next to the run for reproducibility.
3. Logs and metrics. Live stdout tail (SSE), return / loss / entropy / steps-per-second charts from TensorBoard events, terminal histogram (win, death, tick cap, sim error, worker death) next to return, per the phase 6 honesty rule. Checkpoints list with save/reload note.
4. Eval. Checkpoint picker, episodes, seed, baseline checkbox, stochastic toggle. Result table: mean return, mean length, win rate, terminal distribution, side by side with baseline when asked.

## 6. Work items

1. `train/action_filter.py`: `CapabilityFilter` (intent + build-type masks) and `AggressionSettings` (ratio floor, priority order). Pure functions over the flat mask plus unit tests against the fake worker. No bridge change.
2. Wire the filter into training and eval behind a config flag: apply in `SB3OpenFrontVec` (or a thin wrapper) so constrained training learns under the same limits it will be evaluated under, and in `eval.py` predictors for inference-only constraints on existing checkpoints.
3. Reward coefficient overrides: `RewardConfig` already takes `troopScale`/`goldScale` but not the 1.0/0.1/0.1 weights or terminal bonus. Add optional weight fields, thread from worker config, keep defaults identical. Small `env/` change, no schema change.
4. `train/control_server.py`: job manager, SSE log tail, TensorBoard event reading, preset CRUD, generated-YAML runs. Stdlib only, plus the already-present `tensorboard` reader.
5. `dashboard/`: the four panels above as one static page. Poll metrics, stream logs over SSE, draw charts on canvas. No framework.
6. Presets: ship `no-factories`, `berserker`, `turtle`, `no-navy`,
   `hermit`, and `economist` as JSON, each with a one-line description
   shown in the UI, each verified to load through `BehaviorPreset`.
7. Tests: filter unit tests (disabled intent never sampled, factory slots closed, ratio floor respected), server tests (start/stop fake job, SSE tail, preset round-trip), smoke run through the generated-YAML path.

## 7. Success criteria

1. Disabling Factory closes factory build slots in a live eval while everything else stays legal, verified by mask counts and a short eval run.
2. Aggression slider extremes produce visibly different play on the same checkpoint (attack share and mean ratio bucket move), reported from one eval each.
3. A training run started from the UI finishes a smoke config, checkpoints, and its charts plus terminal histogram render on the page.
4. The generated YAML beside each run reproduces the run from the terminal byte for byte.
5. No bridge schema change. If work item 1 proves the cheap site-type path wrong, the protocol addition is versioned with both sides moving together, per phase 6 rule 5.

## 8. Non-goals

Reward tuning as research, curricula, self-play, new network shapes, multi-map training, anything inside the OpenFrontIO game client, public hosting of the dashboard, user accounts. Play-against-the-AI stays phase 4; this quest only constrains and observes training plus scripted/checkpoint eval.

## 9. Open questions

1. ~~Cheapest correct way to label build-site slots by type Python-side~~ **Resolved:**
   neither. Limits are enforced engine-side through worker config flags
   (`--disable-intent`, `--disable-build`, `--reward-weights`), so the
   codec filters its own candidate list and slot indices stay aligned by
   construction. No protocol change, schema still version 1.
2. ~~Exact aggression-slider mapping~~ **Recorded** in
   `train/action_filter.py:aggression_to_ratio_floor`: 0-33 floors attacks
   at 5% (bucket 0), 34-66 at 20% (bucket 2), 67-100 at 50% (bucket 4).
   The dashboard prints the mapping next to the slider.
3. ~~Whether stdlib SSE stays smooth for long runs~~ **Answered by
   construction:** log tail capped at 200 lines in memory, full log stays
   on disk per run, SSE sends a heartbeat every 15s.

## 10. Implementation record

Built 2026-09-09. New files: `env/src/BehaviorConfig.ts`,
`env/tests/BehaviorControls.test.ts`, `train/action_filter.py`,
`train/control_server.py`, `train/control/presets/*.json`,
`train/tests/test_action_filter.py`, `train/tests/test_control_server.py`,
`dashboard/index.html`, `dashboard/app.js`, `dashboard/styles.css`.
Touched: `ActionCodec.ts` (capability enforcement), `Reward.ts`
(weights), `worker_main.ts` (CLI flags), `openfront_gym/worker.py` and
`vec.py` (config threading plus `worker_log_paths` and `spawn_timeout`),
`train_ppo.py` / `eval.py` (behavior section, `--min-attack-ratio`),
both YAML configs.

Verified: 72 Python tests pass, 36 fast Vitest tests pass, `tsc
--noEmit` clean, dashboard JS parses. Live simulated mode streams steps
end to end through the real HTTP + SSE path against the fake worker.

Two things found during implementation, recorded so they stay found:

- The UnitType enum uses spaced names ("Defense Post", "SAM Launcher",
  "Missile Silo"). Both sides pin their build-name lists against the
  codec candidate list with tests.
- `LiveSession.start()` deadlocked on a plain `threading.Lock` by
  calling `state()` while holding it, which wedged every live endpoint
  and every status poll behind it. It is an RLock now, and the server
  tests carry deadlines so a regression fails instead of hanging a
  terminal.

## 11. How to run it

From `train/`: `python control_server.py --port 8765`, then open
http://127.0.0.1:8765/. Simulated predictor answers instantly with no
engine. Real-engine live and training runs boot node workers exactly
like the terminal scripts do; the first boot compiles the engine and is
slow, and the live phase line says so while it happens.

## 12. Review round (PR #4)

A review agent read the full diff and returned 4 blockers, all in
`control_server.py`: a bypassable path-prefix check, SSE subscribers
never removed, unrestricted checkpoint paths into `MaskablePPO.load`,
and a char-count log offset that drifts on CRLF logs. All four fixed,
plus the should-fix items that fit without new architecture (eval.json
reproducibility, override whitelist, job eviction and record deletion,
shaping-delta test, spawn-path tests, finite-weight validation,
XSS-safe eval rendering, min-ratio persisted and honored) and the nits.
One item stays manual: a real-engine multi-worker PPO rollout to
confirm the `get_attr("num_envs")` length correction against SB3.
