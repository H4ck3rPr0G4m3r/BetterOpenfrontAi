# Phase 6: training foundation

**Date:** 2026-09-09
**Status:** Proposed scope, not started
**Scope:** The first trainable loop on top of the phase 1 environment. One trainer, one policy shape, one config path, plus smoke coverage. No reward tuning, no curricula, no self-play, no live lobbies.

Phase numbering here counts landed implementation chunks. Chunks 2 through 5 built GameEnv plus encoder, the action codec plus hardening, the bridge plus protocol, and the Python gym plus baselines. This is chunk 6. It starts spec phase 2 and it does not finish it.

---

## 1. Goal

Run PPO end to end against the real bridge and prove the loop learns without falling over. Success is a short smoke run that finishes, checkpoints, reloads, and evaluates, with honest throughput and terminal accounting. It is not a strong agent.

## 2. What already exists

The env side is done and frozen for this phase. Bridge schema stays at version 1. No changes to GameEnv, the encoder, the codec, or the wire format unless the smoke run finds a load-bearing bug.

Facts the design below depends on:

* Observation is a flat float32 vector of length 65684. Raster first with 65536 entries for 8 planes over a 128 by 64 grid, then the player table with 136 entries for 17 slots times 8 fields, then 12 global scalars. Slot 0 is terra nullius. Raster geometry is letterboxed and padding cells stay masked out.
* Action is MultiDiscrete with sizes 14, 17, 8, 8192, 32 in head order intent type, target, ratio bucket, spatial cell, build site. Flat mask length is 8263 in the same order.
* Masks arrive in `info["action_masks"]` and from `action_masks()`. VecEnv also answers the SB3 mask protocol through `env_method`, `get_attr`, and `has_attr`. Use that, do not invent a second path.
* VecEnv runs W workers times M envs. Workers currently step sequentially. Node auto-resets finished episodes, so a done row carries the fresh next episode observation while reward and terminal describe the finished one. Same-step autoreset.
* Tick cap maps to truncated, win and death map to terminated. Worker death respawns the worker and marks its rows done with reason `worker_death` and reward 0. Sim error ends the episode as a non-scoring terminal with a zeroed observation and all-masked output, latched until reset.
* The terminal tick count is not preserved across auto-reset. Value bootstrapping on tick cap therefore reads the fresh state, not the terminal one. Protocol.ts calls this out and the trainer must handle it in the open rather than pretend it is exact.
* Default reward is unchanged for this phase. Delta land fraction plus 0.1 times delta troop and gold fractions, with plus 1 on win and minus 1 on death. Scales are fixed at 100000 troops and 10000000 gold. Spawn placement stays a reset-time heuristic, not a learned action.
* Measured cost is roughly 16 to 20ms per step early game and 20 to 30ms mid game, about 50 to 60 steps per second per env. Twelve to 14 workers land around 650 to 850 steps per second, so 10M steps take roughly 3.5 to 4 hours before inference overhead. Reference machine is 16 cores with an 8GB RTX 3060 Ti.

## 3. Trainer choice

Use MaskablePPO from sb3-contrib. The gym wrapper already speaks its mask convention and the alternative is a hand-rolled PPO that repeats its masking logic badly.

New dependencies in `train/requirements.txt` are torch with CUDA, stable-baselines3, sb3-contrib, tensorboard, and pyyaml. Pin versions at implementation time and record the working set in the spec file before closing the phase.

## 4. Work items

### 4.1 Observation adapter

Add a train-side adapter that splits the flat vector back into raster, table, and globals. It reshapes the raster to 8 by 64 by 128, keeps the table as 17 by 8, and keeps the 12 scalars separate. It reads sizes from HELLO where possible and asserts against the constants. Normalization, if any, lives here and is documented with ranges. The bridge does not change.

### 4.2 Masked multi-head policy

SB3 default MlpPolicy does not fit this space. Add one custom policy with a small CNN over the raster plus an MLP over table and globals, feeding five masked categorical heads in bridge order. Mask splitting reuses `spaces.split_flat_mask`. Illegal entries must be impossible to sample. Out-of-range head values stay fail-fast errors because they signal caller bugs, while in-range but illegal combinations must never occur under masking and get counted if they do.

### 4.3 Training script and config

Add `train/train_ppo.py` plus YAML configs under `train/configs/`. The script wires OpenFrontVecEnv to MaskablePPO, sets W and M from config, seeds workers explicitly, logs to TensorBoard, and writes checkpoints on a fixed step interval with resume support. Defaults stay World, 20 bots, 10 ticks per step, 6000 tick cap, strict illegal actions on. A fast smoke config uses 1 to 2 workers and a few thousand steps so CI and local checks stay cheap.

Truncation handling must be explicit. Tick cap bootstraps value from the fresh observation with a logged note about the approximation. Sim error and worker death get value zero, reward zero, and a counter increment. Terminal reason histogram is a first class log, with win, death, tick cap, sim error, and worker death as separate series.

### 4.4 Evaluation script

Add `train/eval.py`. It loads a checkpoint, runs deterministic seeds against the scripted baseline behavior for reference, and reports episode return, episode length, win and death rates, and terminal distribution. Eval seeds are fixed and recorded so two runs compare directly.

### 4.5 Tests

Add `train/tests/test_train_smoke.py` against the existing fake worker. It runs the smoke config for a few hundred steps, asserts finite loss, asserts a checkpoint round-trips through save and load, and asserts eval completes. It also asserts masked sampling never picks a closed entry and that sim error and worker death paths increment counters instead of crashing. No live Node worker in unit tests. The live bridge stays covered by the existing gym tests.

## 5. Logging

Every training run records return, episode length, value loss, policy loss, entropy, learning rate, steps per second, and the terminal reason histogram. Steps per second is measured wall clock including inference, not sim only. Sim error and worker death rates are always visible next to return so a rising return built on a rising error rate cannot pass unnoticed.

## 6. Success criteria

1. Smoke config finishes on CPU against the fake worker and against one live worker.
2. A checkpoint saves, reloads, and produces identical greedy actions on fixed seeds.
3. Eval completes and reports the full terminal distribution.
4. A 1M step run holds measured throughput within 30 percent of the phase 1 estimate after inference, with zero worker kills attributed to trainer bugs.
5. All new and existing Python tests pass, and no bridge schema change was needed. If one was needed, it is versioned and both sides move together.

## 7. Non-goals

Reward tuning, auxiliary heads, curricula, self-play pools, opponent modeling, architecture search, multi-map training, live lobby play, and any change to the archived record or replay shape. Spawn stays heuristic. The policy shape stays small on purpose. A bigger network is a later phase once the loop is honest.

## 8. Open questions for implementation

1. Best W and M on the reference machine once inference joins the step cost. Tune by measurement, carry over the spec open question rather than guessing.
2. Whether sequential worker stepping in VecEnv needs parallel stepping to hold throughput. Measure first, change only if the smoke run proves it.
3. Exact PPO rollout length, batch size, and learning rate that fit 8GB VRAM with the CNN plus five heads. Start from MaskablePPO defaults, adjust to fit, record the final values.

## 9. Implementation record

Implemented 2026-09-09. Bridge schema unchanged at version 1. New files are `train/openfront_gym/adapter.py`, `train/sb3_adapter.py`, `train/openfront_policy.py`, `train/train_ppo.py`, `train/eval.py`, `train/configs/default.yaml`, `train/configs/smoke.yaml`, and `train/tests/test_train_smoke.py`.

Working set: Python 3.11, torch 2.8.0+cu128 with CUDA, stable-baselines3 2.9.0, sb3-contrib 2.9.0, gymnasium 1.3.0, tensorboard, pyyaml 6.0.3. The stock pip install of sb3 pulls a CPU torch, so install the CUDA torch from the cu128 index first and let sb3 keep it.

Verified: 58 Python tests pass, smoke config trains 32 steps against one live World worker at about 29 steps per second on CPU with finite losses and a saved checkpoint, the checkpoint reloads with identical greedy actions, and eval finishes a live episode reporting return, length, and terminal reasons. The OpenFrontIO engine deps needed `npm run inst` before the worker could boot. Criteria 1, 2, 3, and 5 hold. Criterion 4, the 1M step throughput run, is still open.

## 10. Deferred: 1M throughput run

Postponed by decision 2026-09-09, not forgotten. This is success criterion 4 and the only open item in this phase. Run it from `train/` on the reference machine:

```
python train_ppo.py --config configs/default.yaml
```

Pass gates: measured wall clock throughput within 30 percent of the phase 1 estimate of 650 to 850 steps per second after inference, zero worker kills attributed to trainer bugs, sim error and worker death rates visible in the terminal histogram next to return. On pass, close this section with the measured numbers. On fail, the failure owns a new work item, not a quiet retry.
