# Phase 7: reward tuning and throughput triage

**Date:** 2026-09-10
**Status:** Proposed scope, not started
**Scope:** Short training runs that learn, plus an explanation for the throughput gap. No long runs, no self-play pool, no curricula, no live lobbies.

Phase numbering counts landed implementation chunks. Chunk 6 built the first trainable loop. The 100k shakedown on top of it showed the loop is stable but the shaping does not teach winning, and throughput is 20x below estimate. This phase fixes both findings on cheap runs before anything long runs again.

---

## 1. Goal

A 50 to 100k run whose return rises and whose win rate is nonzero, with eval beating the scripted baseline on fixed seeds. Next to that, a profiled answer for why 16 envs deliver 35 steps per second instead of the estimated 650 to 850. Success is a weight set plus a worker layout worth spending a long run on. It is not a strong agent.

## 2. What the 100k shakedown showed

Struck through 2026-09-10: the run trained disconnected (phase-6 spec section 12). The agent's actions never reached the sim, so none of the learning curves below describe learning. Kept for the record; do not tune against them. The phase opens instead with a fresh connected baseline.

Facts the plan below depends on, recorded in the phase 6 spec section 11:

* Return flat at -0.91 to -0.85 across 100k steps, win rate 0, death 90 to 100 percent of terminals. The minus 1 death bonus dominates and the 1.0 land plus 0.1 troop and gold deltas carry no usable gradient toward winning.
* Episode length doubled from 75 to about 170 while return stayed flat. The agent learned survival without progress, which is the signature of weak shaping rather than a broken loop.
* Value learning works: value loss 0.52 down to about 0.015, explained variance near 0.5. Policy churns without gain: clip fraction near 0.3, approx KL near 0.05, entropy falling slowly.
* Stability is proven: zero sim errors and zero worker deaths over roughly 600 episodes at 16 envs.
* Throughput baseline is 137 env steps per second on the 16-env default shape, measured 2026-09-10 after PR 5 (8 workers by 2, World, 20 bots, CUDA inference, 2048 steps in 15.0s). That confirms the expected 4x over the pre-parallel 35. The remaining gap to the 650 to 850 estimate is a tuning item for work item 3.3, not a blocker for sweeps: at 137 per second a 50k sweep is about 6 minutes.

## 3. Work items

### 3.1 Weights-only sweep first

The override path already exists: `behavior.reward_weights` in YAML through worker config into `Reward` via `BehaviorConfig.ts`, no bridge change. Sweep win, death, land, troop, and gold weights on 50k runs each from the same seed set. Candidates worth trying early: a win bonus large enough to beat the best survival-only return (5 to 10, not 1), death scaled to match, land turned down so empty grabbing pays less than fighting. Each run gets the same fixed eval seeds against the scripted baseline. If no weights-only set produces nonzero wins, that is a finding, not a failure, and it promotes work item 3.3.

### 3.2 Eval grows behavior stats

`train/eval.py` currently reports return, length, win rate, and terminal mix. Tuning needs to see how the policy plays, not just how it scores. Add per eval action histograms: attack share, expand share, mean ratio bucket on attacks, build counts by type, boat usage. Same fixed seeds every time so weight sets compare directly. Cheap, Python only.

### 3.3 Throughput triage

Profile one rollout step into sim milliseconds, encode milliseconds, mask milliseconds, and inference milliseconds, on the reference machine and on this machine separately. Then measure a small W by M grid on short runs and record total and per env steps per second next to CPU and GPU inference. Output is a table plus the chosen worker layout for tuning sweeps, recorded here. No guessing, same rule as phase 1.

### 3.4 Kill bonus, implemented 2026-09-10

Built ahead of the sweeps at user request rather than gated behind 3.1: a per elimination bonus on top of the existing terms. `Reward` snapshots living opponent IDs at init and pays `kill` for each one gone at the next step, counted exactly once. The agent's own death never counts. Default is 0, so old 5-value configs behave exactly as before; pass a 6th weight to opt in. The berserker preset now carries kill 1.0. Touched `Reward.ts`, `BehaviorConfig.ts`, worker CLI parsing, Python preset validation, both test suites. Bridge schema still version 1.

### 3.5 Phased shaping, implemented 2026-09-10

Per user request, one step further: the reward now runs three phases on the post-spawn tick clock (early to 900, mid to 3000, late after; ~1.5 and ~5 minutes at live cadence). Early prices land with `landEarly` and punishes losses extra with `landLoss`; mid pays `city`/`factory` structure deltas and `troopHold` movement toward keeping 30 percent of peak troops; late pays a tiny per-step `survive` trickle while alive plus `rich` movement toward affording a live-priced MIRV. Everything is a delta except the survival trickle, so episode length cannot mint return by idling. Weights grow to 13 in fixed order; 5- and 6-value configs still work, and `land` without `landEarly` keeps applying early so no old config changes meaning. New `phased` preset carries starter values. Verified live: a 2-episode eval on the full 13 finished with finite return.

## 4. Logging

Every sweep run records the phase 6 set plus the new behavior stats: return, length, losses, entropy, steps per second wall clock with inference, terminal histogram with sim error and worker death visible, attack share, mean attack ratio bucket, build counts. The generated YAML beside each run stays the reproduction record.

## 5. Success criteria

1. One weight set shows rising return and nonzero win rate on a 50 to 100k run, with eval on fixed seeds beating the scripted baseline on win rate or return.
2. Behavior stats move with the weights: at least attack share or mean ratio bucket differs between the extremes, so tuning visibly steers play.
3. Throughput gap explained with a profile table and a chosen W by M layout for the next long run.
4. All new and existing Python and Vitest tests pass, no bridge schema change. If work item 3.4 is needed, its `Reward.ts` change lands with tests and the schema stays at version 1.

## 6. Non-goals

Self-play pools, curricula, architecture search, multi-map training, live lobby play, any change to the archived record or replay shape, long runs beyond 100k steps. Spawn stays heuristic.

## 7. Open questions for implementation

1. Which weight set wins, and whether wins need new reward terms at all.
2. Where the throughput goes: sim, encode, inference, or worker layout.
3. Exact PPO rollout length, batch size, and learning rate that fit 8GB VRAM once sweeps run at scale, carried over from the phase 6 open questions.
