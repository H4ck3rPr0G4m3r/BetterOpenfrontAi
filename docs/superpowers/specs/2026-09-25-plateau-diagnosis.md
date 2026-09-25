# 2M-step plateau: why the AI plays badly, and what changed (2026-09-25)

**Status:** Diagnosis complete, fixes implemented and tested. No long
training runs yet (deferred to the very end per instruction).

## 1. Why 2M steps can't do simple things

Seven compounding causes, all evidenced in code (not guessed):

1. **Attacks punished themselves.** Reward priced symmetric troop deltas
   (`Reward.ts`): launching an attack spends troops, which reads as an
   immediate negative reward. The winning action is priced as a mistake.
   Fix: `troopLossScale` (0..1, default 1.0 = old behavior) mutes only
   negative troop deltas. New configs/presets use 0.0.
2. **Kills were free, farming paid.** Default `kill: 0`: eliminating an
   opponent paid nothing; only the subsequent land delta paid cents, while
   `troop` delta punished the fight. Fix: `fighter` preset
   (`win 5 / kill 2 / troopLossScale 0`) prices fighting above farming.
3. **Horizon mismatch.** `n_steps 128 × 16 envs = 2048`/update vs up to
   600-step episodes, `gamma 0.99` (~100-step horizon) vs wins 300+ steps
   away, plus tick-cap bootstrapping from the FRESH reset obs
   (`sb3_adapter.py:98-102`, documented approximation). Fix: long-horizon
   defaults in all curriculum configs (`n_steps 512, batch 1024,
   gamma 0.995, ent 0.02`; 8192-step rollouts, divisible batches).
4. **Unnormalized mixed-scale obs.** Ownership planes in [0,1] next to raw
   troop/gold fractions, no standardization (`adapter.py` documents the
   gap). Fix: `openfront_gym/normalize.py` (numpy RunningMeanStd,
   clip 5.0), opt-in via `run.normalization`, stats saved beside
   checkpoints ( `*_obs_norm.npz`), consumed in eval via `--obs-norm`.
5. **Full difficulty from step 0.** World/20 bots before the policy can
   expand: win/kill rewards never fire, length 75→170 with flat return is
   survival without progress. Fix: bot curriculum
   (`train/selfplay/curriculum.py`): stage1 (5 bots, nations disabled) →
   stage2 (10 bots) → stage3 (full), promotion at 0.5 eval win rate.
6. **Spatial bottleneck.** 8192-cell head from a 256-dim trunk after 64x
   downsampling; five masked categoricals share one trunk. Documented as
   the next architecture item; not reworked in this pass (needs runs to
   validate, and runs are deferred).
7. **Target-head aliasing.** The `aad3341` retarget fix took attacks from
   86% void to zero fired — by rewriting the policy's target choice to the
   first legal slot. Every open intent fires, but the target head is
   heavily aliased. Same verdict as 6: needs measured runs, deferred.

History void: the pre-fix 100k curves trained DISCONNECTED (clientID bug,
`7298448`); post-fix retargeting changed action semantics again. Any
"2M steps" figure mixing those eras measures three different agents.

## 2. Self-play (gated, as required)

`train/selfplay/pool.py` (snapshot store, uniform/last-k/latest sampling,
win-rate index, `index.json`) + `train/selfplay/curriculum.py` (stages +
promotion) + `train/train_sp.py` (segment driver: train → snapshot →
same-seed eval vs bots AND sampled pool snapshots → promote). Dormant
until eval win rate ≥ 0.5 (`selfplay_ready`); before that the bot
curriculum runs. Same-seed comparison (not same-game: the bridge seats
one learner per env; the pool format is forward-compatible with a future
in-game opponent seat). `--segments 0` validates configs and exits.

## 3. Real-time matches

- **Local vs you:** two-seat sim (`env/src/play/TwoSeatGame.ts`, no bridge
  change) + JSON play server + `train/play_session.py` mounted as
  `/api/play/*` + `dashboard/play.html`. Tested end to end (vitest socket
  test, Python protocol checks).
- **openfront.io:** `env/src/live/` (WS shim, shadow sim + hash invariant,
  deadline agent loop, rate limiter, Turnstile probe, staging runbook).
  Unit-tested; browser/network stages are manual by nature.

## 4. What still needs runs (the deferred end-game)

1. Fresh connected baseline (old curves void) → 2. stage1 → fighter
   sweeps → 3. curriculum promotion → 4. pool activation → 5. long haul.
   Configs are ready (`baseline100k`, `phased100k`, `stage{1,2,3}`,
   `selfplay` driver). Throughput triage (phase-7 3.3) and the manual
   replay-watch pass ride along with run 1.
