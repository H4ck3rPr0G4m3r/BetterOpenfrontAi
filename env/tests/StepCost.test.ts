import path from "path";
import { fileURLToPath } from "url";
import { describe, expect, it } from "vitest";
import { GameMapType } from "../../OpenFrontIO/src/core/game/Game";
import {
  ActionCodecImpl,
  createMaskBuffers,
  NUM_INTENT_TYPES,
  NUM_RATIO_BUCKETS,
  NUM_SPATIAL_CELLS,
  NUM_TARGET_SLOTS,
  MAX_BUILD_SITES,
} from "../src/ActionCodec";
import { GameEnv } from "../src/GameEnv";
import { ObservationEncoder } from "../src/ObservationEncoder";
import { Reward } from "../src/Reward";
import type { Action, EnvConfig } from "../src/types";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MAPS_DIR = path.join(__dirname, "../../OpenFrontIO/resources/maps");
const AGENT_ID = "agent0";

/**
 * Integration step-cost measurement (spec §2.2's budget, §5.4's masking, §10.4's
 * deadline): a full GameEnv with the real encoder and codec, driven by
 * masked-random actions, timed end to end plus per-component. Like the
 * encoder's throughput check, this records honestly with headroom rather
 * than gating on a pre-measurement guess.
 */
describe("StepCost", () => {
  it("measures the real per-step cost of sim + encode + masks", async () => {
    const config: EnvConfig = {
      map: GameMapType.World,
      mapsDir: MAPS_DIR,
      bots: 20,
      ticksPerStep: 10,
      tickCap: 6000,
    };
    const encoder = new ObservationEncoder();
    const codec = new ActionCodecImpl();
    const env = new GameEnv(config, encoder, codec, new Reward());
    // Scanned seed: survives spawn with open borders on clean maps.
    await env.reset("codec3-19");

    // Fixed-seed sampler, not Math.random: the timing gate must measure the
    // same action distribution every run, not luck into cheap ones.
    let rngState = 0x12345678;
    const rand = (): number => {
      rngState = (Math.imul(rngState, 1664525) + 1013904223) | 0;
      return (rngState >>> 0) / 4294967296;
    };
    const sampleMasked = (): Action => {
      const masks = createMaskBuffers();
      codec.masks(env.game, AGENT_ID, masks);
      const pick = (mask: Uint8Array, size: number): number => {
        const open: number[] = [];
        for (let i = 0; i < size; i++) {
          if (mask[i] === 1) open.push(i);
        }
        return open.length > 0 ? open[Math.floor(rand() * open.length)] : 0;
      };
      return {
        intentType: pick(masks.intentType, NUM_INTENT_TYPES),
        target: pick(masks.target, NUM_TARGET_SLOTS),
        ratioBucket: pick(masks.ratioBucket, NUM_RATIO_BUCKETS),
        spatialCell: pick(masks.spatialCell, NUM_SPATIAL_CELLS),
        buildSite: pick(masks.buildSite, MAX_BUILD_SITES),
      };
    };

    // Warm up (JIT, rail-network caches) outside the measurement.
    for (let i = 0; i < 2; i++) {
      const r = env.step(sampleMasked());
      if (r.done) throw new Error("bench seed ended early");
    }

    const steps = 10;
    const t0 = performance.now();
    for (let i = 0; i < steps; i++) {
      const r = env.step(sampleMasked());
      if (r.done) throw new Error("bench seed ended early");
    }
    const meanStepMs = (performance.now() - t0) / steps;

    // Per-component slices on the same live state.
    const obsOut = new Float32Array(encoder.size);
    const iters = 10;
    let t = performance.now();
    for (let i = 0; i < iters; i++) encoder.encode(env.game, AGENT_ID, obsOut);
    const encodeMs = (performance.now() - t) / iters;

    const maskBuf = createMaskBuffers();
    t = performance.now();
    for (let i = 0; i < iters; i++) codec.masks(env.game, AGENT_ID, maskBuf);
    const masksMs = (performance.now() - t) / iters;

    t = performance.now();
    for (let i = 0; i < iters; i++) codec.decode(sampleMasked(), env.game, AGENT_ID);
    const decodeMs = (performance.now() - t) / iters;

    const simMs = Math.max(0, meanStepMs - encodeMs - masksMs - decodeMs);
    console.log(
      `GameEnv.step() on World (20 bots, masked-random actions): ` +
        `step=${meanStepMs.toFixed(2)}ms (≈${(1000 / meanStepMs).toFixed(0)} steps/sec) ` +
        `sim≈${simMs.toFixed(2)}ms encode=${encodeMs.toFixed(2)}ms ` +
        `masks=${masksMs.toFixed(2)}ms decode=${decodeMs.toFixed(3)}ms`,
    );

    // Generous gate: ~2.4x the measured early-game value at authorship
    // (16.5ms/step), there to catch regressions, not to bless a budget.
    // Mid-game steps cost more (richer agent, more legal Factory queries),
    // and masked-random actions vary run to run — hence the headroom.
    expect(meanStepMs).toBeLessThan(40);
  });
});
