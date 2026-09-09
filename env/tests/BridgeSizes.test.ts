import { describe, expect, it } from "vitest";
import {
  BUILD_CANDIDATE_TYPES,
  MAX_BUILD_SITES,
  NUM_INTENT_TYPES,
  NUM_RATIO_BUCKETS,
  NUM_SPATIAL_CELLS,
  NUM_TARGET_SLOTS,
} from "../src/ActionCodec";
import {
  MASK_BUILD_SITES,
  MASK_INTENT_TYPES,
  MASK_RATIO_BUCKETS,
  MASK_SPATIAL_CELLS,
  MASK_TARGET_SLOTS,
  OBSERVATION_SIZE,
} from "../src/bridge/Protocol";
import { OBSERVATION_SIZE as ENCODER_SIZE } from "../src/ObservationEncoder";

/**
 * The bridge protocol duplicates the action and observation shapes as plain
 * numbers so it stays importable without the engine. This test pins the two
 * sides together. If the codec or encoder shapes change, update Protocol.ts
 * and train/openfront_gym/protocol.py with them.
 */
describe("Bridge shape agreement", () => {
  it("mirrors the codec head sizes", () => {
    expect(MASK_INTENT_TYPES).toBe(NUM_INTENT_TYPES);
    expect(MASK_TARGET_SLOTS).toBe(NUM_TARGET_SLOTS);
    expect(MASK_RATIO_BUCKETS).toBe(NUM_RATIO_BUCKETS);
    expect(MASK_SPATIAL_CELLS).toBe(NUM_SPATIAL_CELLS);
    expect(MASK_BUILD_SITES).toBe(MAX_BUILD_SITES);
    expect(BUILD_CANDIDATE_TYPES.length).toBeGreaterThan(0);
  });

  it("mirrors the encoder observation size", () => {
    expect(OBSERVATION_SIZE).toBe(ENCODER_SIZE);
  });
});
