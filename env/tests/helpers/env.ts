import path from "path";
import { fileURLToPath } from "url";
import { GameMapType } from "../../../OpenFrontIO/src/core/game/Game";
import { ActionCodecImpl } from "../../src/ActionCodec";
import type { BehaviorConfig, RewardWeights } from "../../src/BehaviorConfig";
import { GameEnv } from "../../src/GameEnv";
import { HEAD_USAGE, Masks, siteRow, spatialRow, targetRow } from "../../src/Layout";
import type { Action, EnvConfig } from "../../src/types";

export const MAPS_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "../../../OpenFrontIO/resources/maps");

export function makeEnv(
  overrides: Partial<EnvConfig> = {},
  behavior?: BehaviorConfig,
  reward?: Partial<RewardWeights>,
): GameEnv {
  const config: EnvConfig = {
    map: GameMapType.World,
    mapsDir: MAPS_DIR,
    bots: 10,
    ticksPerStep: 10,
    tickCap: 6000,
    seats: 1,
    ...overrides,
  };
  return new GameEnv(config, new ActionCodecImpl(behavior), reward);
}

/** Small deterministic PRNG so test rollouts are reproducible. */
export function seededRandom(seed = 1): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export function openIndices(mask: Uint8Array): number[] {
  const out: number[] = [];
  mask.forEach((v, i) => v && out.push(i));
  return out;
}

/** Conditional masked-uniform action: intent first, then that intent's heads. */
export function randomAction(masks: Masks, rand: () => number): Action {
  const pick = (m: Uint8Array) => {
    const open = openIndices(m);
    return open.length > 0 ? open[Math.floor(rand() * open.length)] : 0;
  };
  const intent = pick(masks.intent);
  const use = HEAD_USAGE[intent];
  return {
    intentType: intent,
    target: use.target ? pick(targetRow(masks, intent)) : 0,
    ratioBucket: use.ratio ? pick(masks.ratio) : 0,
    spatialCell: use.spatial >= 0 ? pick(spatialRow(masks, use.spatial)) : 0,
    buildSite: use.site >= 0 ? pick(siteRow(masks, use.site)) : 0,
  };
}

/** The engine's own desync hash (GameImpl.hash, private in TS; it is what
 *  HashUpdates and server desync detection compare). */
export function stateHash(game: unknown): number {
  return (game as { hash(): number }).hash();
}
