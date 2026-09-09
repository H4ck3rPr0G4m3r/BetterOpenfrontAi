import { describe, expect, it } from "vitest";
import {
  ActionCodecImpl,
  BUILD_CANDIDATE_TYPES,
  IntentType,
  MAX_BUILD_SITES,
  createMaskBuffers,
} from "../src/ActionCodec";
import {
  KNOWN_BUILD_TYPES,
  normalizeBehavior,
} from "../src/BehaviorConfig";
import { Reward } from "../src/Reward";
import type { Action, MaskBuffers } from "../src/types";
import { parseWorkerArgs } from "../src/bridge/worker_main";
import { AGENT_ID, buildPostSpawnGame } from "./helpers/buildWorldGame";
import { buildTestRunner } from "./helpers/buildTestRunner";

function baseAction(overrides: Partial<Action> = {}): Action {
  return {
    intentType: IntentType.Noop,
    target: 0,
    ratioBucket: 4,
    spatialCell: 0,
    buildSite: 0,
    ...overrides,
  };
}

function masksFor(
  codec: ActionCodecImpl,
  game: Parameters<ActionCodecImpl["masks"]>[0],
): MaskBuffers {
  const masks = createMaskBuffers();
  codec.masks(game, AGENT_ID, masks);
  return masks;
}

describe("normalizeBehavior", () => {
  it("defaults to nothing disabled and phase-1 reward weights", () => {
    const behavior = normalizeBehavior();
    expect(behavior.disabledIntents.size).toBe(0);
    expect(behavior.disabledBuilds.size).toBe(0);
    expect(behavior.rewardWeights).toEqual({
      land: 1.0,
      troop: 0.1,
      gold: 0.1,
      win: 1,
      death: -1,
    });
  });

  it("rejects out-of-range intents, Noop, and unknown builds or weights", () => {
    expect(() => normalizeBehavior({ disabledIntents: [14] })).toThrow();
    expect(() => normalizeBehavior({ disabledIntents: [0] })).toThrow();
    expect(() => normalizeBehavior({ disabledBuilds: ["DeathStar"] })).toThrow();
    expect(() =>
      normalizeBehavior({ rewardWeights: { land: Number.NaN } as never }),
    ).toThrow();
    expect(() =>
      normalizeBehavior({ rewardWeights: { vibes: 2 } as never }),
    ).toThrow();
  });

  it("accepts every known build type name", () => {
    const behavior = normalizeBehavior({
      disabledBuilds: [...KNOWN_BUILD_TYPES],
    });
    expect(behavior.disabledBuilds.size).toBe(KNOWN_BUILD_TYPES.length);
  });

  it("pins KNOWN_BUILD_TYPES to the codec candidate list", () => {
    const candidateNames = new Set(
      BUILD_CANDIDATE_TYPES.map((t) => t as string),
    );
    expect(new Set(KNOWN_BUILD_TYPES)).toEqual(candidateNames);
  });
});

describe("codec capability limits", () => {
  it("keeps a disabled intent shut in masks and drops it in decode", async () => {
    const { game } = await buildPostSpawnGame("behavior-attack");
    const open = masksFor(new ActionCodecImpl(), game);
    expect(open.intentType[IntentType.Attack]).toBe(1);

    const limited = new ActionCodecImpl({ disabledIntents: [IntentType.Attack] });
    const masks = masksFor(limited, game);
    expect(masks.intentType[IntentType.Attack]).toBe(0);
    expect(masks.intentType[IntentType.Noop]).toBe(1);

    const intents = limited.decode(
      baseAction({ intentType: IntentType.Attack, target: 0 }),
      game,
      AGENT_ID,
    );
    expect(intents).toEqual([]);
  });

  it("excludes disabled build types from both masks and decode", async () => {
    const { game } = await buildPostSpawnGame("behavior-build");
    game.player(AGENT_ID).addGold(1_000_000_000n);

    const plain = new ActionCodecImpl();
    const plainMasks = masksFor(plain, game);
    expect(plainMasks.intentType[IntentType.BuildUnit]).toBe(1);

    const limited = new ActionCodecImpl({ disabledBuilds: ["Factory"] });
    const masks = masksFor(limited, game);
    const openSites: number[] = [];
    for (let i = 0; i < MAX_BUILD_SITES; i++) {
      if (masks.buildSite[i] === 1) openSites.push(i);
    }
    // Slot indices stay aligned with decode: every open site resolves to a
    // real intent for build or upgrade, and none of them is a Factory.
    let decodedAny = false;
    for (const site of openSites) {
      for (const upgrade of [false, true]) {
        const intents = limited.decode(
          baseAction({
            intentType: upgrade ? IntentType.UpgradeStructure : IntentType.BuildUnit,
            buildSite: site,
          }),
          game,
          AGENT_ID,
        );
        for (const intent of intents) {
          decodedAny = true;
          expect((intent as { unit?: string }).unit).not.toBe("Factory");
        }
      }
    }
    expect(decodedAny).toBe(true);
  });

  it("closes the build marginal when every candidate type is disabled", async () => {
    const { game } = await buildPostSpawnGame("behavior-build-all");
    game.player(AGENT_ID).addGold(1_000_000_000n);
    const limited = new ActionCodecImpl({
      disabledBuilds: [...KNOWN_BUILD_TYPES],
    });
    const masks = masksFor(limited, game);
    expect(masks.intentType[IntentType.BuildUnit]).toBe(0);
    expect(masks.intentType[IntentType.UpgradeStructure]).toBe(0);
    expect([...masks.buildSite].every((v) => v === 0)).toBe(true);
  });
});

describe("reward weight overrides", () => {
  it("keeps phase-1 values by default and applies overrides", async () => {
    const { runner } = await buildTestRunner("behavior-reward");
    const plain = new Reward();
    plain.init(runner.game, AGENT_ID);
    expect(plain.step(runner.game, AGENT_ID, "win")).toBeCloseTo(1, 10);

    const custom = new Reward({
      weights: { land: 2, troop: 0, gold: 0, win: 5, death: -5 },
    });
    custom.init(runner.game, AGENT_ID);
    expect(custom.step(runner.game, AGENT_ID, "win")).toBeCloseTo(5, 10);

    const custom2 = new Reward({ weights: { death: -3 } });
    custom2.init(runner.game, AGENT_ID);
    expect(custom2.step(runner.game, AGENT_ID, "death")).toBeCloseTo(-3, 10);
  });
});

describe("worker CLI behavior flags", () => {
  const base = ["--port", "5001"];
  it("parses repeatable disable flags and reward weights", () => {
    const options = parseWorkerArgs([
      ...base,
      "--disable-intent",
      "1",
      "--disable-build",
      "Factory",
      "--reward-weights",
      "2,0,0,5,-5",
    ]);
    expect(options.disabledIntents).toEqual([1]);
    expect(options.disabledBuilds).toEqual(["Factory"]);
    expect(options.rewardWeights).toEqual({
      land: 2,
      troop: 0,
      gold: 0,
      win: 5,
      death: -5,
    });
  });

  it("rejects Noop, out-of-range intents, unknown builds, bad weights", () => {
    expect(() => parseWorkerArgs([...base, "--disable-intent", "0"])).toThrow();
    expect(() => parseWorkerArgs([...base, "--disable-intent", "14"])).toThrow();
    expect(() =>
      parseWorkerArgs([...base, "--disable-build", "DeathStar"]),
    ).toThrow();
    expect(() =>
      parseWorkerArgs([...base, "--reward-weights", "1,2,3"]),
    ).toThrow();
  });
});
