import path from "path";
import { fileURLToPath } from "url";
import { describe, expect, it } from "vitest";
import { GameMapType } from "../../OpenFrontIO/src/core/game/Game";
import { TwoSeatGame } from "../src/play/TwoSeatGame";
import type { EnvConfig } from "../src/types";
import { NOOP_ACTION, NoopActionCodec, StubObservationEncoder } from "./helpers/stubs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MAPS_DIR = path.join(__dirname, "../../OpenFrontIO/resources/maps");

function makeConfig(overrides: Partial<EnvConfig> = {}): EnvConfig {
  return {
    map: GameMapType.World,
    mapsDir: MAPS_DIR,
    bots: 5,
    ticksPerStep: 10,
    tickCap: 6000,
    nations: "disabled",
    ...overrides,
  };
}

describe("TwoSeatGame", () => {
  it("seats two distinct alive players with territory", async () => {
    const game = new TwoSeatGame(makeConfig(), new StubObservationEncoder(), new NoopActionCodec());
    await game.reset("versus-seats");

    expect(game.aiPlayerID).not.toBe(game.humanPlayerID);
    expect(game.ticks).toBeGreaterThan(0);
  });

  it("steps both seats forward and returns paired results", async () => {
    const game = new TwoSeatGame(makeConfig(), new StubObservationEncoder(), new NoopActionCodec());
    await game.reset("versus-step");
    const before = game.ticks;

    const result = game.step(NOOP_ACTION, NOOP_ACTION);

    expect(game.ticks).toBe(before + 10);
    expect(result.done).toBe(false);
    expect(result.terminal).toBeNull();
    expect(Number.isFinite(result.aiReward)).toBe(true);
    expect(Number.isFinite(result.humanReward)).toBe(true);
    expect(result.aiMasks.intentType[0]).toBe(1);
    expect(result.humanMasks.intentType[0]).toBe(1);
  });

  it("ends at the tick cap with a terminal both seats agree on", async () => {
    const game = new TwoSeatGame(
      makeConfig({ tickCap: 20 }),
      new StubObservationEncoder(),
      new NoopActionCodec(),
    );
    await game.reset("versus-cap");
    let result = game.step(NOOP_ACTION, NOOP_ACTION);
    let guard = 0;
    while (!result.done && guard++ < 10) {
      result = game.step(NOOP_ACTION, NOOP_ACTION);
    }
    expect(result.done).toBe(true);
    expect(result.terminal).toBe("tick_cap");
  });

  it("emits a compact snapshot for the dashboard canvas", async () => {
    const game = new TwoSeatGame(makeConfig(), new StubObservationEncoder(), new NoopActionCodec());
    await game.reset("versus-snapshot");

    const snap = game.snapshot();

    expect(snap.w).toBe(64);
    expect(snap.h).toBe(32);
    expect(snap.cells.length).toBe(64 * 32);
    for (const code of snap.cells) {
      expect(code).toBeGreaterThanOrEqual(0);
      expect(code).toBeLessThanOrEqual(4);
    }
    // Both seats spawned: AI and human own land.
    expect(snap.cells).toContain(2);
    expect(snap.cells).toContain(3);
    expect(snap.ai.alive).toBe(true);
    expect(snap.human.alive).toBe(true);
    expect(snap.done).toBe(false);
  });
});
