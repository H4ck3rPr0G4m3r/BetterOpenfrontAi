import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { describe, expect, it } from "vitest";
import {
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Difficulty,
} from "../../OpenFrontIO/src/core/game/Game";
import type { GameConfig } from "../../OpenFrontIO/src/core/Schemas";
import { GameRecordSchema } from "../../OpenFrontIO/src/core/Schemas";
import { GameEnv } from "../src/GameEnv";
import {
  buildGameRecord,
  replayGameID,
  replayOptionsFromEnv,
  shouldRecordReplay,
  toJsonSafe,
} from "../src/Replay";
import type { EnvConfig } from "../src/types";
import { NOOP_ACTION, NoopActionCodec, StubObservationEncoder } from "./helpers/stubs";
import { AGENT_ID, buildTestRunner } from "./helpers/buildTestRunner";
import { TurnFeeder } from "../src/TurnFeeder";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MAPS_DIR = path.join(__dirname, "../../OpenFrontIO/resources/maps");

function makeConfig(overrides: Partial<EnvConfig> = {}): EnvConfig {
  return {
    map: GameMapType.World,
    mapsDir: MAPS_DIR,
    bots: 20,
    ticksPerStep: 10,
    tickCap: 6000,
    ...overrides,
  };
}

/** The same GameConfig shape GameEnv.reset() builds, for record tests. */
function gameConfigForTest(): GameConfig {
  return {
    gameMap: GameMapType.World,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Public,
    difficulty: Difficulty.Medium,
    nations: "default",
    donateGold: false,
    donateTroops: false,
    bots: 20,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
  };
}

describe("replay helpers", () => {
  it("records episode 0 and every Nth after", () => {
    expect(shouldRecordReplay(0, 1)).toBe(true);
    expect(shouldRecordReplay(0, 10000)).toBe(true);
    expect(shouldRecordReplay(9999, 10000)).toBe(false);
    expect(shouldRecordReplay(10000, 10000)).toBe(true);
    expect(shouldRecordReplay(3, 0)).toBe(false);
  });

  it("derives deterministic 8-char alphanumeric game IDs", () => {
    expect(replayGameID("seed-a")).toBe(replayGameID("seed-a"));
    expect(replayGameID("seed-a")).toMatch(/^[A-Za-z0-9]{8}$/);
    expect(replayGameID("another?seed!")).toMatch(/^[A-Za-z0-9]{8}$/);
  });

  it("converts bigints so records stay JSON-safe", () => {
    expect(toJsonSafe({ a: 10n, b: [1n, { c: 2n }], d: "x" })).toEqual({
      a: 10,
      b: [1, { c: 2 }],
      d: "x",
    });
    expect(() => JSON.stringify(toJsonSafe({ g: 999999999999999999999n }))).not.toThrow();
  });

  it("reads replay options from the environment", () => {
    expect(replayOptionsFromEnv({})).toBeUndefined();
    expect(
      replayOptionsFromEnv({ OPENFRONT_REPLAY_DIR: "/tmp/r" }),
    ).toEqual({ dir: "/tmp/r", every: 10000 });
    expect(
      replayOptionsFromEnv({ OPENFRONT_REPLAY_DIR: "/tmp/r", OPENFRONT_REPLAY_EVERY: "1" }),
    ).toEqual({ dir: "/tmp/r", every: 1 });
    expect(() =>
      replayOptionsFromEnv({ OPENFRONT_REPLAY_DIR: "/tmp/r", OPENFRONT_REPLAY_EVERY: "0" }),
    ).toThrow();
  });
});

describe("buildGameRecord", () => {
  it("produces a record the client schema accepts", async () => {
    // Authentic turns: a real runner fed a real spawn intent.
    const { runner } = await buildTestRunner("replay-schema");
    const feeder = new TurnFeeder(runner);
    const map = runner.game.map();
    let spawnTile = -1;
    for (let i = 0; i < map.width() * map.height(); i++) {
      if (map.isLand(i) && !map.hasOwner(i)) {
        spawnTile = i;
        break;
      }
    }
    feeder.step(1, [
      { clientID: "client0", playerID: AGENT_ID, type: "spawn", tile: spawnTile },
    ] as any);
    feeder.step(10, []);

    const record = buildGameRecord({
      seed: "replay-schema",
      gameConfig: gameConfigForTest(),
      turns: feeder.turns,
      agentClientID: "client0",
      agentWon: false,
      startedAt: 1700000000000,
      endedAt: 1700000060000,
    });
    const parsed = GameRecordSchema.safeParse(record);
    if (!parsed.success) {
      console.log(JSON.stringify(parsed.error.issues.slice(0, 5), null, 2));
    }
    expect(parsed.success).toBe(true);
  });
});

describe("GameEnv replay writing", () => {
  function tmpDir(): string {
    return fs.mkdtempSync(path.join(os.tmpdir(), "replay-test-"));
  }

  it("writes one parseable record per episode with every: 1", async () => {
    const dir = tmpDir();
    // Same env twice: episode counting must survive reset().
    const env = new GameEnv(
      makeConfig({ ticksPerStep: 10, tickCap: 5, replay: { dir, every: 1 } }),
      new StubObservationEncoder(),
      new NoopActionCodec(),
    );
    for (const seed of ["replay-ep0", "replay-ep1"]) {
      await env.reset(seed);
      let result = env.step(NOOP_ACTION);
      while (!result.done) result = env.step(NOOP_ACTION);
    }
    const files = fs.readdirSync(dir).filter((f) => f.endsWith(".json")).sort();
    expect(files.length).toBe(2);
    expect(files[0]).toContain("replay-ep0");
    expect(files[1]).toContain("replay-ep1");
    for (const file of files) {
      const parsed = GameRecordSchema.safeParse(
        JSON.parse(fs.readFileSync(path.join(dir, file), "utf8")),
      );
      expect(parsed.success).toBe(true);
    }
  });

  it("writes only every Nth episode with every: 2", async () => {
    const dir = tmpDir();
    const env = new GameEnv(
      makeConfig({ ticksPerStep: 10, tickCap: 5, replay: { dir, every: 2 } }),
      new StubObservationEncoder(),
      new NoopActionCodec(),
    );
    for (const seed of ["replay-n0", "replay-n1", "replay-n2"]) {
      await env.reset(seed);
      let result = env.step(NOOP_ACTION);
      while (!result.done) result = env.step(NOOP_ACTION);
    }
    const files = fs.readdirSync(dir).filter((f) => f.endsWith(".json")).sort();
    expect(files.length).toBe(2);
    expect(files[0]).toContain("replay-n0");
    expect(files[1]).toContain("replay-n2");
  });

  it("writes nothing without replay options", async () => {
    const dir = tmpDir();
    const env = new GameEnv(
      makeConfig({ ticksPerStep: 10, tickCap: 5 }),
      new StubObservationEncoder(),
      new NoopActionCodec(),
    );
    await env.reset("replay-off");
    let result = env.step(NOOP_ACTION);
    while (!result.done) result = env.step(NOOP_ACTION);
    expect(fs.readdirSync(dir).length).toBe(0);
  });
});
