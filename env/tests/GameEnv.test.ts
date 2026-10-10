import fs from "fs";
import os from "os";
import path from "path";
import { beforeAll, describe, expect, it } from "vitest";
import { GameRecordSchema, type Turn } from "../../OpenFrontIO/src/core/Schemas";
import { createGameRunner } from "../../OpenFrontIO/src/core/GameRunner";
import { NodeGameMapLoader } from "../../OpenFrontIO/tests/perf/fullgame/NodeGameMapLoader";
import type { SeatStep } from "../src/GameEnv";
import { GlobalField, GLOBAL_OFFSET } from "../src/Layout";
import { NOOP_ACTION } from "../src/types";
import { MAPS_DIR, makeEnv, randomAction, seededRandom, stateHash } from "./helpers/env";

const LONG = 180_000;

describe("GameEnv lifecycle", () => {
  it("seats several agents in one game with distinct ids and spread spawns", async () => {
    const env = makeEnv({ seats: 3 });
    const rows = await env.reset("seats-1");
    expect(new Set(env.seatIDs).size).toBe(3);
    expect(rows).toHaveLength(3);
    for (const row of rows) {
      expect(row.active).toBe(true);
      expect(row.done).toBe(false);
      expect(row.masks.intent[0]).toBe(1);
      expect(row.tiles).toBeGreaterThan(0);
      expect(row.obs.features[GLOBAL_OFFSET + GlobalField.Alive]).toBe(1);
    }
    const w = env.game.map().width();
    const centers = env.seatIDs.map((id) => [...env.game.player(id).tiles()][0]);
    for (let i = 0; i < centers.length; i++) {
      for (let j = i + 1; j < centers.length; j++) {
        const d = Math.hypot((centers[i] % w) - (centers[j] % w), Math.floor(centers[i] / w) - Math.floor(centers[j] / w));
        expect(d).toBeGreaterThan(30);
      }
    }
  }, LONG);

  it("is deterministic for a seed and action sequence", async () => {
    const run = async () => {
      const env = makeEnv({ seats: 2 });
      let rows = await env.reset("determinism");
      const rand = seededRandom(7);
      for (let i = 0; i < 30 && !env.gameOver; i++) {
        rows = env.step(rows.map((r) => (r.active ? randomAction(r.masks, rand) : NOOP_ACTION)));
      }
      return stateHash(env.game);
    };
    expect(await run()).toBe(await run());
  }, LONG);

  it("ends every seat at the tick cap with a placement reward, then refuses to step", async () => {
    const env = makeEnv({ seats: 2, tickCap: 50 });
    let rows = await env.reset("cap");
    for (let i = 0; i < 5; i++) rows = env.step([NOOP_ACTION, NOOP_ACTION]);
    for (const row of rows) {
      expect(row.done).toBe(true);
      expect(row.terminal).toBe("tick_cap");
      expect(row.placement).toBeGreaterThanOrEqual(0);
    }
    expect(env.gameOver).toBe(true);
    expect(() => env.step([NOOP_ACTION, NOOP_ACTION])).toThrow(/reset/);
  }, LONG);

  it("aborts on request with zero reward", async () => {
    const env = makeEnv({ seats: 2 });
    await env.reset("abort");
    const rows = env.abort();
    expect(rows.map((r) => r.terminal)).toEqual(["aborted", "aborted"]);
    expect(rows.every((r) => r.reward === 0 && r.done)).toBe(true);
    expect(env.gameOver).toBe(true);
  }, LONG);

  it("takes a dead seat out of play while the game goes on", async () => {
    // A passive seat gets eaten within a couple hundred steps (measured).
    const env = makeEnv({ seats: 2, bots: 20 });
    let rows: SeatStep[] = await env.reset("death-1");
    let deathStep = -1;
    for (let i = 0; i < 400 && deathStep < 0 && !env.gameOver; i++) {
      rows = env.step([NOOP_ACTION, NOOP_ACTION]);
      const dead = rows.findIndex((r) => r.terminal === "death");
      if (dead >= 0) {
        deathStep = i;
        expect(rows[dead].done).toBe(true);
        expect(rows[dead].reward).toBeLessThan(-2);
      }
    }
    expect(deathStep).toBeGreaterThanOrEqual(0);
    if (!env.gameOver) {
      const deadSeat = rows.findIndex((r) => r.terminal === "death");
      rows = env.step([NOOP_ACTION, NOOP_ACTION]);
      expect(rows[deadSeat].active).toBe(false);
      expect(rows[deadSeat].reward).toBe(0);
      expect(Array.from(rows[deadSeat].masks.intent)).toEqual([1, ...new Array(13).fill(0)]);
    }
  }, LONG);
});

describe("replays", () => {
  it("records a multi-seat game that re-simulates to the same state hash", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "of-replay-"));
    const env = makeEnv({ seats: 2, tickCap: 120, replay: { dir, every: 1 } });
    let rows = await env.reset("replay-1");
    const rand = seededRandom(3);
    while (!env.gameOver) rows = env.step(rows.map((r) => (r.active ? randomAction(r.masks, rand) : NOOP_ACTION)));
    const finalHash = stateHash(env.game);
    const finalTick = env.game.ticks();

    const files = fs.readdirSync(dir);
    expect(files).toHaveLength(1);
    const record = JSON.parse(fs.readFileSync(path.join(dir, files[0]), "utf8"));
    const parsed = GameRecordSchema.safeParse(record);
    expect(parsed.success).toBe(true);
    expect(record.info.players.map((p: { clientID: string }) => p.clientID)).toEqual(["RLSEAT00", "RLSEAT01"]);

    // Re-simulate exactly as the real client does (createGameRunner), with
    // the record's sparse turns expanded back to one turn per tick.
    const runner = await createGameRunner(
      { gameID: record.info.gameID, lobbyCreatedAt: 0, config: record.info.config, players: record.info.players },
      undefined,
      new NodeGameMapLoader(MAPS_DIR),
      () => {},
    );
    const byNumber = new Map<number, Turn>(record.turns.map((t: Turn) => [t.turnNumber, t]));
    for (let n = 0; n < finalTick; n++) {
      runner.addTurn({ turnNumber: n, intents: byNumber.get(n)?.intents ?? [] });
      runner.executeNextTick();
    }
    expect(stateHash(runner.game)).toBe(finalHash);
  }, LONG);
});

describe("ObservationEncoder", () => {
  let env: ReturnType<typeof makeEnv>;
  let rows: SeatStep[];
  beforeAll(async () => {
    env = makeEnv({ seats: 3, bots: 20 });
    rows = await env.reset("encoder-1");
    const rand = seededRandom(11);
    for (let i = 0; i < 60 && !env.gameOver; i++) {
      rows = env.step(rows.map((r) => (r.active ? randomAction(r.masks, rand) : NOOP_ACTION)));
    }
  }, LONG);

  it("keeps every feature finite and in [0, 1]", () => {
    for (const row of rows) {
      for (const v of row.obs.features) {
        expect(Number.isFinite(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });

  it("partitions land ownership in every land cell", () => {
    const raster = rows[0].obs.raster;
    const cells = 128 * 64;
    let landCells = 0;
    for (let c = 0; c < cells; c++) {
      if (raster[c] === 0) continue;
      landCells++;
      const sum = raster[1 * cells + c] + raster[2 * cells + c] + raster[3 * cells + c] + raster[4 * cells + c];
      if (raster[c] > 64) expect(Math.abs(sum - 255)).toBeLessThanOrEqual(4);
    }
    expect(landCells).toBeGreaterThan(1000);
  });

  it("gives each seat the same answer as encoding it alone", () => {
    for (let s = 0; s < env.numSeats; s++) {
      const alone = { raster: new Uint8Array(rows[s].obs.raster.length), features: new Float32Array(rows[s].obs.features.length) };
      env.encoder.encodeOne(env.game, env.seatIDs[s], alone);
      expect(Buffer.from(alone.raster).equals(Buffer.from(rows[s].obs.raster))).toBe(true);
      expect(Array.from(alone.features)).toEqual(Array.from(rows[s].obs.features));
    }
  });

  it("reports attacks and army size from the engine", () => {
    for (let s = 0; s < env.numSeats; s++) {
      const p = env.game.player(env.seatIDs[s]);
      const f = rows[s].obs.features;
      expect(f[GLOBAL_OFFSET + GlobalField.IncomingCount]).toBeCloseTo(Math.min(1, p.incomingAttacks().length / 4), 5);
      expect(f[GLOBAL_OFFSET + GlobalField.OutgoingCount]).toBeCloseTo(Math.min(1, p.outgoingAttacks().length / 4), 5);
      if (p.isAlive()) {
        const ratio = p.troops() / env.game.config().maxTroops(p);
        expect(f[GLOBAL_OFFSET + GlobalField.TroopRatio]).toBeCloseTo(Math.min(1, ratio), 5);
      }
    }
  });
});
