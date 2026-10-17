import { beforeAll, describe, expect, it } from "vitest";
import { UnitType } from "../../OpenFrontIO/src/core/game/Game";
import { ActionCodecImpl } from "../src/ActionCodec";
import type { SeatStep } from "../src/GameEnv";
import { cellGeometry, tileCell } from "../src/Geometry";
import {
  createMasks,
  GLOBAL_OFFSET,
  GlobalField,
  IntentType,
  Plane,
  RASTER_CELLS,
  RASTER_WIDTH,
  spatialRow,
} from "../src/Layout";
import { nukeEvents } from "../src/Reward";
import { NOOP_ACTION } from "../src/types";
import { makeEnv, openIndices } from "./helpers/env";

const LONG = 180_000;
const NUKES = [IntentType.NukeAtom, IntentType.NukeHydrogen, IntentType.NukeMIRV];

describe("nukes in a real game", () => {
  const codec = new ActionCodecImpl();
  let env: ReturnType<typeof makeEnv>;
  let rows: SeatStep[];

  beforeAll(async () => {
    env = makeEnv({ seats: 1, bots: 10 });
    rows = await env.reset("nukes-1");
    // Past spawn immunity (nukes are refused until then).
    for (let i = 0; i < 8; i++) rows = env.step([NOOP_ACTION]);
  }, LONG);

  const masksNow = () => {
    const m = createMasks();
    codec.masks(env.game, env.seatIDs[0], m);
    return m;
  };

  it("keeps every bomb shut without a silo", () => {
    env.game.player(env.seatIDs[0]).addGold(100_000_000n);
    const m = masksNow();
    for (const intent of NUKES) expect(m.intent[intent]).toBe(0);
    expect(openIndices(spatialRow(m, 2))).toEqual([]);
  });

  it("opens the bombs it can afford once a silo stands, aims only at hostile land, and pays a hit", () => {
    const me = env.game.player(env.seatIDs[0]);
    const home = [...me.tiles()][0];
    me.buildUnit(UnitType.MissileSilo, home, {});
    // Enough for an atom and a hydrogen bomb, not a MIRV.
    me.removeGold(me.gold());
    me.addGold(6_000_000n);
    const m = masksNow();
    expect(m.intent[IntentType.NukeAtom]).toBe(1);
    expect(m.intent[IntentType.NukeHydrogen]).toBe(1);
    expect(m.intent[IntentType.NukeMIRV]).toBe(0);

    const cells = openIndices(spatialRow(m, 2));
    expect(cells.length).toBeGreaterThan(0);
    const map = env.game.map();
    const geometry = cellGeometry(map.width(), map.height());
    for (const cell of cells) {
      const owner = map.ownerID(geometry.centerTile[cell]);
      expect(owner).not.toBe(0);
      expect(owner).not.toBe(me.smallID());
    }
    // The open cell farthest from home, so the blast cannot touch our land.
    const homeCell = tileCell(home, geometry);
    const dist = (c: number) => Math.hypot((c % RASTER_WIDTH) - (homeCell % RASTER_WIDTH), Math.floor(c / RASTER_WIDTH) - Math.floor(homeCell / RASTER_WIDTH));
    const cell = cells.reduce((a, b) => (dist(b) > dist(a) ? b : a));

    const action = { ...NOOP_ACTION, intentType: IntentType.NukeAtom, spatialCell: cell };
    const intents = codec.decode(action, env.game, env.seatIDs[0]) as { type: string; unit: string; tile: number }[];
    expect(intents).toHaveLength(1);
    expect(intents[0].type).toBe("build_unit");
    expect(intents[0].unit).toBe(UnitType.AtomBomb);
    // The intent carries the target, not the silo: the engine picks the silo.
    expect(map.ownerID(intents[0].tile)).not.toBe(me.smallID());
    expect(tileCell(intents[0].tile, geometry)).toBe(cell);

    rows = env.step([action]);
    rows = env.step([NOOP_ACTION]);
    const raster = rows[0].obs.raster;
    expect(raster.subarray(Plane.OwnNukes * RASTER_CELLS, (Plane.OwnNukes + 1) * RASTER_CELLS).some((v) => v > 0)).toBe(true);
    expect(rows[0].obs.features[GLOBAL_OFFSET + GlobalField.SilosReady]).toBe(0); // reloading

    // Fly it home. The seat's own reward term records the hit.
    const reward = (env as unknown as { seats: { reward: { terms: Record<string, number> } }[] }).seats[0].reward;
    for (let i = 0; i < 60 && reward.terms.nuke === undefined && !env.gameOver; i++) rows = env.step([NOOP_ACTION]);
    expect(reward.terms.nuke).toBeGreaterThan(0);
  }, LONG);
});

describe("SAM cover", () => {
  it("shows hostile SAM range on the seat's map, not its own SAMs", async () => {
    const env = makeEnv({ seats: 1, bots: 10 });
    let rows = await env.reset("sam-1");
    const game = env.game;
    const me = game.player(env.seatIDs[0]);
    const nation = game.players().find((p) => p !== me && p.numTilesOwned() > 0 && p.type() === "NATION")!;
    const theirs = [...nation.tiles()][0];
    const mine = [...me.tiles()][0];
    nation.buildUnit(UnitType.SAMLauncher, theirs, {});
    me.buildUnit(UnitType.SAMLauncher, mine, {});
    rows = env.step([NOOP_ACTION]);
    const plane = rows[0].obs.raster.subarray(Plane.EnemySAMCover * RASTER_CELLS, (Plane.EnemySAMCover + 1) * RASTER_CELLS);
    const geometry = cellGeometry(game.map().width(), game.map().height());
    expect(plane[tileCell(theirs, geometry)]).toBe(255);
    expect(plane[tileCell(mine, geometry)]).toBe(0);
  }, LONG);
});

describe("nuke reward bookkeeping", () => {
  const fakeNuke = (id: number, reached: boolean) => ({
    id: () => id,
    type: () => UnitType.AtomBomb,
    targetTile: () => 0,
    reachedTarget: () => reached,
  });
  const player = (nukes: unknown[]) => ({ isAlive: () => true, units: () => nukes }) as never;

  it("pays a nuke that landed and nothing for one that was shot down", () => {
    const known = new Map();
    const landed = fakeNuke(1, true);
    const intercepted = fakeNuke(2, false);
    known.set(1, { unit: landed, damage: { hostile: 4, friendly: 1 } });
    known.set(2, { unit: intercepted, damage: { hostile: 7, friendly: 3 } });
    expect(nukeEvents({} as never, player([]), known)).toEqual({ hostile: 4, friendly: 1 });
    expect(known.size).toBe(0);
  });
});
