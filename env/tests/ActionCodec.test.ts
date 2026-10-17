import { beforeAll, describe, expect, it } from "vitest";
import { ActionCodecImpl } from "../src/ActionCodec";
import { UnitType } from "../../OpenFrontIO/src/core/game/Game";
import { BUILD_UNIT_TYPES, buildCandidates } from "../src/BuildSites";
import type { SeatStep } from "../src/GameEnv";
import {
  BUILD_TYPES,
  createMasks,
  HEAD_USAGE,
  IntentType,
  MASK_SIZE,
  NUM_INTENTS,
  NUM_RATIO_BUCKETS,
  NUM_SLOTS,
  RATIO_BUCKETS,
  siteRow,
  targetRow,
} from "../src/Layout";
import { NOOP_ACTION, type Action } from "../src/types";
import { makeEnv, openIndices, randomAction, seededRandom } from "./helpers/env";

const LONG = 180_000;
const TARGETED = HEAD_USAGE.map((u, i) => (u.target ? i : -1)).filter((i) => i >= 0);

describe("layout", () => {
  it("lines up mask views with the flat buffer", () => {
    const m = createMasks();
    expect(m.flat.length).toBe(MASK_SIZE);
    expect(m.intent.length + m.target.length + m.ratio.length + m.spatial.length + m.site.length).toBe(MASK_SIZE);
    expect(m.target.length).toBe(NUM_INTENTS * NUM_SLOTS);
    expect(BUILD_UNIT_TYPES.map(String)).toEqual([...BUILD_TYPES]);
  });
});

describe("ActionCodec masks and decode", () => {
  const codec = new ActionCodecImpl();
  let env: ReturnType<typeof makeEnv>;
  const snapshots: SeatStep[][] = [];

  beforeAll(async () => {
    env = makeEnv({ seats: 2, bots: 20 });
    let rows = await env.reset("codec-1");
    const rand = seededRandom(5);
    snapshots.push(rows);
    for (let i = 0; i < 80 && !env.gameOver; i++) {
      rows = env.step(rows.map((r) => (r.active ? randomAction(r.masks, rand) : NOOP_ACTION)));
    }
  }, LONG);

  const decodeAll = (check: (seat: number, action: Action, intents: unknown[]) => void) => {
    for (let s = 0; s < env.numSeats; s++) {
      const masks = createMasks();
      codec.masks(env.game, env.seatIDs[s], masks);
      for (const intent of TARGETED) {
        const row = targetRow(masks, intent);
        expect(masks.intent[intent]).toBe(row.some((v) => v === 1) ? 1 : 0);
        for (let slot = 0; slot < NUM_SLOTS; slot++) {
          const action = { ...NOOP_ACTION, intentType: intent, target: slot, ratioBucket: 0 };
          const intents = codec.decode(action, env.game, env.seatIDs[s]);
          check(s, action, intents);
          if (row[slot]) expect(intents.length, `intent ${intent} slot ${slot} open but void`).toBe(1);
          else expect(intents, `intent ${intent} slot ${slot} closed but fired`).toEqual([]);
        }
      }
      for (const kind of [0, 1]) {
        const intent = kind === 0 ? IntentType.BuildUnit : IntentType.UpgradeStructure;
        const sites = siteRow(masks, kind);
        for (let site = 0; site < sites.length; site++) {
          const intents = codec.decode({ ...NOOP_ACTION, intentType: intent, buildSite: site }, env.game, env.seatIDs[s]);
          expect(intents.length).toBe(sites[site] ? 1 : 0);
        }
      }
    }
  };

  it("fires exactly the open targets and nothing else", () => {
    decodeAll(() => {});
  });

  it("never retargets an attack onto a slot it did not pick", () => {
    decodeAll((seat, action, intents) => {
      if (action.intentType !== IntentType.Attack || intents.length === 0) return;
      const fired = intents[0] as { targetID: string | null };
      if (action.target === 0) expect(fired.targetID).toBeNull();
      else expect(fired.targetID).not.toBeNull();
    });
  });

  it("stamps the seat's own client id on every intent", () => {
    for (let s = 0; s < env.numSeats; s++) {
      const masks = createMasks();
      codec.masks(env.game, env.seatIDs[s], masks);
      if (!masks.intent[IntentType.Attack]) continue;
      const slot = openIndices(targetRow(masks, IntentType.Attack))[0];
      const [intent] = codec.decode({ ...NOOP_ACTION, intentType: IntentType.Attack, target: slot }, env.game, env.seatIDs[s]);
      expect((intent as unknown as { clientID: string }).clientID).toBe(env.game.player(env.seatIDs[s]).clientID());
    }
  });

  it("keeps a troop reserve: ratios that would empty the seat are shut and never fire", () => {
    for (const id of env.seatIDs) {
      const masks = createMasks();
      codec.masks(env.game, id, masks);
      const player = env.game.player(id);
      if (!player.isAlive()) continue;
      const max = env.game.config().maxTroops(player);
      expect(masks.ratio[0]).toBe(1);
      for (let b = 1; b < NUM_RATIO_BUCKETS; b++) {
        expect(masks.ratio[b]).toBe(player.troops() * (1 - RATIO_BUCKETS[b]) >= 0.1 * max ? 1 : 0);
      }
      expect(masks.ratio[NUM_RATIO_BUCKETS - 1]).toBe(0); // sending everything is never allowed
      if (!masks.intent[IntentType.Attack]) continue;
      const slot = openIndices(targetRow(masks, IntentType.Attack))[0];
      const fired = codec.decode({ ...NOOP_ACTION, intentType: IntentType.Attack, target: slot, ratioBucket: NUM_RATIO_BUCKETS - 1 }, env.game, id);
      expect(fired).toEqual([]);
    }
  });

  it("keeps breaking alliances off unless allowed", () => {
    for (const id of env.seatIDs) {
      const masks = createMasks();
      codec.masks(env.game, id, masks);
      expect(masks.intent[IntentType.BreakAlliance]).toBe(0);
      expect(targetRow(masks, IntentType.BreakAlliance).every((v) => v === 0)).toBe(true);
      const fired = codec.decode({ ...NOOP_ACTION, intentType: IntentType.BreakAlliance, target: 1 }, env.game, id);
      expect(fired).toEqual([]);
    }
  });

  it("keeps embargoes off unless allowed", () => {
    for (const id of env.seatIDs) {
      const masks = createMasks();
      codec.masks(env.game, id, masks);
      expect(masks.intent[IntentType.Embargo]).toBe(0);
      const fired = codec.decode({ ...NOOP_ACTION, intentType: IntentType.Embargo, target: 1 }, env.game, id);
      expect(fired).toEqual([]);
    }
  });

  it("flags build sites that join the rail network exactly like the engine", () => {
    const max = env.game.config().trainStationMaxRange();
    for (const id of env.seatIDs) {
      const player = env.game.player(id);
      if (!player.isAlive()) continue;
      for (const c of buildCandidates(env.game, player)) {
        if (c.type === UnitType.City || c.type === UnitType.Port) {
          expect(c.rail).toBe(env.game.hasUnitNearby(c.tile, max, UnitType.Factory));
        } else if (c.type !== UnitType.Factory) {
          expect(c.rail).toBe(false);
        }
      }
    }
  });

  it("keeps donations off unless allowed", () => {
    for (const id of env.seatIDs) {
      const masks = createMasks();
      codec.masks(env.game, id, masks);
      expect(masks.intent[IntentType.DonateGold]).toBe(0);
      expect(masks.intent[IntentType.DonateTroops]).toBe(0);
    }
    const allowed = new ActionCodecImpl({ allowDonations: true });
    expect(() => allowed.masks(env.game, env.seatIDs[0], createMasks())).not.toThrow();
  });

  it("offers each structure only where it belongs (defense posts behind a land front)", () => {
    const map = env.game.map();
    for (const id of env.seatIDs) {
      const player = env.game.player(id);
      if (!player.isAlive()) continue;
      for (const c of buildCandidates(env.game, player)) {
        const ns = map.neighbors(c.tile);
        if (c.type === UnitType.DefensePost) {
          // Behind the front: our own land, no enemy within 4 tiles, but one within reach of the post.
          const enemyWithin = (r: number) => {
            for (let dy = -r; dy <= r; dy++) {
              for (let dx = -r; dx <= r; dx++) {
                const x = map.x(c.tile) + dx;
                const y = map.y(c.tile) + dy;
                if (!map.isValidCoord(x, y)) continue;
                const o = map.ownerID(map.ref(x, y));
                if (map.isLand(map.ref(x, y)) && o !== 0 && o !== player.smallID()) return true;
              }
            }
            return false;
          };
          expect(map.ownerID(c.tile)).toBe(player.smallID());
          expect(enemyWithin(4)).toBe(false);
          expect(enemyWithin(20)).toBe(true);
        }
        if (c.type === UnitType.Port || c.type === UnitType.Warship) expect(ns.some((n) => !map.isLand(n))).toBe(true);
        if (c.type === UnitType.City) expect(c.border && ns.some((n) => !map.isLand(n))).toBe(false);
      }
    }
  });

  it("throws on out-of-range heads", () => {
    const id = env.seatIDs[0];
    expect(() => codec.decode({ ...NOOP_ACTION, intentType: NUM_INTENTS }, env.game, id)).toThrow(RangeError);
    expect(() => codec.decode({ ...NOOP_ACTION, target: NUM_SLOTS }, env.game, id)).toThrow(RangeError);
    expect(() => codec.decode({ ...NOOP_ACTION, spatialCell: 8192 }, env.game, id)).toThrow(RangeError);
  });

  it("keeps disabled intents and build types shut", () => {
    const limited = new ActionCodecImpl({ disabledIntents: [IntentType.Attack], disabledBuilds: [...BUILD_TYPES] });
    for (const id of env.seatIDs) {
      const masks = createMasks();
      limited.masks(env.game, id, masks);
      expect(masks.intent[IntentType.Attack]).toBe(0);
      expect(targetRow(masks, IntentType.Attack).every((v) => v === 0)).toBe(true);
      expect(masks.intent[IntentType.BuildUnit]).toBe(0);
      expect(siteRow(masks, 0).every((v) => v === 0)).toBe(true);
      expect(limited.decode({ ...NOOP_ACTION, intentType: IntentType.Attack, target: 0 }, env.game, id)).toEqual([]);
    }
    expect(() => new ActionCodecImpl({ disabledIntents: [0] })).toThrow();
    expect(() => new ActionCodecImpl({ disabledBuilds: ["Castle"] })).toThrow();
  });
});
