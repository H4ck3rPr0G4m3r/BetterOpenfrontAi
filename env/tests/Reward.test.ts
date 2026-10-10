import { describe, expect, it } from "vitest";
import { normalizeRewardWeights } from "../src/BehaviorConfig";
import { LAND_REF, NO_EVENTS, placementScore, Reward } from "../src/Reward";

interface FakePlayer {
  id: string;
  tiles: number;
  alive: boolean;
  structures: { level: number; building?: boolean }[];
  /** Share of MAX_TROOPS; defaults to full (no low-troop penalty). */
  troops?: number;
}

const MAX_TROOPS = 1000;

function fakeGame(players: FakePlayer[]) {
  const wrap = (p: FakePlayer) => ({
    id: () => p.id,
    isAlive: () => p.alive,
    numTilesOwned: () => p.tiles,
    troops: () => (p.troops ?? 1) * MAX_TROOPS,
    units: () => p.structures.map((s) => ({ level: () => s.level, isUnderConstruction: () => !!s.building })),
  });
  return {
    player: (id: string) => wrap(players.find((p) => p.id === id)!),
    players: () => players.filter((p) => p.alive).map(wrap),
    config: () => ({ maxTroops: () => MAX_TROOPS }),
    railNetwork: () => ({ stationManager: () => ({ findStation: () => null }) }),
  } as never;
}

describe("Reward v3", () => {
  const land = (tiles: number) => Math.log1p(tiles / LAND_REF);

  it("pays the change in land potential, not raw land", () => {
    const me: FakePlayer = { id: "me", tiles: 50, alive: true, structures: [] };
    const game = fakeGame([me]);
    const r = new Reward({ structures: 0, cities: 0, popCap: 0 });
    r.init(game, "me");
    me.tiles = 500;
    expect(r.step(game, "me", null, NO_EVENTS)).toBeCloseTo(land(500) - land(50));
    // Holding still pays nothing: idling never mints return.
    expect(r.step(game, "me", null, NO_EVENTS)).toBeCloseTo(0);
  });

  it("makes dying give back everything built, plus the death penalty", () => {
    const me: FakePlayer = { id: "me", tiles: 52, alive: true, structures: [{ level: 2 }] };
    const game = fakeGame([me]);
    const r = new Reward({ cities: 0, popCap: 0 });
    r.init(game, "me");
    let total = 0;
    me.tiles = 20_000;
    total += r.step(game, "me", null, NO_EVENTS);
    expect(total).toBeGreaterThan(4);
    me.tiles = 0;
    me.alive = false;
    total += r.step(game, "me", "death", NO_EVENTS);
    const spawnPotential = land(52) + 0.2 * Math.log1p(2);
    expect(total).toBeCloseTo(-spawnPotential - 2);
  });

  it("credits only the kills it is told about", () => {
    const me: FakePlayer = { id: "me", tiles: 100, alive: true, structures: [] };
    const game = fakeGame([me]);
    const r = new Reward({ kill: 0.5 });
    r.init(game, "me");
    expect(r.step(game, "me", null, NO_EVENTS)).toBeCloseTo(0);
    expect(r.step(game, "me", null, { ...NO_EVENTS, kills: 2 })).toBeCloseTo(1);
  });

  it("ignores structures under construction and counts levels", () => {
    const me: FakePlayer = { id: "me", tiles: 100, alive: true, structures: [{ level: 1, building: true }] };
    const game = fakeGame([me]);
    const r = new Reward({ land: 0, structures: 1, cities: 0, popCap: 0 });
    r.init(game, "me");
    expect(r.step(game, "me", null, NO_EVENTS)).toBeCloseTo(0);
    me.structures = [{ level: 3 }];
    expect(r.step(game, "me", null, NO_EVENTS)).toBeCloseTo(Math.log1p(3));
  });

  it("scores the tick cap by land rank among the living", () => {
    const me: FakePlayer = { id: "me", tiles: 300, alive: true, structures: [] };
    const others: FakePlayer[] = [
      { id: "a", tiles: 1000, alive: true, structures: [] },
      { id: "b", tiles: 100, alive: true, structures: [] },
      { id: "c", tiles: 5000, alive: false, structures: [] },
    ];
    const game = fakeGame([me, ...others]);
    expect(placementScore(game, (game as any).player("me"))).toBeCloseTo(0.5);
    expect(placementScore(game, (game as any).player("a"))).toBeCloseTo(1);
    const r = new Reward({ placement: 2 });
    r.init(game, "me");
    expect(r.step(game, "me", "tick_cap", NO_EVENTS)).toBeCloseTo(1);
  });

  it("pays win and nothing for sim errors or aborts", () => {
    const me: FakePlayer = { id: "me", tiles: 100, alive: true, structures: [] };
    const game = fakeGame([me]);
    const r = new Reward({ win: 5 });
    r.init(game, "me");
    expect(r.step(game, "me", "win", NO_EVENTS)).toBeCloseTo(5);
    expect(r.step(game, "me", "sim_error", NO_EVENTS)).toBeCloseTo(0);
    expect(r.step(game, "me", "aborted", NO_EVENTS)).toBeCloseTo(0);
  });

  it("charges for sitting below the troop floor, more the lower it goes", () => {
    const me: FakePlayer = { id: "me", tiles: 100, alive: true, structures: [], troops: 0.5 };
    const game = fakeGame([me]);
    const r = new Reward({ lowTroops: -0.02, lowTroopsBelow: 0.3 });
    r.init(game, "me");
    expect(r.step(game, "me", null, NO_EVENTS)).toBeCloseTo(0);
    me.troops = 0.15;
    expect(r.step(game, "me", null, NO_EVENTS)).toBeCloseTo(-0.01);
    me.troops = 0;
    expect(r.step(game, "me", null, NO_EVENTS)).toBeCloseTo(-0.02);
  });

  it("prices betrayal and diplomatic spam", () => {
    const me: FakePlayer = { id: "me", tiles: 100, alive: true, structures: [] };
    const game = fakeGame([me]);
    const r = new Reward({ betrayal: -1, diplomacy: -0.005 });
    r.init(game, "me");
    expect(r.step(game, "me", null, { ...NO_EVENTS, betrayals: 1, diplomacy: 2 })).toBeCloseTo(-1.01);
  });

  it("pays extra for land taken from weaker players and for repelled attacks", () => {
    const me: FakePlayer = { id: "me", tiles: 100, alive: true, structures: [] };
    const game = fakeGame([me]);
    const r = new Reward({ advantage: 0.5, defend: 0.3 });
    r.init(game, "me");
    me.tiles = 110;
    const plain = land(110) - land(100);
    // 10 tiles from a weaker player: the land gain plus half again (approximately).
    const step = r.step(game, "me", null, { ...NO_EVENTS, advantageTiles: 10, repelled: 1 });
    expect(step).toBeCloseTo(plain + (0.5 * 10) / (LAND_REF + 110) + 0.3);
  });

  it("values the income rate as a potential, given back on death", () => {
    const me: FakePlayer = { id: "me", tiles: 100, alive: true, structures: [] };
    const game = fakeGame([me]);
    const r = new Reward({ land: 0, structures: 0, cities: 0, popCap: 0, income: 1, death: 0 });
    r.init(game, "me");
    let total = 0;
    // 120 steps at 5k gold a step: 300k a minute once the average settles.
    for (let i = 0; i < 120; i++) total += r.step(game, "me", null, { ...NO_EVENTS, earned: 5_000 });
    const settled = Math.log1p(300_000 / 100_000);
    expect(total).toBeGreaterThan(0.8 * settled);
    expect(total).toBeLessThan(settled);
    me.alive = false;
    total += r.step(game, "me", "death", NO_EVENTS);
    expect(total).toBeCloseTo(0);
  });

  it("values cities on top of other structures", () => {
    const me: FakePlayer = { id: "me", tiles: 100, alive: true, structures: [] };
    const game = fakeGame([me]);
    const r = new Reward({ land: 0, structures: 0.2, cities: 0.5, popCap: 0 });
    r.init(game, "me");
    me.structures = [{ level: 2 }];
    expect(r.step(game, "me", null, NO_EVENTS)).toBeCloseTo(0.7 * Math.log1p(2));
  });

  it("pays for a higher population cap and gives it back on death", () => {
    const me: FakePlayer = { id: "me", tiles: 100, alive: true, structures: [] };
    let max = 120_000;
    const game = { ...(fakeGame([me]) as object), config: () => ({ maxTroops: () => max }) } as never;
    const r = new Reward({ land: 0, structures: 0, cities: 0, popCap: 0.3, lowTroops: 0 });
    r.init(game, "me");
    max = 370_000; // one city level
    expect(r.step(game, "me", null, NO_EVENTS)).toBeCloseTo(0.3 * (Math.log1p(3.7) - Math.log1p(1.2)));
    me.alive = false;
    expect(r.step(game, "me", "death", NO_EVENTS)).toBeCloseTo(-0.3 * Math.log1p(3.7) - 2);
  });

  it("values cities and ports on the rail network", () => {
    const me: FakePlayer = { id: "me", tiles: 100, alive: true, structures: [] };
    const units = [1, 2, 3].map((id) => ({ id, level: () => 1, isUnderConstruction: () => false }));
    const connected = new Set<number>();
    const base = fakeGame([me]) as unknown as { player: (id: string) => object };
    const game = {
      ...base,
      player: (id: string) => ({ ...base.player(id), units: () => units }),
      railNetwork: () => ({ stationManager: () => ({ findStation: (u: { id: number }) => (connected.has(u.id) ? {} : null) }) }),
    } as never;
    const r = new Reward({ land: 0, structures: 0, cities: 0, popCap: 0, income: 0, rail: 0.5 });
    r.init(game, "me");
    connected.add(1).add(2);
    expect(r.step(game, "me", null, NO_EVENTS)).toBeCloseTo(0.5 * Math.log1p(2));
  });

  it("validates weight overrides by name", () => {
    expect(normalizeRewardWeights({ land: 2 }).land).toBe(2);
    expect(() => normalizeRewardWeights({ lnad: 2 })).toThrow(/unknown reward weight/);
    expect(() => normalizeRewardWeights({ land: Number.NaN })).toThrow(/finite/);
  });
});
