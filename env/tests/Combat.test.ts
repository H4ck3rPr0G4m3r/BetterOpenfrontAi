import { describe, expect, it } from "vitest";
import { CombatTracker } from "../src/Combat";

const player = (smallID: number, troops: number, extra: Record<string, unknown> = {}) =>
  ({ smallID: () => smallID, troops: () => troops, isAlive: () => true, ...extra }) as never;

describe("CombatTracker", () => {
  it("counts only tiles taken from stronger players", () => {
    const t = new CombatTracker(1);
    t.onConquer(player(1, 100), player(2, 500));
    t.onConquer(player(1, 100), player(3, 50));
    t.onConquer(player(1, 100), null);
    t.onConquer(player(4, 100), player(2, 500));
    expect(t.take()).toEqual({ strongTiles: 1, repelled: 0 });
    expect(t.take()).toEqual({ strongTiles: 0, repelled: 0 });
  });

  it("credits a repelled stronger attack, not a lost one or a probe", () => {
    const t = new CombatTracker(1);
    const enemy = player(2, 5000);
    let attacks: unknown[] = [];
    const me = player(1, 1000, { numTilesOwned: () => 200, incomingAttacks: () => attacks });
    const attack = (id: string, troops: number) => ({ id: () => id, troops: () => troops, attacker: () => enemy });
    attacks = [attack("held", 800), attack("lost", 800), attack("probe", 50)];
    t.update(me);
    // "lost" takes 20 of our 200 tiles (10%), "held" takes nothing new.
    for (let i = 0; i < 20; i++) t.onConquer(enemy, me);
    attacks = [attack("held", 800)];
    t.update(me);
    expect(t.take().repelled).toBe(0); // "lost" lost land, "probe" was a probe
    attacks = [];
    t.update(me);
    expect(t.take().repelled).toBe(0); // shares the attacker's tile count with "lost"
  });

  it("credits a held line when the stronger attacker takes almost nothing", () => {
    const t = new CombatTracker(1);
    const enemy = player(2, 5000);
    let attacks: unknown[] = [{ id: () => "a", troops: () => 900, attacker: () => enemy }];
    const me = player(1, 1000, { numTilesOwned: () => 200, incomingAttacks: () => attacks });
    t.update(me);
    t.onConquer(enemy, me); // 1 of 200 tiles
    attacks = [];
    t.update(me);
    expect(t.take().repelled).toBe(1);
  });
});
