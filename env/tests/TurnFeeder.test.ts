import { describe, expect, it, vi } from "vitest";
import { TurnFeeder } from "../src/TurnFeeder";
import { buildTestRunner } from "./helpers/buildTestRunner";

describe("TurnFeeder", () => {
  it("advances game.ticks() by exactly N for an N-tick step", async () => {
    const { runner } = await buildTestRunner("turnfeeder-basic");
    const feeder = new TurnFeeder(runner);
    const before = runner.game.ticks();

    const result = feeder.step(10, []);

    expect(result.ok).toBe(true);
    expect(result.ticksExecuted).toBe(10);
    expect(runner.game.ticks()).toBe(before + 10);
  });

  it("carries intents only on the final tick of the step", async () => {
    const { runner } = await buildTestRunner("turnfeeder-intents");
    const feeder = new TurnFeeder(runner);
    const addTurnSpy = vi.spyOn(runner, "addTurn");

    const map = runner.game.map();
    let spawnTile = -1;
    for (let i = 0; i < map.width() * map.height(); i++) {
      if (map.isLand(i) && !map.hasOwner(i)) {
        spawnTile = i;
        break;
      }
    }
    expect(spawnTile).toBeGreaterThanOrEqual(0);
    const intents = [
      { clientID: "client0", playerID: "agent0", type: "spawn", tile: spawnTile },
    ] as any;

    const result = feeder.step(5, intents);

    expect(result.ok).toBe(true);
    expect(result.ticksExecuted).toBe(5);
    expect(addTurnSpy).toHaveBeenCalledTimes(5);
    for (let i = 0; i < 4; i++) {
      expect(addTurnSpy.mock.calls[i][0].intents).toEqual([]);
    }
    expect(addTurnSpy.mock.calls[4][0].intents).toEqual(intents);
  });

  it("advances turn numbers monotonically across multiple steps", async () => {
    const { runner } = await buildTestRunner("turnfeeder-monotonic");
    const feeder = new TurnFeeder(runner);

    feeder.step(3, []);
    feeder.step(4, []);
    const before = runner.game.ticks();
    feeder.step(1, []);

    expect(runner.game.ticks()).toBe(before + 1);
  });

  it("rejects a step of fewer than 1 tick", async () => {
    const { runner } = await buildTestRunner("turnfeeder-invalid");
    const feeder = new TurnFeeder(runner);

    expect(() => feeder.step(0, [])).toThrow();
  });
});
