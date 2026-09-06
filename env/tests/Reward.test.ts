import { describe, expect, it } from "vitest";
import { GOLD_SCALE, Reward, TROOP_SCALE } from "../src/Reward";
import { TurnFeeder } from "../src/TurnFeeder";
import { AGENT_ID, buildTestRunner } from "./helpers/buildTestRunner";

describe("Reward", () => {
  it("is zero when nothing changed between init() and step()", async () => {
    const { runner } = await buildTestRunner("reward-flat");
    const reward = new Reward();
    reward.init(runner.game, AGENT_ID);

    expect(reward.step(runner.game, AGENT_ID, null)).toBe(0);
  });

  it("rewards gaining land relative to the previous snapshot", async () => {
    const { runner } = await buildTestRunner("reward-land");
    const reward = new Reward();
    reward.init(runner.game, AGENT_ID); // pre-spawn: 0 tiles owned

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
    while (runner.game.inSpawnPhase()) feeder.step(1, []);

    const r = reward.step(runner.game, AGENT_ID, null);
    expect(r).toBeGreaterThan(0);
  });

  it("adds the terminal bonus on top of shaping", async () => {
    const { runner } = await buildTestRunner("reward-terminal");
    const flat = new Reward();
    flat.init(runner.game, AGENT_ID);
    expect(flat.step(runner.game, AGENT_ID, "win")).toBeCloseTo(1, 10);

    const flat2 = new Reward();
    flat2.init(runner.game, AGENT_ID);
    expect(flat2.step(runner.game, AGENT_ID, "death")).toBeCloseTo(-1, 10);

    const flat3 = new Reward();
    flat3.init(runner.game, AGENT_ID);
    expect(flat3.step(runner.game, AGENT_ID, "tick_cap")).toBeCloseTo(0, 10);
  });

  it("matches a hand-computed shaping value using the fixed scale constants", async () => {
    // Regression guard for the reward-normalization defect caught in review:
    // troop/gold deltas must divide by fixed constants, not a per-step
    // maximum over living players (which would make this value depend on
    // opponents' state, not just the agent's own transition).
    const { runner } = await buildTestRunner("reward-fixed-scale");
    const reward = new Reward();
    reward.init(runner.game, AGENT_ID);
    const preTroops = runner.game.player(AGENT_ID).troops();
    const preGold = Number(runner.game.player(AGENT_ID).gold());

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
    while (runner.game.inSpawnPhase()) feeder.step(1, []);

    const agent = runner.game.player(AGENT_ID);
    const expected =
      1.0 * (agent.numTilesOwned() / map.numLandTiles() - 0) +
      0.1 * ((agent.troops() - preTroops) / TROOP_SCALE) +
      0.1 * ((Number(agent.gold()) - preGold) / GOLD_SCALE);

    expect(reward.step(runner.game, AGENT_ID, null)).toBeCloseTo(expected, 6);
  });
});
