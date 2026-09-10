import { describe, expect, it } from "vitest";
import type { Game, PlayerID } from "../../OpenFrontIO/src/core/game/Game";
import { GOLD_SCALE, Reward, TROOP_SCALE } from "../src/Reward";
import { TurnFeeder } from "../src/TurnFeeder";
import { AGENT_ID, buildTestRunner } from "./helpers/buildTestRunner";

/**
 * Minimal game double: Reward only reads map().numLandTiles() once,
 * player() for the agent's own stats, and players() for the kill scan.
 * A live engine kill is impractical in unit tests (test games seat no
 * opponents), so the kill bonus is tested here instead of in GameEnv.
 */
function stubGame(opponentAlive: Record<string, boolean>) {
  const agent = {
    numTilesOwned: () => 0,
    troops: () => 0,
    gold: () => 0n,
  };
  const game = {
    map: () => ({ numLandTiles: () => 1000 }),
    player: (_id: string) => agent,
    players: () =>
      Object.entries(opponentAlive).map(([id, alive]) => ({
        id: () => id,
        isAlive: () => alive,
      })),
  } as unknown as Game;
  return { game, agentId: AGENT_ID as PlayerID };
}

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

  it("pays the kill bonus once per eliminated opponent", () => {
    const alive: Record<string, boolean> = { bot1: true, bot2: true };
    const { game, agentId } = stubGame(alive);
    const reward = new Reward({
      weights: { land: 0, troop: 0, gold: 0, win: 0, death: 0, kill: 2 },
    });
    reward.init(game, agentId);

    alive.bot1 = false;
    expect(reward.step(game, agentId, null)).toBeCloseTo(2, 10);

    // Same state again: the elimination was already counted.
    expect(reward.step(game, agentId, null)).toBeCloseTo(0, 10);

    alive.bot2 = false;
    expect(reward.step(game, agentId, null)).toBeCloseTo(2, 10);
  });

  it("never counts the agent's own death as a kill", () => {
    const alive: Record<string, boolean> = { bot1: true };
    const { game, agentId } = stubGame(alive);
    const reward = new Reward({
      weights: { land: 0, troop: 0, gold: 0, win: 0, death: -3, kill: 2 },
    });
    reward.init(game, agentId);

    expect(reward.step(game, agentId, "death")).toBeCloseTo(-3, 10);
  });

  it("pays nothing for kills under default weights", () => {
    const alive: Record<string, boolean> = { bot1: true };
    const { game, agentId } = stubGame(alive);
    const reward = new Reward();
    reward.init(game, agentId);

    // Old 5-value configs behave exactly as before: eliminations pass
    // through silently until a sweep opts into the bonus.
    alive.bot1 = false;
    expect(reward.step(game, agentId, null)).toBeCloseTo(0, 10);
  });
});
