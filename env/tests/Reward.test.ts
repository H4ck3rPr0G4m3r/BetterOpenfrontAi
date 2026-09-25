import { describe, expect, it } from "vitest";
import type { Game, PlayerID } from "../../OpenFrontIO/src/core/game/Game";
import { GOLD_SCALE, Reward, TROOP_SCALE } from "../src/Reward";
import { TurnFeeder } from "../src/TurnFeeder";
import { AGENT_ID, buildTestRunner } from "./helpers/buildTestRunner";

/**
 * Minimal game double: Reward only reads map().numLandTiles() once,
 * player() for the agent's own stats, players() for the kill scan,
 * ticks() for the phase clock, and unitInfo() for the MIRV price.
 * A live engine kill is impractical in unit tests (test games seat no
 * opponents), so the kill bonus is tested here instead of in GameEnv.
 */
interface StubState {
  opponents: Record<string, boolean>;
  ticks?: number;
  tiles?: number;
  troops?: number;
  gold?: bigint;
  alive?: boolean;
  cities?: number;
  factories?: number;
  mirvCost?: number;
}

function stubGame(state: StubState) {
  const s = {
    ticks: 0,
    tiles: 0,
    troops: 0,
    gold: 0n,
    alive: true,
    cities: 0,
    factories: 0,
    mirvCost: 25_000_000,
    ...state,
  };
  const agent = {
    numTilesOwned: () => s.tiles,
    troops: () => s.troops,
    gold: () => s.gold,
    isAlive: () => s.alive,
    unitsOwned: (type: string) =>
      type === "City" ? s.cities : type === "Factory" ? s.factories : 0,
  };
  const game = {
    map: () => ({ numLandTiles: () => 1000 }),
    ticks: () => s.ticks,
    player: (_id: string) => agent,
    players: () =>
      Object.entries(s.opponents).map(([id, alive]) => ({
        id: () => id,
        isAlive: () => alive,
      })),
    unitInfo: (_type: string) => ({ cost: () => BigInt(s.mirvCost) }),
  } as unknown as Game;
  return { game, agentId: AGENT_ID as PlayerID, state: s };
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
    const { game, agentId } = stubGame({ opponents: alive });
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
    const { game, agentId } = stubGame({ opponents: alive });
    const reward = new Reward({
      weights: { land: 0, troop: 0, gold: 0, win: 0, death: -3, kill: 2 },
    });
    reward.init(game, agentId);

    expect(reward.step(game, agentId, "death")).toBeCloseTo(-3, 10);
  });

  it("pays nothing for kills under default weights", () => {
    const alive: Record<string, boolean> = { bot1: true };
    const { game, agentId } = stubGame({ opponents: alive });
    const reward = new Reward();
    reward.init(game, agentId);

    // Old 5-value configs behave exactly as before: eliminations pass
    // through silently until a sweep opts into the bonus.
    alive.bot1 = false;
    expect(reward.step(game, agentId, null)).toBeCloseTo(0, 10);
  });

  it("lets land follow into landEarly unless overridden", () => {
    // Backward compat: pre-phase configs price land without naming the
    // early price, and the early game must honor it.
    const stub = stubGame({ opponents: {}, tiles: 0 });
    const reward = new Reward({ weights: { land: 2 } });
    reward.init(stub.game, stub.agentId);
    stub.state.tiles = 10;
    expect(reward.step(stub.game, stub.agentId, null)).toBeCloseTo(0.02, 10);
  });

  it("prices early land with landEarly and late land with land", () => {
    const weights = {
      land: 1, troop: 0, gold: 0, win: 0, death: 0, kill: 0,
      landEarly: 3, landLoss: 1, city: 0, factory: 0,
      troopHold: 0, survive: 0, rich: 0,
    };
    const early = stubGame({ opponents: {}, tiles: 0 });
    const reward = new Reward({ weights });
    reward.init(early.game, early.agentId);
    early.state.tiles = 10;
    expect(reward.step(early.game, early.agentId, null)).toBeCloseTo(3 * 0.01, 10);

    const late = stubGame({ opponents: {}, tiles: 0, ticks: 5000 });
    const rewardLate = new Reward({ weights });
    // Phase counts from init: fast-forward past it, then gain land late.
    late.state.ticks = 0;
    rewardLate.init(late.game, late.agentId);
    late.state.ticks = 5000;
    late.state.tiles = 10;
    expect(rewardLate.step(late.game, late.agentId, null)).toBeCloseTo(0.01, 10);
  });

  it("penalizes early land losses extra through landLoss", () => {
    const stub = stubGame({ opponents: {}, tiles: 10 });
    const reward = new Reward({
      weights: {
        land: 1, troop: 0, gold: 0, win: 0, death: 0, kill: 0,
        landEarly: 2, landLoss: 2, city: 0, factory: 0,
        troopHold: 0, survive: 0, rich: 0,
      },
    });
    reward.init(stub.game, stub.agentId);
    stub.state.tiles = 4;
    expect(reward.step(stub.game, stub.agentId, null)).toBeCloseTo(-0.012, 10);
  });

  it("mutes negative troop deltas with troopLossScale but keeps gains", () => {
    const weights = {
      land: 0, troop: 0.1, gold: 0, win: 0, death: 0, kill: 0,
      landEarly: 0, landLoss: 1, city: 0, factory: 0,
      troopHold: 0, survive: 0, rich: 0,
    };
    // Symmetric default: spending 5000 of 10000 troops costs 0.005.
    const sym = stubGame({ opponents: {}, troops: 10000 });
    const symReward = new Reward({ weights });
    symReward.init(sym.game, sym.agentId);
    sym.state.troops = 5000;
    expect(symReward.step(sym.game, sym.agentId, null)).toBeCloseTo(-0.005, 10);

    // Muted: the same attack-like loss is free.
    const muted = stubGame({ opponents: {}, troops: 10000 });
    const mutedReward = new Reward({ weights, troopLossScale: 0 });
    mutedReward.init(muted.game, muted.agentId);
    muted.state.troops = 5000;
    expect(mutedReward.step(muted.game, muted.agentId, null)).toBeCloseTo(0, 10);

    // Gains still count in full under the mute.
    const gain = stubGame({ opponents: {}, troops: 10000 });
    const gainReward = new Reward({ weights, troopLossScale: 0 });
    gainReward.init(gain.game, gain.agentId);
    gain.state.troops = 20000;
    expect(gainReward.step(gain.game, gain.agentId, null)).toBeCloseTo(0.01, 10);

    expect(() => new Reward({ troopLossScale: 1.5 })).toThrow();
  });

  it("pays city and factory deltas only mid game", () => {
    const weights = {
      land: 0, troop: 0, gold: 0, win: 0, death: 0, kill: 0,
      landEarly: 0, landLoss: 1, city: 0.5, factory: 0.5,
      troopHold: 0, survive: 0, rich: 0,
    };
    const mid = stubGame({ opponents: {} });
    const reward = new Reward({ weights });
    reward.init(mid.game, mid.agentId);
    mid.state.ticks = 1000;
    mid.state.cities = 1;
    mid.state.factories = 1;
    expect(reward.step(mid.game, mid.agentId, null)).toBeCloseTo(1.0, 10);

    const early = stubGame({ opponents: {} });
    const rewardEarly = new Reward({ weights });
    rewardEarly.init(early.game, early.agentId);
    early.state.cities = 2;
    expect(rewardEarly.step(early.game, early.agentId, null)).toBeCloseTo(0, 10);
  });

  it("pays holding troops above 30 percent of peak mid game", () => {
    const weights = {
      land: 0, troop: 0, gold: 0, win: 0, death: 0, kill: 0,
      landEarly: 0, landLoss: 1, city: 0, factory: 0,
      troopHold: 0.3, survive: 0, rich: 0,
    };
    const stub = stubGame({ opponents: {}, troops: 10000 });
    const reward = new Reward({ weights });
    reward.init(stub.game, stub.agentId);
    stub.state.ticks = 1000;
    stub.state.troops = 2000;
    // Floor is 3000; hold falls 1 -> 2/3.
    expect(reward.step(stub.game, stub.agentId, null)).toBeCloseTo(
      0.3 * (2000 / 3000 - 1), 10,
    );
  });

  it("pays the survival trickle only late while alive", () => {    const weights = {
      land: 0, troop: 0, gold: 0, win: 0, death: 0, kill: 0,
      landEarly: 0, landLoss: 1, city: 0, factory: 0,
      troopHold: 0, survive: 0.0002, rich: 0,
    };
    const late = stubGame({ opponents: {} });
    const reward = new Reward({ weights });
    reward.init(late.game, late.agentId);
    late.state.ticks = 5000;
    expect(reward.step(late.game, late.agentId, null)).toBeCloseTo(0.0002, 10);

    const early = stubGame({ opponents: {} });
    const rewardEarly = new Reward({ weights });
    rewardEarly.init(early.game, early.agentId);
    expect(rewardEarly.step(early.game, early.agentId, null)).toBeCloseTo(0, 10);
  });

  it("pays approaching MIRV affordability late, capped at full", () => {
    const weights = {
      land: 0, troop: 0, gold: 0, win: 0, death: 0, kill: 0,
      landEarly: 0, landLoss: 1, city: 0, factory: 0,
      troopHold: 0, survive: 0, rich: 0.5,
    };
    const stub = stubGame({ opponents: {}, gold: 0n, mirvCost: 10_000_000 });
    const reward = new Reward({ weights });
    reward.init(stub.game, stub.agentId);
    stub.state.ticks = 5000;
    stub.state.gold = 5_000_000n;
    expect(reward.step(stub.game, stub.agentId, null)).toBeCloseTo(0.25, 10);
    stub.state.gold = 50_000_000n;
    expect(reward.step(stub.game, stub.agentId, null)).toBeCloseTo(0.25, 10);
  });
});
