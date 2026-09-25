import type { Game, PlayerID } from "../../OpenFrontIO/src/core/game/Game";
import { UnitType } from "../../OpenFrontIO/src/core/game/Game";
import {
  DEFAULT_REWARD_WEIGHTS,
  RewardWeights,
} from "./BehaviorConfig";

/** Why an episode ended, for the terminal term of the reward (§5.5). */
export type TerminalReason = "win" | "death" | "tick_cap" | "sim_error" | null;

/**
 * Fixed normalization constants for the reward, not per-step maxima over
 * living players — a moving denominator would hand the agent negative
 * reward whenever a rival's stats spiked, which isn't a signal about its
 * own play (spec §5.5).
 *
 * Starting orders of magnitude, not tuned values: a spawn nets ~25,000
 * troops (§2.7), and mid/late-game structure costs run 1M-5M gold
 * (Factory/HydrogenBomb) up to 25M+ (MIRV). Reward *design* is phase 2;
 * phase 1 owes one unambiguous, swappable default.
 */
export const TROOP_SCALE = 100_000;
export const GOLD_SCALE = 10_000_000;

/**
 * Phase boundaries in ticks since spawn end. At the live ~10 ticks/sec
 * cadence that is ~1.5min of early expansion and ~5min to mid game, with
 * the 6000-tick cap covering ~10min. init() runs post-spawn, so the clock
 * starts there, not at episode start.
 */
export const EARLY_END_TICK = 900;
export const MID_END_TICK = 3000;

/** Troop-hold floor: fraction of peak troops the mid game pays to keep. */
export const TROOP_HOLD_FLOOR = 0.3;

export interface RewardConfig {
  troopScale?: number;
  goldScale?: number;
  /** Partial shaping override; missing fields keep phase-1 defaults. */
  weights?: Partial<RewardWeights>;
  /** Phase boundary overrides, in post-spawn ticks. */
  earlyEndTick?: number;
  midEndTick?: number;
  /** Scale on negative troop deltas only (0..1, default 1.0 = symmetric).
   *  Spending troops on an attack reads as a troop loss under the plain
   *  delta, so the default reward punishes fighting. 0.0 keeps troop gains
   *  while ignoring losses. */
  troopLossScale?: number;
}

/**
 * Maps a state transition to a scalar. Selected by `EnvConfig` so it can be
 * swapped without touching `GameEnv`.
 *
 * Three phases shape different play at different times. Early pays land
 * (harder gains, harsher losses); mid pays building and holding troops;
 * late pays staying alive and affording the endgame. Everything is a delta
 * except the tiny late survival trickle, so longer episodes don't mint
 * return by idling — except late survival, which is capped by the tick cap
 * itself and must stay below the win bonus.
 */
export class Reward {
  private readonly troopScale: number;
  private readonly goldScale: number;
  private readonly weights: RewardWeights;
  private readonly earlyEndTick: number;
  private readonly midEndTick: number;
  private readonly troopLossScale: number;
  private prevLandFraction = 0;
  private prevTroopFraction = 0;
  private prevGoldFraction = 0;
  // Snapshotted once in init(), not re-read from game.map() every step().
  // GameMap.numLandTiles() mutates (nukes and other terrain edits change
  // land tiles to non-land, see GameMap.ts's numLandTiles_--), so reading
  // it fresh each step would silently reintroduce the moving-denominator
  // problem this reward was specifically designed to avoid (§5.5) — just
  // driven by terrain destruction instead of rival stats.
  private totalLandTiles = 0;
  private prevAliveIds = new Set<PlayerID>();
  private startTick = 0;
  private prevCities = 0;
  private prevFactories = 0;
  private maxTroops = 0;
  private prevHold = 0;
  private prevRich = 0;

  constructor(config: RewardConfig = {}) {
    this.troopScale = config.troopScale ?? TROOP_SCALE;
    this.goldScale = config.goldScale ?? GOLD_SCALE;
    this.weights = { ...DEFAULT_REWARD_WEIGHTS, ...config.weights };
    // An old override that prices land without naming landEarly means the
    // early price too: without this, setting land alone would silently stop
    // applying during the early phase, breaking every pre-phase config.
    if (config.weights?.landEarly === undefined && config.weights?.land !== undefined) {
      this.weights.landEarly = config.weights.land;
    }
    this.earlyEndTick = config.earlyEndTick ?? EARLY_END_TICK;
    this.midEndTick = config.midEndTick ?? MID_END_TICK;
    const troopLossScale = config.troopLossScale ?? 1.0;
    if (!Number.isFinite(troopLossScale) || troopLossScale < 0 || troopLossScale > 1) {
      throw new Error(`troopLossScale must be 0..1, got ${config.troopLossScale}`);
    }
    this.troopLossScale = troopLossScale;
  }

  /** Call once after `reset()`, before the first `step()`, to seed the
   *  deltas against the post-spawn state rather than against zero. */
  init(game: Game, agentPlayerID: PlayerID): void {
    this.totalLandTiles = game.map().numLandTiles();
    const agent = game.player(agentPlayerID);
    this.prevLandFraction = this.landFraction(agent.numTilesOwned());
    this.prevTroopFraction = agent.troops() / this.troopScale;
    this.prevGoldFraction = Number(agent.gold()) / this.goldScale;
    this.prevAliveIds = this.liveOpponentIds(game, agentPlayerID);
    this.startTick = game.ticks();
    this.prevCities = agent.unitsOwned(UnitType.City);
    this.prevFactories = agent.unitsOwned(UnitType.Factory);
    this.maxTroops = agent.troops();
    this.prevHold = this.holdFraction(agent.troops());
    this.prevRich = this.richFraction(game, agentPlayerID);
  }

  step(game: Game, agentPlayerID: PlayerID, terminal: TerminalReason): number {
    const agent = game.player(agentPlayerID);
    const phase = this.phase(game);

    const landFraction = this.landFraction(agent.numTilesOwned());
    const troopFraction = agent.troops() / this.troopScale;
    const goldFraction = Number(agent.gold()) / this.goldScale;

    const landDelta = landFraction - this.prevLandFraction;
    // Early game has its own land price; early losses additionally scale
    // by landLoss (1.0 keeps gains and losses symmetric, as before).
    const landCoef =
      phase === "early"
        ? landDelta >= 0
          ? this.weights.landEarly
          : this.weights.land * this.weights.landLoss
        : this.weights.land;
    // Troop spending (attack launches) reads as a troop loss under the
    // plain delta, punishing the winning action. troopLossScale mutes only
    // the negative side; gains always count in full.
    const troopDelta = troopFraction - this.prevTroopFraction;
    const scaledTroopDelta =
      troopDelta >= 0 ? troopDelta : troopDelta * this.troopLossScale;
    let shaping =
      landCoef * landDelta +
      this.weights.troop * scaledTroopDelta +
      this.weights.gold * (goldFraction - this.prevGoldFraction);

    if (phase === "mid") {
      const cities = agent.unitsOwned(UnitType.City);
      const factories = agent.unitsOwned(UnitType.Factory);
      shaping += this.weights.city * (cities - this.prevCities);
      shaping += this.weights.factory * (factories - this.prevFactories);
      this.prevCities = cities;
      this.prevFactories = factories;

      if (agent.troops() > this.maxTroops) this.maxTroops = agent.troops();
      const hold = this.holdFraction(agent.troops());
      shaping += this.weights.troopHold * (hold - this.prevHold);
      this.prevHold = hold;
    }

    if (phase === "late") {
      if (agent.isAlive()) shaping += this.weights.survive;
      const rich = this.richFraction(game, agentPlayerID);
      shaping += this.weights.rich * (rich - this.prevRich);
      this.prevRich = rich;
    }

    this.prevLandFraction = landFraction;
    this.prevTroopFraction = troopFraction;
    this.prevGoldFraction = goldFraction;

    const kills = this.countKills(game, agentPlayerID);

    return shaping + this.weights.kill * kills + this.terminalBonus(terminal);
  }

  private phase(game: Game): "early" | "mid" | "late" {
    const elapsed = game.ticks() - this.startTick;
    if (elapsed < this.earlyEndTick) return "early";
    if (elapsed < this.midEndTick) return "mid";
    return "late";
  }

  private landFraction(ownedTiles: number): number {
    return this.totalLandTiles > 0 ? ownedTiles / this.totalLandTiles : 0;
  }

  private holdFraction(troops: number): number {
    const floor = TROOP_HOLD_FLOOR * this.maxTroops;
    if (floor <= 0) return 1;
    return Math.min(1, troops / floor);
  }

  private richFraction(game: Game, agentPlayerID: PlayerID): number {
    const cost = Number(
      game.unitInfo(UnitType.MIRV).cost(game, game.player(agentPlayerID)),
    );
    if (!(cost > 0)) return 0;
    return Math.min(1, Number(game.player(agentPlayerID).gold()) / cost);
  }

  private liveOpponentIds(game: Game, agentPlayerID: PlayerID): Set<PlayerID> {
    const ids = new Set<PlayerID>();
    for (const p of game.players()) {
      if (p.id() !== agentPlayerID && p.isAlive()) ids.add(p.id());
    }
    return ids;
  }

  private countKills(game: Game, agentPlayerID: PlayerID): number {
    const now = this.liveOpponentIds(game, agentPlayerID);
    let kills = 0;
    for (const id of this.prevAliveIds) {
      if (!now.has(id)) kills++;
    }
    this.prevAliveIds = now;
    return kills;
  }

  private terminalBonus(terminal: TerminalReason): number {
    switch (terminal) {
      case "win":
        return this.weights.win;
      case "death":
        return this.weights.death;
      case "tick_cap":
      case "sim_error":
      case null:
        return 0;
    }
  }
}
