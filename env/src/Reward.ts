import type { Game, PlayerID } from "../../OpenFrontIO/src/core/game/Game";
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

export interface RewardConfig {
  troopScale?: number;
  goldScale?: number;
  /** Partial shaping override; missing fields keep phase-1 defaults. */
  weights?: Partial<RewardWeights>;
}

/**
 * Maps a state transition to a scalar. Selected by `EnvConfig` so it can be
 * swapped without touching `GameEnv`.
 */
export class Reward {
  private readonly troopScale: number;
  private readonly goldScale: number;
  private readonly weights: RewardWeights;
  private prevLandFraction = 0;
  private prevTroopFraction = 0;
  private prevGoldFraction = 0;
  /** Alive non-agent player IDs at the previous snapshot. Compared across
   *  steps to count eliminations for the kill bonus. */
  private prevAliveIds = new Set<PlayerID>();
  // Snapshotted once in init(), not re-read from game.map() every step().
  // GameMap.numLandTiles() mutates (nukes and other terrain edits change
  // land tiles to non-land, see GameMap.ts's numLandTiles_--), so reading
  // it fresh each step would silently reintroduce the moving-denominator
  // problem this reward was specifically designed to avoid (§5.5) — just
  // driven by terrain destruction instead of rival stats.
  private totalLandTiles = 0;

  constructor(config: RewardConfig = {}) {
    this.troopScale = config.troopScale ?? TROOP_SCALE;
    this.goldScale = config.goldScale ?? GOLD_SCALE;
    this.weights = { ...DEFAULT_REWARD_WEIGHTS, ...config.weights };
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
  }

  step(game: Game, agentPlayerID: PlayerID, terminal: TerminalReason): number {
    const agent = game.player(agentPlayerID);

    const landFraction = this.landFraction(agent.numTilesOwned());
    const troopFraction = agent.troops() / this.troopScale;
    const goldFraction = Number(agent.gold()) / this.goldScale;

    const shaping =
      this.weights.land * (landFraction - this.prevLandFraction) +
      this.weights.troop * (troopFraction - this.prevTroopFraction) +
      this.weights.gold * (goldFraction - this.prevGoldFraction);

    this.prevLandFraction = landFraction;
    this.prevTroopFraction = troopFraction;
    this.prevGoldFraction = goldFraction;

    const kills = this.countKills(game, agentPlayerID);

    return shaping + this.weights.kill * kills + this.terminalBonus(terminal);
  }

  private landFraction(ownedTiles: number): number {
    return this.totalLandTiles > 0 ? ownedTiles / this.totalLandTiles : 0;
  }

  /**
   * IDs of living players excluding the agent. Built from game.players(),
   * the same call the encoder already makes every step, so this adds no
   * new cost class. An opponent that spawns and dies between two snapshots
   * is missed; with 10 ticks per step that window is small and the miss
   * only costs one bonus, never a wrong one.
   */
  private liveOpponentIds(game: Game, agentPlayerID: PlayerID): Set<PlayerID> {
    const ids = new Set<PlayerID>();
    for (const p of game.players()) {
      if (p.id() !== agentPlayerID && p.isAlive()) ids.add(p.id());
    }
    return ids;
  }

  /** Opponents alive at the previous snapshot but not now. Refreshes the
   *  snapshot, so each elimination is counted exactly once. */
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
