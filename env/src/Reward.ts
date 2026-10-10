import type { Game, Player, PlayerID } from "../../OpenFrontIO/src/core/game/Game";
import { UnitType } from "../../OpenFrontIO/src/core/game/Game";
import { DEFAULT_REWARD_WEIGHTS, RewardWeights } from "./BehaviorConfig";
import type { TerminalReason } from "./types";

/** Tile count at which the land potential reaches log(2). Early growth from a
 *  ~50-tile spawn is priced steeply; doubling a large empire still pays. */
export const LAND_REF = 100;

const STRUCTURE_TYPES = [
  UnitType.City,
  UnitType.Port,
  UnitType.Factory,
  UnitType.DefensePost,
  UnitType.SAMLauncher,
  UnitType.MissileSilo,
];

/**
 * Per-seat reward, v2.
 *
 *   r_t = Φ(s_t) − Φ(s_{t−1}) + kill · (eliminations this seat caused)
 *         + survive · alive + terminal
 *   Φ(s) = land · log(1 + tiles / LAND_REF) + structures · log(1 + structure levels)
 *
 * Why this shape (2026-10 diagnosis): the v1 reward paid `kill` for every
 * elimination anywhere on the map (~70 per World game, nearly all bots dying
 * to nations), which made a passive agent that died at step 171 score +16
 * while land — the thing that keeps you alive — moved return by ~0.08. The
 * policy was trained on noise. Here every term is either a potential (so
 * dying gives back everything the seat built, and idling mints nothing) or
 * an event the seat caused, attributed by the engine's ConquestEvent.
 *
 * Terminals: win → `win`; death → `death` (Φ also collapses to ~0, so the
 * real price of dying is everything built plus `death`); tick cap →
 * `placement` × land rank among living players (1 = largest). The tick cap
 * is therefore a true terminal of a time-aware MDP (the observation carries
 * the time fraction), so trainers never need to bootstrap across it.
 */
export class Reward {
  readonly weights: RewardWeights;
  private prevPotential = 0;

  constructor(weights: Partial<RewardWeights> = {}) {
    this.weights = { ...DEFAULT_REWARD_WEIGHTS, ...weights };
  }

  /** Seed the potential from the post-spawn state. */
  init(game: Game, playerID: PlayerID): void {
    this.prevPotential = this.potential(game.player(playerID));
  }

  /**
   * @param kills eliminations this seat caused since the last step
   */
  step(game: Game, playerID: PlayerID, terminal: TerminalReason, kills: number): number {
    const player = game.player(playerID);
    const potential = this.potential(player);
    let r = potential - this.prevPotential;
    this.prevPotential = potential;
    r += this.weights.kill * kills;
    if (player.isAlive()) r += this.weights.survive;
    switch (terminal) {
      case "win":
        r += this.weights.win;
        break;
      case "death":
        r += this.weights.death;
        break;
      case "tick_cap":
        r += this.weights.placement * placementScore(game, player);
        break;
      default:
        break;
    }
    return r;
  }

  potential(player: Player): number {
    if (!player.isAlive()) return 0;
    const land = Math.log1p(player.numTilesOwned() / LAND_REF);
    return this.weights.land * land + this.weights.structures * Math.log1p(structureLevels(player));
  }
}

export function structureLevels(player: Player): number {
  let levels = 0;
  for (const unit of player.units(STRUCTURE_TYPES)) {
    if (!unit.isUnderConstruction()) levels += unit.level();
  }
  return levels;
}

/** 1 for the largest living player by land, 0 for the smallest (or dead). */
export function placementScore(game: Game, player: Player): number {
  if (!player.isAlive()) return 0;
  const alive = game.players().filter((p) => p.isAlive());
  if (alive.length <= 1) return 1;
  const tiles = player.numTilesOwned();
  const larger = alive.filter((p) => p.numTilesOwned() > tiles).length;
  return 1 - larger / (alive.length - 1);
}
