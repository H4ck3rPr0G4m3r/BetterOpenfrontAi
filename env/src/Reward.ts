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

/** What a seat did or suffered during one step, besides its state change. */
export interface SeatEvents {
  /** Eliminations this seat caused (engine ConquestEvent attribution). */
  kills: number;
  /** Alliances this seat broke. */
  betrayals: number;
  /** Diplomatic intents this seat sent (request, break, embargo, target). */
  diplomacy: number;
  /** Tiles taken from players with more troops than this seat. */
  strongTiles: number;
  /** Attacks repelled from players with more troops than this seat. */
  repelled: number;
}

export const NO_EVENTS: SeatEvents = { kills: 0, betrayals: 0, diplomacy: 0, strongTiles: 0, repelled: 0 };

/**
 * Per-seat reward, v3.
 *
 *   r_t = Φ(s_t) − Φ(s_{t−1}) + kill · kills + survive · alive + terminal
 *         + lowTroops · max(0, 1 − troopShare / lowTroopsBelow)
 *         + betrayal · betrayals + diplomacy · diplomatic intents
 *         + underdog · land · strongTiles / (LAND_REF + tiles) + defend · repelled
 *   Φ(s) = land · log(1 + tiles / LAND_REF) + structures · log(1 + structure levels)
 *
 * v3 (from watching v2 replays): seats expanded themselves to zero troops,
 * farmed alliance request/break cycles, spammed embargo toggles, and never
 * picked fights they could lose. The extra terms price exactly those
 * behaviours. The underdog term is the land potential's own per-tile slope
 * scaled up, so a tile taken from a stronger player is worth (1 + underdog)
 * tiles.
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
  /** Episode sums per term, for probes and diagnostics. */
  readonly terms: Record<string, number> = {};
  private prevPotential = 0;

  constructor(weights: Partial<RewardWeights> = {}) {
    this.weights = { ...DEFAULT_REWARD_WEIGHTS, ...weights };
  }

  /** Seed the potential from the post-spawn state. */
  init(game: Game, playerID: PlayerID): void {
    this.prevPotential = this.potential(game.player(playerID));
    for (const k of Object.keys(this.terms)) delete this.terms[k];
  }

  private add(term: string, value: number): number {
    if (value !== 0) this.terms[term] = (this.terms[term] ?? 0) + value;
    return value;
  }

  step(game: Game, playerID: PlayerID, terminal: TerminalReason, events: SeatEvents = NO_EVENTS): number {
    const w = this.weights;
    const player = game.player(playerID);
    const potential = this.potential(player);
    let r = this.add("potential", potential - this.prevPotential);
    this.prevPotential = potential;
    r += this.add("kill", w.kill * events.kills);
    r += this.add("betrayal", w.betrayal * events.betrayals);
    r += this.add("diplomacy", w.diplomacy * events.diplomacy);
    if (player.isAlive()) {
      r += this.add("survive", w.survive);
      r += this.add("lowTroops", w.lowTroops * lowTroopDeficit(game, player, w.lowTroopsBelow));
      r += this.add("underdog", (w.underdog * w.land * events.strongTiles) / (LAND_REF + player.numTilesOwned()));
      r += this.add("defend", w.defend * events.repelled);
    }
    switch (terminal) {
      case "win":
        r += this.add("win", w.win);
        break;
      case "death":
        r += this.add("death", w.death);
        break;
      case "tick_cap":
        r += this.add("placement", w.placement * placementScore(game, player));
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

/** 0 at or above `below` of max troops, rising linearly to 1 at zero troops. */
export function lowTroopDeficit(game: Game, player: Player, below: number): number {
  if (below <= 0) return 0;
  const share = player.troops() / Math.max(1, game.config().maxTroops(player));
  return Math.max(0, 1 - share / below);
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
