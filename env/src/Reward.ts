import type { Game, Player, PlayerID } from "../../OpenFrontIO/src/core/game/Game";
import { UnitType } from "../../OpenFrontIO/src/core/game/Game";
import { GOLD_INDEX_WORK } from "../../OpenFrontIO/src/core/StatsSchemas";
import { DEFAULT_REWARD_WEIGHTS, RewardWeights } from "./BehaviorConfig";
import type { TerminalReason } from "./types";

/** Tile count at which the land potential reaches log(2). Early growth from a
 *  ~50-tile spawn is priced steeply; doubling a large empire still pays. */
export const LAND_REF = 100;

/** Max troops at which the population-cap potential reaches log(2). A fresh
 *  spawn is ~120k; each city level adds 250k. */
export const POP_REF = 100_000;

/** Income per minute at which the income potential reaches log(2). A city
 *  costs 125k; a working train network earns several hundred k a minute. */
export const INCOME_REF = 100_000;
/** Steps per minute at 10 ticks per step, and the smoothing window. */
const STEPS_PER_MINUTE = 60;

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
  /** Tiles taken from players with fewer troops than this seat. */
  advantageTiles: number;
  /** Real attacks this seat repelled. */
  repelled: number;
  /** Gold earned this step beyond the flat worker income. */
  earned: number;
}

export const NO_EVENTS: SeatEvents = {
  kills: 0,
  betrayals: 0,
  diplomacy: 0,
  advantageTiles: 0,
  repelled: 0,
  earned: 0,
};

/**
 * Per-seat reward, v4.
 *
 *   r_t = Φ(s_t) − Φ(s_{t−1}) + kill · kills + survive · alive + terminal
 *         + lowTroops · max(0, 1 − troopShare / lowTroopsBelow)
 *         + betrayal · betrayals + diplomacy · diplomatic intents
 *         + advantage · land · advantageTiles / (LAND_REF + tiles) + defend · repelled
 *   Φ(s) = land · log(1 + tiles / LAND_REF) + structures · log(1 + structure levels)
 *          + cities · log(1 + city levels) + popCap · log(1 + max troops / POP_REF)
 *          + income · log(1 + income per minute / INCOME_REF)
 *
 * v3 (from watching v2 replays): seats expanded themselves to zero troops,
 * farmed alliance request/break cycles, spammed embargo toggles, and never
 * picked fights they could lose. The extra terms price exactly those
 * behaviours. The advantage term is the land potential's own per-tile slope
 * scaled up, so a tile taken from a weaker player is worth (1 + advantage)
 * tiles: attack from strength.
 *
 * v4 (from v3 replays): the policy played Territorial, land only. Cities,
 * the population cap and the income rate are now potentials too. Income is
 * a rate, not a sum: paying per gold earned let a city builder that died
 * at minute 10 out-score every survivor (DeathProbe, +17.5).
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
  /** Smoothed gold earned per step. */
  private incomeRate = 0;

  constructor(weights: Partial<RewardWeights> = {}) {
    this.weights = { ...DEFAULT_REWARD_WEIGHTS, ...weights };
  }

  /** Seed the potential from the post-spawn state. */
  init(game: Game, playerID: PlayerID): void {
    this.incomeRate = 0;
    this.prevPotential = this.potential(game.player(playerID), game);
    for (const k of Object.keys(this.terms)) delete this.terms[k];
  }

  private add(term: string, value: number): number {
    if (value !== 0) this.terms[term] = (this.terms[term] ?? 0) + value;
    return value;
  }

  step(game: Game, playerID: PlayerID, terminal: TerminalReason, events: SeatEvents = NO_EVENTS): number {
    const w = this.weights;
    const player = game.player(playerID);
    this.incomeRate += (events.earned - this.incomeRate) / STEPS_PER_MINUTE;
    const potential = this.potential(player, game);
    let r = this.add("potential", potential - this.prevPotential);
    this.prevPotential = potential;
    r += this.add("kill", w.kill * events.kills);
    r += this.add("betrayal", w.betrayal * events.betrayals);
    r += this.add("diplomacy", w.diplomacy * events.diplomacy);
    if (player.isAlive()) {
      r += this.add("survive", w.survive);
      r += this.add("lowTroops", w.lowTroops * lowTroopDeficit(game, player, w.lowTroopsBelow));
      r += this.add("advantage", (w.advantage * w.land * events.advantageTiles) / (LAND_REF + player.numTilesOwned()));
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

  potential(player: Player, game?: Game): number {
    if (!player.isAlive()) return 0;
    const land = Math.log1p(player.numTilesOwned() / LAND_REF);
    const w = this.weights;
    const cap = game === undefined || w.popCap === 0 ? 0 : w.popCap * Math.log1p(game.config().maxTroops(player) / POP_REF);
    const income = w.income * Math.log1p((this.incomeRate * STEPS_PER_MINUTE) / INCOME_REF);
    return w.land * land + w.structures * Math.log1p(structureLevels(player)) + w.cities * Math.log1p(cityLevels(player)) + cap + income;
  }
}

export function cityLevels(player: Player): number {
  let levels = 0;
  for (const unit of player.units(UnitType.City)) {
    if (!unit.isUnderConstruction()) levels += unit.level();
  }
  return levels;
}

/** Gold earned so far beyond the flat worker income (trade, trains, conquest). */
export function earnedGold(game: Game, player: Player): number {
  const gold = game.stats().getPlayerStats(player)?.gold;
  if (!gold) return 0;
  let sum = 0;
  gold.forEach((v, i) => {
    if (i !== GOLD_INDEX_WORK) sum += Number(v);
  });
  return sum;
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
