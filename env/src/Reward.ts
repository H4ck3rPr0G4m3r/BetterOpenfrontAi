import type { Game, Player, PlayerID, TerraNullius, Unit } from "../../OpenFrontIO/src/core/game/Game";
import { Structures, UnitType } from "../../OpenFrontIO/src/core/game/Game";
import type { TileRef } from "../../OpenFrontIO/src/core/game/GameMap";
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
/** A new defense post this close (tiles) to another own post is crowded. */
export const POST_SPACING = 20;
/** A boat launch pays only with at least this share of max troops aboard. */
export const BOAT_MIN_SHARE = 0.1;
/** Nuke damage counts hostile structure levels plus hostile tiles / NUKE_TILE_REF. */
export const NUKE_TILE_REF = 1000;
/** Bound on the nuke term per step: a MIRV's warheads can wipe a whole
 *  empire, and one step must not outweigh winning the game. */
export const NUKE_REWARD_CAP = 3;

/** Paid by the structures potential. Defense posts are left out: they are
 *  priced per build (defensePost, crowdedPost) instead. */
const STRUCTURE_TYPES = [
  UnitType.City,
  UnitType.Port,
  UnitType.Factory,
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
  /** Defense posts this seat started this step. */
  postsBuilt: number;
  /** Of those, posts within POST_SPACING tiles of another own post. */
  postsCrowded: number;
  /** Paid boat launches this step (real force, a slot left free). */
  boatsLaunched: number;
  /** Own boats that landed and took their landing tile this step. */
  boatsLanded: number;
  /** Hostile damage of own nukes that detonated this step (nukeDamage units). */
  nukeDamage: number;
  /** Own and allied damage of those nukes, same units (>= 0). */
  nukeSelfDamage: number;
}

export const NO_EVENTS: SeatEvents = {
  kills: 0,
  betrayals: 0,
  diplomacy: 0,
  advantageTiles: 0,
  repelled: 0,
  earned: 0,
  postsBuilt: 0,
  postsCrowded: 0,
  boatsLaunched: 0,
  boatsLanded: 0,
  nukeDamage: 0,
  nukeSelfDamage: 0,
};

/**
 * Per-seat reward, v4.
 *
 *   r_t = Φ(s_t) − Φ(s_{t−1}) + kill · kills + survive · alive + terminal
 *         + lowTroops · max(0, 1 − troopShare / lowTroopsBelow)
 *         + betrayal · betrayals + diplomacy · diplomatic intents
 *         + advantage · land · advantageTiles / (LAND_REF + tiles) + defend · repelled
 *         + defensePost · posts built + crowdedPost · posts built near another post
 *         + boatLaunch · paid boat launches + boatLanding · boats landed
 *         + min(nuke · hostile damage of own nukes that detonated, NUKE_REWARD_CAP)
 *         + max(nukeSelf · own and allied damage of them, −NUKE_REWARD_CAP)
 *   Φ(s) = land · log(1 + tiles / LAND_REF) + structures · log(1 + structure levels)
 *          + cities · log(1 + city levels) + popCap · log(1 + max troops / POP_REF)
 *          + income · log(1 + income per minute / INCOME_REF)
 *          + rail · log(1 + own cities and ports on the rail network)
 *
 * v3 (from watching v2 replays): seats expanded themselves to zero troops,
 * farmed alliance request/break cycles, spammed embargo toggles, and never
 * picked fights they could lose. The extra terms price exactly those
 * behaviours. The advantage term is the land potential's own per-tile slope
 * scaled up, so a tile taken from a tribe or a weaker player is worth
 * (1 + advantage) tiles: clear the easy land, attack from strength.
 *
 * v4 (from v3 replays): the policy played Territorial, land only. Cities,
 * the population cap and the income rate are now potentials too. Income is
 * a rate, not a sum: paying per gold earned let a city builder that died
 * at minute 10 out-score every survivor (DeathProbe, +17.5).
 *
 * v7: defense posts were cheap structure-potential points, so policies
 * stacked them side by side. They now pay nothing by existing, cost a
 * little each, and cost more when built next to another own post.
 *
 * v8: boats. Trained policies stopped boating (10% of actions early, 1%
 * by 300k steps): a boat costs troops at once and its land arrives later.
 * Launches with a real force pay a little while a slot stays free, and a
 * landing that takes its tile pays more.
 *
 * v9: nukes. A bomb costs its gold at once and pays here only if it lands
 * (not shot down): hostile structure levels in the blast plus hostile land,
 * own and allied losses are a separate penalty (nukeSelf) that hostile
 * damage in the same blast cannot offset.
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
    r += this.add("defensePost", w.defensePost * events.postsBuilt);
    r += this.add("crowdedPost", w.crowdedPost * events.postsCrowded);
    r += this.add("boatLaunch", w.boatLaunch * events.boatsLaunched);
    r += this.add("boatLanding", w.boatLanding * events.boatsLanded);
    r += this.add("nuke", Math.min(NUKE_REWARD_CAP, w.nuke * events.nukeDamage));
    r += this.add("nukeSelf", Math.max(-NUKE_REWARD_CAP, w.nukeSelf * events.nukeSelfDamage));
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
    const rail = game === undefined || w.rail === 0 ? 0 : w.rail * Math.log1p(railStations(game, player));
    return w.land * land + w.structures * Math.log1p(structureLevels(player)) + w.cities * Math.log1p(cityLevels(player)) + cap + income + rail;
  }
}

/** Own cities and ports that are train stations, i.e. connected by rail. */
export function railStations(game: Game, player: Player): number {
  const stations = game.railNetwork().stationManager();
  let n = 0;
  for (const unit of player.units([UnitType.City, UnitType.Port])) {
    if (stations.findStation(unit) !== null) n++;
  }
  return n;
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

/**
 * Defense posts started since the last call, and how many of them sit within
 * POST_SPACING tiles of another of the player's posts. `known` holds the post
 * IDs seen last time and is updated in place.
 */
export function newDefensePosts(game: Game, player: Player, known: Set<number>): { built: number; crowded: number } {
  const posts = player.isAlive() ? player.units(UnitType.DefensePost) : [];
  const limit = POST_SPACING * POST_SPACING;
  let built = 0;
  let crowded = 0;
  for (const post of posts) {
    if (known.has(post.id())) continue;
    built++;
    if (posts.some((other) => other !== post && game.euclideanDistSquared(post.tile(), other.tile()) <= limit)) crowded++;
  }
  known.clear();
  for (const post of posts) known.add(post.id());
  return { built, crowded };
}

/** What a boat looked like at the previous step. */
export interface BoatTrack {
  target: TileRef | undefined;
  retreating: boolean;
  /** The landing tile was already ours (a landing there takes nothing). */
  targetOwned: boolean;
  /** Launched with at least BOAT_MIN_SHARE of max troops. */
  real: boolean;
}

/**
 * Boat launches worth paying and landings since the last call. Only boats
 * launched with at least BOAT_MIN_SHARE of max troops count, so tiny
 * throwaway boats earn nothing; the launch that fills the last boat slot
 * is not paid either. A boat that is gone counts as landed when it was not
 * retreating and its landing tile, not ours before, now is: the engine
 * conquers that tile as the boat unloads. Sunk boats leave it with someone
 * else. `known` is updated in place.
 */
export function boatEvents(game: Game, player: Player, known: Map<number, BoatTrack>): { launched: number; landed: number } {
  const alive = player.isAlive();
  const boats = alive ? player.units(UnitType.TransportShip) : [];
  const ours = (tile: TileRef | undefined): boolean => tile !== undefined && game.owner(tile).id() === player.id();
  const minTroops = BOAT_MIN_SHARE * game.config().maxTroops(player);
  let real = 0;
  for (const boat of boats) {
    if (!known.has(boat.id()) && boat.troops() >= minTroops) real++;
  }
  const launched = boats.length >= game.config().boatMaxNumber() ? Math.max(0, real - 1) : real;
  let landed = 0;
  if (alive) {
    const now = new Set(boats.map((b) => b.id()));
    for (const [id, track] of known) {
      if (!now.has(id) && track.real && !track.retreating && !track.targetOwned && ours(track.target)) landed++;
    }
  }
  const before = new Map(known);
  known.clear();
  for (const boat of boats) {
    const target = boat.targetTile();
    const real = before.get(boat.id())?.real ?? boat.troops() >= minTroops;
    known.set(boat.id(), { target, retreating: boat.transportShipState().isRetreating, targetOwned: ours(target), real });
  }
  return { launched, landed };
}

/** Blast damage split by who suffers it (nukeDamage units, both >= 0). */
export interface NukeDamage {
  hostile: number;
  friendly: number;
}

/** One of our nukes in flight, with the damage it was worth when first seen. */
export interface NukeTrack {
  unit: Unit;
  damage: NukeDamage;
}

const PAID_NUKES = [UnitType.AtomBomb, UnitType.HydrogenBomb, UnitType.MIRVWarhead] as const;

/**
 * Damage of own nukes that detonated since the last call. A nuke that is
 * gone counts only when it reached its target: intercepted ones are deleted
 * without that flag. A MIRV pays through its warheads. The damage is valued
 * when a nuke is first seen, on the land as it stood at launch (flights are
 * short). `known` is updated in place.
 */
export function nukeEvents(game: Game, player: Player, known: Map<number, NukeTrack>): NukeDamage {
  const nukes = player.isAlive() ? player.units(...PAID_NUKES) : [];
  const now = new Set(nukes.map((n) => n.id()));
  const damage = { hostile: 0, friendly: 0 };
  for (const [id, track] of known) {
    if (now.has(id)) continue;
    if (track.unit.reachedTarget()) {
      damage.hostile += track.damage.hostile;
      damage.friendly += track.damage.friendly;
    }
    known.delete(id);
  }
  for (const nuke of nukes) {
    const target = nuke.targetTile();
    if (known.has(nuke.id()) || target === undefined) continue;
    known.set(nuke.id(), { unit: nuke, damage: nukeDamage(game, player, target, nuke.type()) });
  }
  return damage;
}

/**
 * What a blast at `tile` would destroy, from `player`'s point of view:
 * structure levels + tiles / NUKE_TILE_REF, hostile and own-or-allied
 * counted apart. Inner-radius
 * tiles count fully, outer-ring tiles half (the engine destroys those with
 * probability 1/2); every structure inside the outer radius is destroyed.
 */
export function nukeDamage(game: Game, player: Player, tile: TileRef, type: UnitType): NukeDamage {
  const { inner, outer } = game.config().nukeMagnitudes(type);
  const map = game.map();
  const w = map.width();
  const h = map.height();
  const x0 = tile % w;
  const y0 = Math.floor(tile / w);
  const inner2 = inner * inner;
  const outer2 = outer * outer;
  const friendlyByOwner = new Map<number, boolean>();
  const friendly = (smallID: number): boolean => {
    let f = friendlyByOwner.get(smallID);
    if (f === undefined) {
      // A dangling ownerID (no registered player, see TargetSlots) has no
      // player behind it; count its tiles as hostile.
      const p = game.playerBySmallID(smallID) as Player | TerraNullius | undefined;
      f = smallID === player.smallID() || (p !== undefined && p.isPlayer() && player.isFriendly(p as Player));
      friendlyByOwner.set(smallID, f);
    }
    return f;
  };
  let hostileTiles = 0;
  let friendlyTiles = 0;
  for (let y = Math.max(0, y0 - outer); y <= Math.min(h - 1, y0 + outer); y++) {
    for (let x = Math.max(0, x0 - outer); x <= Math.min(w - 1, x0 + outer); x++) {
      const d2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
      if (d2 > outer2) continue;
      const t = y * w + x;
      if (!map.isLand(t)) continue;
      const owner = map.ownerID(t);
      if (owner === 0) continue;
      const weight = d2 <= inner2 ? 1 : 0.5;
      if (friendly(owner)) friendlyTiles += weight;
      else hostileTiles += weight;
    }
  }
  let hostileLevels = 0;
  let friendlyLevels = 0;
  for (const { unit, distSquared } of game.nearbyUnits(tile, outer, Structures.types)) {
    if (distSquared >= outer2) continue;
    if (friendly(unit.owner().smallID())) friendlyLevels += unit.level();
    else hostileLevels += unit.level();
  }
  return {
    hostile: hostileLevels + hostileTiles / NUKE_TILE_REF,
    friendly: friendlyLevels + friendlyTiles / NUKE_TILE_REF,
  };
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
