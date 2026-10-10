import type { Game, Player, PlayerBuildableUnitType } from "../../OpenFrontIO/src/core/game/Game";
import { UnitType } from "../../OpenFrontIO/src/core/game/Game";
import type { TileRef } from "../../OpenFrontIO/src/core/game/GameMap";
import { BUILD_TYPES, MAX_BUILD_SITES } from "./Layout";

/** Engine unit types in Layout.BUILD_TYPES order. */
export const BUILD_UNIT_TYPES: PlayerBuildableUnitType[] = [
  UnitType.City,
  UnitType.DefensePost,
  UnitType.Port,
  UnitType.SAMLauncher,
  UnitType.MissileSilo,
  UnitType.Factory,
  UnitType.Warship,
];
if (BUILD_UNIT_TYPES.some((t, i) => (t as string) !== BUILD_TYPES[i])) {
  throw new Error("BUILD_UNIT_TYPES drifted from Layout.BUILD_TYPES");
}

export interface BuildCandidate {
  tile: TileRef;
  /** Index into BUILD_TYPES / BUILD_UNIT_TYPES. */
  typeIndex: number;
  type: PlayerBuildableUnitType;
  border: boolean;
}

const MAX_FRONT_SITES = 8;
const MAX_SHORE_SITES = 3;
const MAX_INTERIOR_SITES = 4;
const FRONT_TYPES: PlayerBuildableUnitType[] = [UnitType.DefensePost];
const SHORE_TYPES: PlayerBuildableUnitType[] = [UnitType.Port];
/** The first WARSHIP_SITES shore tiles also offer a warship. */
const WARSHIP_SITES = 2;
const INTERIOR_TYPES: PlayerBuildableUnitType[] = [UnitType.City, UnitType.Factory, UnitType.SAMLauncher, UnitType.MissileSilo];

/** Up to `n` entries spread evenly over `list`. */
function spread<T>(list: T[], n: number): T[] {
  if (list.length <= n) return list.slice();
  const out: T[] = [];
  const stride = list.length / n;
  for (let i = 0; i < n; i++) out.push(list[Math.floor(i * stride)]);
  return out;
}

/**
 * Bounded build-site candidates, each tile offered only the structures that
 * belong there:
 *
 *   front    border land touching another player's land  -> Defense Post
 *   shore    border land touching water                  -> Port (+ Warship)
 *   interior owned land off the border                   -> City, Factory, SAM, Silo
 *
 * v2 rotated every type across every border tile, and since most of a
 * coastal empire's border is shoreline, defense posts went up facing the
 * sea. Dense per-cell build masking was measured at ~100x the step budget,
 * hence a short list. Deterministic for a given state, and the same list
 * feeds the observation's site table and the codec's site head, so site
 * index i means the same (tile, type) on both sides. Disabled build types
 * are masked, never filtered, for the same reason.
 */
export function buildCandidates(game: Game, agent: Player): BuildCandidate[] {
  let entry = cache.get(game);
  if (entry === undefined || entry.tick !== game.ticks()) {
    entry = { tick: game.ticks(), byPlayer: new Map() };
    cache.set(game, entry);
  }
  const hit = entry.byPlayer.get(agent.smallID());
  if (hit !== undefined) return hit;

  const map = game.map();
  const me = agent.smallID();
  const border = [...agent.borderTiles()].sort((a, b) => a - b);
  const front: TileRef[] = [];
  const shore: TileRef[] = [];
  const inland: TileRef[] = [];
  for (const tile of border) {
    let water = false;
    let enemy = false;
    for (const n of map.neighbors(tile)) {
      if (!map.isLand(n)) water = true;
      else {
        const owner = map.ownerID(n);
        if (owner !== me && owner !== 0) enemy = true;
      }
    }
    if (enemy) front.push(tile);
    if (water) shore.push(tile);
    if (!enemy && !water) inland.push(tile);
  }

  let interior: TileRef[] = [];
  if (agent.numTilesOwned() > border.length) {
    const borderSet = new Set<TileRef>(border);
    const step = Math.max(1, Math.floor(agent.numTilesOwned() / MAX_INTERIOR_SITES));
    let i = 0;
    for (const tile of agent.tiles()) {
      if (!borderSet.has(tile) && i++ % step === 0) {
        interior.push(tile);
        if (interior.length >= MAX_INTERIOR_SITES) break;
      }
    }
    interior.sort((a, b) => a - b);
  }
  // A territory with no interior yet builds behind its unowned-land border.
  if (interior.length === 0) interior = spread(inland, MAX_INTERIOR_SITES);

  const out: BuildCandidate[] = [];
  const add = (tile: TileRef, type: PlayerBuildableUnitType, border: boolean) => {
    if (out.length < MAX_BUILD_SITES) out.push({ tile, typeIndex: BUILD_UNIT_TYPES.indexOf(type), type, border });
  };
  for (const tile of spread(front, MAX_FRONT_SITES)) for (const type of FRONT_TYPES) add(tile, type, true);
  spread(shore, MAX_SHORE_SITES).forEach((tile, i) => {
    for (const type of SHORE_TYPES) add(tile, type, true);
    if (i < WARSHIP_SITES) add(tile, UnitType.Warship, true);
  });
  for (const tile of interior) for (const type of INTERIOR_TYPES) add(tile, type, false);
  entry.byPlayer.set(agent.smallID(), out);
  return out;
}

/** Per-game, per-tick cache: encode() and masks() ask for the same list on
 *  the same tick for every seat, and the border scan is not free on big
 *  territories. A new tick drops the whole entry. */
const cache = new WeakMap<Game, { tick: number; byPlayer: Map<number, BuildCandidate[]> }>();
