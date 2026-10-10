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

const MAX_BORDER_SITES = 10;
const MAX_INTERIOR_SITES = 4;
const TYPES_PER_SITE = 3;

/**
 * Bounded build-site candidates: spread-sampled border tiles plus a few
 * interior tiles, each offering TYPES_PER_SITE rotated unit types, capped at
 * MAX_BUILD_SITES. Dense per-cell build masking was measured at ~100x the
 * step budget, hence a short list. Deterministic for a given state, and the
 * same list feeds the observation's site table and the codec's site head, so
 * site index i means the same (tile, type) on both sides. Disabled build
 * types are masked, never filtered, for the same reason.
 */
export function buildCandidates(game: Game, agent: Player): BuildCandidate[] {
  let entry = cache.get(game);
  if (entry === undefined || entry.tick !== game.ticks()) {
    entry = { tick: game.ticks(), byPlayer: new Map() };
    cache.set(game, entry);
  }
  const hit = entry.byPlayer.get(agent.smallID());
  if (hit !== undefined) return hit;

  const border = [...agent.borderTiles()].sort((a, b) => a - b);
  const borderSites: TileRef[] = [];
  if (border.length > 0) {
    const stride = Math.max(1, Math.floor(border.length / MAX_BORDER_SITES));
    for (let i = 0; i < border.length && borderSites.length < MAX_BORDER_SITES; i += stride) {
      borderSites.push(border[i]);
    }
  }
  const interiorSites: TileRef[] = [];
  const owned = agent.numTilesOwned();
  if (owned > border.length) {
    const borderSet = new Set<TileRef>(border);
    const step = Math.max(1, Math.floor(owned / MAX_INTERIOR_SITES));
    let i = 0;
    for (const tile of agent.tiles()) {
      if (!borderSet.has(tile) && i++ % step === 0) {
        interiorSites.push(tile);
        if (interiorSites.length >= MAX_INTERIOR_SITES) break;
      }
    }
    interiorSites.sort((a, b) => a - b);
  }

  const out: BuildCandidate[] = [];
  const sites = [...borderSites.map((t) => [t, true] as const), ...interiorSites.map((t) => [t, false] as const)];
  for (let s = 0; s < sites.length && out.length < MAX_BUILD_SITES; s++) {
    for (let j = 0; j < TYPES_PER_SITE && out.length < MAX_BUILD_SITES; j++) {
      const typeIndex = (s + j) % BUILD_UNIT_TYPES.length;
      out.push({ tile: sites[s][0], typeIndex, type: BUILD_UNIT_TYPES[typeIndex], border: sites[s][1] });
    }
  }
  entry.byPlayer.set(agent.smallID(), out);
  return out;
}

/** Per-game, per-tick cache: encode() and masks() ask for the same list on
 *  the same tick for every seat, and the border scan is not free on big
 *  territories. A new tick drops the whole entry. */
const cache = new WeakMap<Game, { tick: number; byPlayer: Map<number, BuildCandidate[]> }>();
