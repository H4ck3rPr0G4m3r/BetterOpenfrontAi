import { Structures } from "../../OpenFrontIO/src/core/game/Game";
import type { Game, Player, PlayerID } from "../../OpenFrontIO/src/core/game/Game";
import { GOLD_SCALE, TROOP_SCALE } from "./Reward";
import { assignTargetSlots } from "./TargetSlots";
import type { ObservationEncoder as ObservationEncoderInterface } from "./types";

export const RASTER_WIDTH = 128;
export const RASTER_HEIGHT = 64;
export const RASTER_CELLS = RASTER_WIDTH * RASTER_HEIGHT;

/** Raster plane order — spec §5.3's 8 planes, in the order they're packed. */
export enum Plane {
  Land = 0,
  SelfOwnership = 1,
  AllyOwnership = 2,
  EnemyOwnership = 3,
  UnownedOwnership = 4,
  SelfTroopDensity = 5,
  EnemyTroopDensity = 6,
  StructurePresence = 7,
}
export const NUM_PLANES = 8;
export const RASTER_SIZE = RASTER_CELLS * NUM_PLANES;

/** One player slot's fields, in the order they're packed. Slot 0 is always
 *  terra nullius; slots 1-16 are real players. */
export enum SlotField {
  Troops = 0,
  Gold = 1,
  Tiles = 2,
  Allied = 3,
  EmbargoedAgainstUs = 4,
  Traitor = 5,
  SharedBorder = 6,
  Valid = 7,
}
export const NUM_SLOT_FIELDS = 8;
export const NUM_SLOTS = 17; // slot 0 = terra nullius, 1-16 = real players
export const PLAYER_TABLE_SIZE = NUM_SLOTS * NUM_SLOT_FIELDS;

/** Global scalars packed after the player table. A real, justified set —
 *  not padded out to the spec's "~32" estimate with placeholder zeros.
 *  `size` is authoritative; this list can grow later without breaking
 *  anything that reads it by field name. */
export enum GlobalField {
  TickFraction = 0, // game.ticks() / TICK_CAP_REFERENCE
  InSpawnPhase = 1,
  AlivePlayerFraction = 2, // alive players / total players
  SelfRank = 3, // 0 = most land among alive players, 1 = least
  SelfTroopFraction = 4,
  SelfGoldFraction = 5,
  SelfTileFraction = 6,
  SelfAlive = 7,
  SelfTraitor = 8,
  SelfDoomsdayClock = 9,
  SelfAllyFraction = 10, // allies / (NUM_SLOTS - 1)
  SelfEmbargoedFraction = 11, // players embargoing us / (NUM_SLOTS - 1)
}
export const NUM_GLOBALS = 12;
// Reference tick count for TickFraction — the spec's default tick cap
// (§5.1). Not read from EnvConfig: the encoder must stay a pure function of
// (game, agentPlayerID) with no episode-config dependency, since it's also
// meant to run against a live-lobby shadow sim (§10.7) that has no EnvConfig.
const TICK_CAP_REFERENCE = 6000;

export const OBSERVATION_SIZE = RASTER_SIZE + PLAYER_TABLE_SIZE + NUM_GLOBALS;

/**
 * Letterbox mapping shared with the action codec's spatial head (spec §5.4):
 * one scale factor `s = max(width/128, height/64)` on both axes, map centred
 * in the 128x64 grid. A single source of truth so encoder cells and codec
 * cells always mean the same tiles.
 */
export function letterboxTransform(
  mapWidth: number,
  mapHeight: number,
): { scale: number; offsetX: number; offsetY: number } {
  const scale = Math.max(mapWidth / RASTER_WIDTH, mapHeight / RASTER_HEIGHT);
  const scaledW = mapWidth / scale;
  const scaledH = mapHeight / scale;
  return {
    scale,
    offsetX: Math.floor((RASTER_WIDTH - scaledW) / 2),
    offsetY: Math.floor((RASTER_HEIGHT - scaledH) / 2),
  };
}

/**
 * Static per-map terrain data, built once per GameMap and reused across
 * steps. Base terrain never changes within a game, but nukes convert land
 * to water through `setWater` / `updateTile`, which advance
 * `waterVersion()`. Geometry (letterbox, per-cell tile totals) is truly
 * static; land membership is cached against that version and rebuilt when
 * it moves. Only ownership and troops are dynamic and stay in the
 * per-step loop.
 *
 * Layout is axis-separated on purpose: `cxForX` (w ints) plus row bases for
 * y (h ints) replace the per-tile `Math.floor(x / scale)` float math with
 * two array reads. A full per-tile cell LUT would cost 4 bytes per tile
 * (~8 MB on World); the engine deliberately avoids such tables for the
 * same reason (`GameMap.ts` yToRef comment). The hot loop stays sequential
 * and prefetcher-friendly.
 */
interface StaticTerrain {
  w: number;
  h: number;
  scale: number;
  offsetX: number;
  offsetY: number;
  waterVersion: number;
  isLand: Uint8Array;
  cxForX: Int32Array;
  rowBaseForY: Int32Array;
  totalTilesInCell: Float64Array;
  landTilesInCell: Float64Array;
}
export class ObservationEncoder implements ObservationEncoderInterface {
  readonly size = OBSERVATION_SIZE;

  /**
   * Full recompute every call, but only of dynamic state — no dependency
   * on `drainPackedTileUpdates()`. That queue is destructive (whoever reads
   * it consumes it), so incremental maintenance would need per-episode
   * state this interface doesn't have. Static terrain comes from the
   * per-GameMap cache below; ownership is re-read from `tileStateBuffer()`
   * each call. See spec §11 for the record of that decision.
   */
  private staticByMap = new WeakMap<object, StaticTerrain>();
  // Scratch accumulators, reused across calls to avoid ~500 KB of
  // Float64Array garbage per encode. Fixed at RASTER_CELLS so sharing is
  // safe; every encode fills them from zero.
  private readonly scratchLandByCode = [
    new Float64Array(RASTER_CELLS),
    new Float64Array(RASTER_CELLS),
    new Float64Array(RASTER_CELLS),
  ];
  private readonly scratchTroopByCode = [
    new Float64Array(RASTER_CELLS),
    new Float64Array(RASTER_CELLS),
  ];
  private readonly scratchUnowned = new Float64Array(RASTER_CELLS);
  private readonly scratchStructures = new Float64Array(RASTER_CELLS);
  private readonly scratchSelfTiles: number[] = [];
  private readonly scratchBordering = new Set<number>();

  private getStatic(map: {
    width(): number;
    height(): number;
    isLand(ref: number): boolean;
    waterVersion(): number;
  }): StaticTerrain {
    const cached = this.staticByMap.get(map);
    if (cached !== undefined) {
      // Nukes flip land to water mid-episode (`setWater` / `updateTile`
      // advance waterVersion). Geometry never changes, so only the land
      // arrays need a rebuild when the version moves.
      if (map.waterVersion() === cached.waterVersion) return cached;
      this.refreshLand(map, cached);
      return cached;
    }
    const w = map.width();
    const h = map.height();
    const { scale, offsetX, offsetY } = letterboxTransform(w, h);
    const isLand = new Uint8Array(w * h);
    const cxForX = new Int32Array(w);
    for (let x = 0; x < w; x++) {
      cxForX[x] = Math.min(RASTER_WIDTH - 1, Math.max(0, offsetX + Math.floor(x / scale)));
    }
    const rowBaseForY = new Int32Array(h);
    for (let y = 0; y < h; y++) {
      const cy = Math.min(RASTER_HEIGHT - 1, Math.max(0, offsetY + Math.floor(y / scale)));
      rowBaseForY[y] = cy * RASTER_WIDTH;
    }
    const totalTilesInCell = new Float64Array(RASTER_CELLS);
    const landTilesInCell = new Float64Array(RASTER_CELLS);
    let t = 0;
    for (let y = 0; y < h; y++) {
      const rowBase = rowBaseForY[y];
      for (let x = 0; x < w; x++, t++) {
        const cell = rowBase + cxForX[x];
        totalTilesInCell[cell]++;
      }
    }
    const built: StaticTerrain = {
      w,
      h,
      scale,
      offsetX,
      offsetY,
      waterVersion: -1,
      isLand,
      cxForX,
      rowBaseForY,
      totalTilesInCell,
      landTilesInCell,
    };
    this.staticByMap.set(map, built);
    this.refreshLand(map, built);
    return built;
  }

  /** (Re)build land membership from the live map. Runs once per GameMap
   *  plus once per waterVersion bump (nukes). One `isLand` call per tile
   *  here, zero in the per-step loop. */
  private refreshLand(
    map: { isLand(ref: number): boolean; waterVersion(): number },
    terrain: StaticTerrain,
  ): void {
    const { w, h, cxForX, rowBaseForY, isLand, landTilesInCell } = terrain;
    isLand.fill(0);
    landTilesInCell.fill(0);
    let t = 0;
    for (let y = 0; y < h; y++) {
      const rowBase = rowBaseForY[y];
      for (let x = 0; x < w; x++, t++) {
        if (map.isLand(t)) {
          isLand[t] = 1;
          landTilesInCell[rowBase + cxForX[x]]++;
        }
      }
    }
    terrain.waterVersion = map.waterVersion();
  }

  encode(game: Game, agentPlayerID: PlayerID, out: Float32Array): void {
    if (out.length !== this.size) {
      throw new Error(`out must have length ${this.size}, got ${out.length}`);
    }
    out.fill(0);

    const agent = game.player(agentPlayerID);
    const map = game.map();
    const terrain = this.getStatic(map);
    const { w, h } = terrain;

    // Per-player lookups, computed once (O(numPlayers), not O(tiles)) as
    // flat typed arrays indexed directly by smallID — a dense small integer
    // (packed into 12 bits of tile state, see GameImpl.tileStateBuffer) — so
    // the 2M-tile hot loop below never pays JS Map's hashing cost per tile.
    // Swapping the two Maps this replaced for these arrays cut
    // encode()'s measured cost roughly 3x (see ObservationEncoder.test.ts's
    // throughput check and spec §11.1's resolution).
    const players = game.players();
    const agentSmallID = agent.smallID();
    let maxSmallID = agentSmallID;
    for (const p of players) maxSmallID = Math.max(maxSmallID, p.smallID());
    // 0 = self, 1 = ally, 2 = enemy, 255 = no player at this smallID.
    const relationCodeBySmallID = new Uint8Array(maxSmallID + 1).fill(255);
    // Troops-per-owned-tile, not total troops — the engine has no per-tile
    // troop data, so "density" has to be built from what it does have
    // (a player-level total and a tile count). Storing the player's raw
    // troop fraction here would make every tile that player owns report
    // that same total-troops value verbatim: not a density, just their
    // global troop count broadcast onto an ownership mask that's already
    // captured by planes 1/3 and GlobalField.SelfTroopFraction. Dividing by
    // their own tile count makes it "how thin are this player's troops
    // spread", which is genuinely per-player spatial information and
    // — because it's summed across owners per cell before normalizing by
    // that cell's land-tile count below — varies with which of possibly
    // several enemy owners hold a given cell, not just uniform per player.
    const troopPerTileBySmallID = new Float32Array(maxSmallID + 1);
    for (const p of players) {
      const code = p.smallID() === agentSmallID ? 0 : p.isFriendly(agent) ? 1 : 2;
      relationCodeBySmallID[p.smallID()] = code;
      troopPerTileBySmallID[p.smallID()] =
        p.troops() / TROOP_SCALE / Math.max(1, p.numTilesOwned());
    }
    // Reused scratch, zeroed per call. Same arrays every encode so the
    // ~500 KB of Float64 accumulators never hits the allocator.
    const landByCode = this.scratchLandByCode;
    const troopDensityByCode = this.scratchTroopByCode;
    const unownedLandInCell = this.scratchUnowned;
    const structureCountInCell = this.scratchStructures;
    for (let i = 0; i < landByCode.length; i++) landByCode[i].fill(0);
    for (let i = 0; i < troopDensityByCode.length; i++) troopDensityByCode[i].fill(0);
    unownedLandInCell.fill(0);
    structureCountInCell.fill(0);
    const selfOwnedTiles = this.scratchSelfTiles;
    selfOwnedTiles.length = 0;
    const borderingSmallIDs = this.scratchBordering;
    borderingSmallIDs.clear();

    const cellIndexOf = (x: number, y: number): number =>
      terrain.rowBaseForY[y] + terrain.cxForX[x];

    // Read ownerID straight out of the packed state plane instead of
    // through map.hasOwner(t)/map.ownerID(t)'s virtual dispatch — this loop
    // runs per tile (up to ~2M for World) and those two calls were a
    // measurable fraction of encode()'s cost. 0xfff mirrors
    // GameMap.tileStateBuffer()'s documented low-12-bits-are-ownerID layout
    // (relied on elsewhere per spec §2.6). Land comes from the static cache
    // built once per GameMap, so the loop pays no isLand() dispatch and no
    // float-division geometry per tile — just sequential array reads.
    const OWNER_ID_MASK = 0xfff;
    const stateBuf = map.tileStateBuffer();
    const { isLand, cxForX, rowBaseForY } = terrain;

    let t = 0;
    for (let y = 0; y < h; y++) {
      const cellRowBase = rowBaseForY[y];
      for (let x = 0; x < w; x++, t++) {
        if (isLand[t] === 0) continue;
        const cell = cellRowBase + cxForX[x];

        const ownerSmallID = stateBuf[t] & OWNER_ID_MASK;
        if (ownerSmallID === 0) {
          unownedLandInCell[cell]++;
          continue;
        }
        // Out-of-bounds (undefined) or the 255 sentinel both mean "no
        // tracked player at this smallID" — defensive against a dangling
        // ownerID (see buildWorldGame's doc comment on the engine crash
        // this same failure mode causes elsewhere in the engine).
        const code = relationCodeBySmallID[ownerSmallID];
        if (code === undefined || code === 255) continue;
        landByCode[code][cell]++;
        if (code === 0 || code === 2) {
          troopDensityByCode[code === 0 ? 0 : 1][cell] += troopPerTileBySmallID[ownerSmallID];
        }
        if (code === 0) selfOwnedTiles.push(t);
      }
    }

    // Bordering-opponent detection is bounded by the agent's own tile count,
    // not the whole map — cheap even late-game, and kept out of the O(tiles)
    // loop above so it doesn't pay map.neighbors()'s per-call allocation for
    // every one of 2M tiles.
    for (const tile of selfOwnedTiles) {
      for (const n of map.neighbors(tile)) {
        if (!map.hasOwner(n)) continue;
        const nSmallID = map.ownerID(n);
        if (nSmallID !== agentSmallID) borderingSmallIDs.add(nSmallID);
      }
    }

    // Structure presence: real unit positions, not a per-tile ownership proxy.
    // Uses the scratch buffer zeroed above; structure counts are sparse.
    for (const unit of game.units(Structures.types)) {
      const tile = unit.tile();
      const ux = map.x(tile);
      const uy = map.y(tile);
      if (ux < 0 || ux >= w || uy < 0 || uy >= h) continue;
      structureCountInCell[cellIndexOf(ux, uy)]++;
    }

    const { totalTilesInCell, landTilesInCell } = terrain;
    const raster = out.subarray(0, RASTER_SIZE);
    for (let cell = 0; cell < RASTER_CELLS; cell++) {
      const totalT = totalTilesInCell[cell];
      const landT = landTilesInCell[cell];
      raster[Plane.Land * RASTER_CELLS + cell] = totalT > 0 ? landT / totalT : 0;
      if (landT > 0) {
        raster[Plane.SelfOwnership * RASTER_CELLS + cell] = landByCode[0][cell] / landT;
        raster[Plane.AllyOwnership * RASTER_CELLS + cell] = landByCode[1][cell] / landT;
        raster[Plane.EnemyOwnership * RASTER_CELLS + cell] = landByCode[2][cell] / landT;
        raster[Plane.UnownedOwnership * RASTER_CELLS + cell] = unownedLandInCell[cell] / landT;
        raster[Plane.SelfTroopDensity * RASTER_CELLS + cell] = troopDensityByCode[0][cell] / landT;
        raster[Plane.EnemyTroopDensity * RASTER_CELLS + cell] = troopDensityByCode[1][cell] / landT;
      }
      // Structures are sparse; a handful in one cell already means "built
      // up" for the policy's purposes, so clip rather than let a rare
      // Factory-heavy cell dominate the plane's scale.
      raster[Plane.StructurePresence * RASTER_CELLS + cell] =
        Math.min(1, structureCountInCell[cell] / 4);
    }

    this.encodePlayerTable(
      out.subarray(RASTER_SIZE, RASTER_SIZE + PLAYER_TABLE_SIZE),
      game,
      agent,
      borderingSmallIDs,
    );
    this.encodeGlobals(out.subarray(RASTER_SIZE + PLAYER_TABLE_SIZE), game, agent);
  }

  private encodePlayerTable(
    out: Float32Array,
    game: Game,
    agent: Player,
    borderingSmallIDs: Set<number>,
  ): void {
    const totalLandTiles = game.map().numLandTiles();
    const unownedTiles =
      totalLandTiles -
      game.players().reduce((sum, p) => sum + p.numTilesOwned(), 0);

    // Slot 0: terra nullius.
    out[0 * NUM_SLOT_FIELDS + SlotField.Tiles] =
      totalLandTiles > 0 ? Math.max(0, unownedTiles) / totalLandTiles : 0;
    out[0 * NUM_SLOT_FIELDS + SlotField.Valid] = unownedTiles > 0 ? 1 : 0;

    // Slot order shared with the action codec's target head: assign once,
    // read here and there, so the policy's slot i is the codec's slot i.
    const slots = assignTargetSlots(game, agent, borderingSmallIDs);
    const maxOtherSlots = NUM_SLOTS - 1;
    for (let i = 0; i < maxOtherSlots; i++) {
      const p = slots[i + 1];
      const slot = i + 1;
      const base = slot * NUM_SLOT_FIELDS;
      if (p === null) {
        out[base + SlotField.Valid] = 0;
        continue;
      }
      out[base + SlotField.Troops] = p.troops() / TROOP_SCALE;
      out[base + SlotField.Gold] = Number(p.gold()) / GOLD_SCALE;
      out[base + SlotField.Tiles] =
        totalLandTiles > 0 ? p.numTilesOwned() / totalLandTiles : 0;
      out[base + SlotField.Allied] = agent.isAlliedWith(p) ? 1 : 0;
      out[base + SlotField.EmbargoedAgainstUs] = p.hasEmbargoAgainst(agent) ? 1 : 0;
      out[base + SlotField.Traitor] = p.isTraitor() ? 1 : 0;
      out[base + SlotField.SharedBorder] = borderingSmallIDs.has(p.smallID()) ? 1 : 0;
      out[base + SlotField.Valid] = 1;
    }
  }

  private encodeGlobals(out: Float32Array, game: Game, agent: Player): void {
    const players = game.players();
    const alive = players.filter((p) => p.isAlive());
    const totalLandTiles = game.map().numLandTiles();

    out[GlobalField.TickFraction] = Math.min(1, game.ticks() / TICK_CAP_REFERENCE);
    out[GlobalField.InSpawnPhase] = game.inSpawnPhase() ? 1 : 0;
    out[GlobalField.AlivePlayerFraction] =
      players.length > 0 ? alive.length / players.length : 0;

    const rankPool = alive.length > 0 ? alive : players;
    const sortedByTiles = [...rankPool].sort((a, b) => b.numTilesOwned() - a.numTilesOwned());
    const rankIndex = sortedByTiles.findIndex((p) => p.smallID() === agent.smallID());
    out[GlobalField.SelfRank] =
      sortedByTiles.length > 1 && rankIndex >= 0 ? rankIndex / (sortedByTiles.length - 1) : 0;

    out[GlobalField.SelfTroopFraction] = agent.troops() / TROOP_SCALE;
    out[GlobalField.SelfGoldFraction] = Number(agent.gold()) / GOLD_SCALE;
    out[GlobalField.SelfTileFraction] =
      totalLandTiles > 0 ? agent.numTilesOwned() / totalLandTiles : 0;
    out[GlobalField.SelfAlive] = agent.isAlive() ? 1 : 0;
    out[GlobalField.SelfTraitor] = agent.isTraitor() ? 1 : 0;
    out[GlobalField.SelfDoomsdayClock] = agent.inDoomsdayClock() ? 1 : 0;

    const maxOtherSlots = NUM_SLOTS - 1;
    const allyCount = players.filter(
      (p) => p.smallID() !== agent.smallID() && agent.isAlliedWith(p),
    ).length;
    const embargoedCount = players.filter(
      (p) => p.smallID() !== agent.smallID() && p.hasEmbargoAgainst(agent),
    ).length;
    out[GlobalField.SelfAllyFraction] = allyCount / maxOtherSlots;
    out[GlobalField.SelfEmbargoedFraction] = embargoedCount / maxOtherSlots;
  }
}
