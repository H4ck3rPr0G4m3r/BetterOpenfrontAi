import { Structures } from "../../OpenFrontIO/src/core/game/Game";
import type { Game, Player, PlayerID } from "../../OpenFrontIO/src/core/game/Game";
import { GOLD_SCALE, TROOP_SCALE } from "./Reward";
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
 * Encodes a `Game` snapshot into a fixed-size observation (spec §5.3).
 *
 * Full recompute from `map.ownerID()` / `map.isLand()` every call — no
 * incremental state, no dependency on `drainPackedTileUpdates()`. That
 * queue is destructive (whoever reads it consumes it), so incremental
 * maintenance would mean either GameEnv owns draining it exclusively or
 * this encoder does, and either way it needs per-episode state this
 * interface doesn't have. A full 128x64-cell recompute over the whole map
 * was measured cheap enough (see ObservationEncoder.test.ts's throughput
 * check) that the incremental design isn't needed in phase 1 — see spec
 * §11 for the record of that decision.
 */
export class ObservationEncoder implements ObservationEncoderInterface {
  readonly size = OBSERVATION_SIZE;

  encode(game: Game, agentPlayerID: PlayerID, out: Float32Array): void {
    if (out.length !== this.size) {
      throw new Error(`out must have length ${this.size}, got ${out.length}`);
    }
    out.fill(0);

    const agent = game.player(agentPlayerID);
    const map = game.map();
    const w = map.width();
    const h = map.height();
    const scale = Math.max(w / RASTER_WIDTH, h / RASTER_HEIGHT);
    const scaledW = w / scale;
    const scaledH = h / scale;
    const offsetX = Math.floor((RASTER_WIDTH - scaledW) / 2);
    const offsetY = Math.floor((RASTER_HEIGHT - scaledH) / 2);

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
    const totalTilesInCell = new Float64Array(RASTER_CELLS);
    const landTilesInCell = new Float64Array(RASTER_CELLS);
    // Indexed by relation code (0=self, 1=ally, 2=enemy) rather than a
    // string-keyed Record — this loop runs per land tile (up to ~2M for
    // World), and a string property lookup per tile was a measurable chunk
    // of encode()'s cost versus a plain array index.
    const landByCode = [
      new Float64Array(RASTER_CELLS),
      new Float64Array(RASTER_CELLS),
      new Float64Array(RASTER_CELLS),
    ];
    const troopDensityByCode = [new Float64Array(RASTER_CELLS), new Float64Array(RASTER_CELLS)]; // self, enemy only
    const unownedLandInCell = new Float64Array(RASTER_CELLS);
    const borderingSmallIDs = new Set<number>();

    // cellIndexOf(x, y) instead of cellIndexOf(tile) so the hot loop below
    // can track x/y with plain increments instead of calling map.x(t)/
    // map.y(t) (a division/modulo each) two million times. cy only depends
    // on y, so it's hoisted out of the inner x loop.
    const cellIndexOf = (x: number, y: number): number => {
      const cx = Math.min(RASTER_WIDTH - 1, Math.max(0, offsetX + Math.floor(x / scale)));
      const cy = Math.min(RASTER_HEIGHT - 1, Math.max(0, offsetY + Math.floor(y / scale)));
      return cy * RASTER_WIDTH + cx;
    };

    // Read ownerID straight out of the packed state plane instead of
    // through map.hasOwner(t)/map.ownerID(t)'s virtual dispatch — this loop
    // runs per tile (up to ~2M for World) and those two calls were a
    // measurable fraction of encode()'s cost. 0xfff mirrors
    // GameMap.tileStateBuffer()'s documented low-12-bits-are-ownerID layout
    // (relied on elsewhere per spec §2.6); land still goes through
    // map.isLand() since terrain isn't exposed as a bulk buffer.
    const OWNER_ID_MASK = 0xfff;
    const stateBuf = map.tileStateBuffer();

    const selfOwnedTiles: number[] = [];
    let t = 0;
    for (let y = 0; y < h; y++) {
      const cy = Math.min(RASTER_HEIGHT - 1, Math.max(0, offsetY + Math.floor(y / scale)));
      const cellRowBase = cy * RASTER_WIDTH;
      for (let x = 0; x < w; x++, t++) {
        const cx = Math.min(RASTER_WIDTH - 1, Math.max(0, offsetX + Math.floor(x / scale)));
        const cell = cellRowBase + cx;
        totalTilesInCell[cell]++;
        if (!map.isLand(t)) continue;
        landTilesInCell[cell]++;

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
    const structureCountInCell = new Float64Array(RASTER_CELLS);
    for (const unit of game.units(Structures.types)) {
      const tile = unit.tile();
      structureCountInCell[cellIndexOf(map.x(tile), map.y(tile))]++;
    }

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

    const others = game
      .players()
      .filter((p) => p.smallID() !== agent.smallID() && p.isAlive());
    others.sort((a, b) => {
      const aBorders = borderingSmallIDs.has(a.smallID()) ? 0 : 1;
      const bBorders = borderingSmallIDs.has(b.smallID()) ? 0 : 1;
      if (aBorders !== bBorders) return aBorders - bBorders;
      return b.numTilesOwned() - a.numTilesOwned();
    });

    const maxOtherSlots = NUM_SLOTS - 1;
    for (let i = 0; i < maxOtherSlots; i++) {
      const p = others[i];
      const slot = i + 1;
      const base = slot * NUM_SLOT_FIELDS;
      if (p === undefined) {
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
