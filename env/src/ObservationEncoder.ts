import type { Game, Player, PlayerID } from "../../OpenFrontIO/src/core/game/Game";
import { PlayerType, Structures, UnitType } from "../../OpenFrontIO/src/core/game/Game";
import { buildCandidates } from "./BuildSites";
import { cellGeometry, CellGeometry, tileCell } from "./Geometry";
import {
  FEATURE_SIZE,
  GLOBAL_OFFSET,
  GlobalField,
  MAX_BUILD_SITES,
  NUM_PLANES,
  NUM_SLOTS,
  Plane,
  RASTER_CELLS,
  RASTER_SIZE,
  SITE_FIELDS,
  SITE_OFFSET,
  SiteField,
  SLOT_FIELDS,
  SLOT_OFFSET,
  SlotField,
} from "./Layout";
import { assignTargetSlots, borderingSmallIDs } from "./TargetSlots";

export interface SeatObservation {
  /** NUM_PLANES x 64 x 128, values 0..255 (fraction * 255). */
  raster: Uint8Array;
  /** Slot table, build-site table, globals (see Layout.ts). */
  features: Float32Array;
}

export function createSeatObservation(): SeatObservation {
  return { raster: new Uint8Array(RASTER_SIZE), features: new Float32Array(FEATURE_SIZE) };
}

const OWNER_ID_MASK = 0xfff;
const ROLE_SELF = 0;
const ROLE_ALLY = 1;
const ROLE_THREAT = 2;
const ROLE_TARGET = 3;
const ROLES = 4;
export const MAX_SEATS = 8; // 4 role bits per seat in a u32

/** Squash a count into ~[0, 1] on a log scale: log1p(x / ref) / span. */
function logScale(x: number, ref: number, span: number): number {
  return Math.min(1, Math.log1p(Math.max(0, x) / ref) / span);
}
const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);

interface LandCache {
  waterVersion: number;
  isLand: Uint8Array;
  landInCell: Float32Array;
  totalInCell: Float32Array;
}

/**
 * Observation encoder, schema v2 (layout in Layout.ts).
 *
 * Every value is scaled to roughly [0, 1] by construction. The v1 encoder
 * put troop densities at 1e-4..1e-3 next to [0, 1] ownership planes (the
 * network could not see armies), had no notion of incoming or outgoing
 * attacks anywhere, and no troops-vs-max signal, so a policy could not
 * perceive the thing that kills it. v2 adds threat/target planes, per-player
 * attack and strength fields, troop ratio and growth, and the build-site
 * table the site head points into.
 *
 * Multi-seat: one tile pass serves all seats. Each owner smallID carries a
 * bitmask of (seat, role) pairs — self, ally, attacking-this-seat,
 * attacked-by-this-seat — so the 2M-tile World scan runs once per game step
 * instead of once per seat.
 */
export class ObservationEncoder {
  private landByMap = new WeakMap<object, LandCache>();
  private unowned = new Float32Array(RASTER_CELLS);
  private ownedAll = new Float32Array(RASTER_CELLS);
  private troopAll = new Float32Array(RASTER_CELLS);
  private roleCount = new Float32Array(0);
  private allyTroop = new Float32Array(0);
  private ownStructures = new Float32Array(0);
  private enemyStructures = new Float32Array(0);

  /** @param horizonTicks post-spawn ticks that map to TimeFraction = 1 (the env's tick cap). */
  constructor(private readonly horizonTicks: number = 6000) {}

  encode(game: Game, seatIDs: PlayerID[], out: SeatObservation[]): void {
    const S = seatIDs.length;
    if (S < 1 || S > MAX_SEATS) throw new Error(`seats must be 1..${MAX_SEATS}, got ${S}`);
    if (out.length !== S) throw new Error(`need ${S} outputs, got ${out.length}`);
    this.ensureScratch(S);

    const map = game.map();
    const geometry = cellGeometry(map.width(), map.height());
    const land = this.landCache(map, geometry);
    const seats = seatIDs.map((id) => game.player(id));
    const players = game.players();

    let maxSmallID = 0;
    for (const p of players) maxSmallID = Math.max(maxSmallID, p.smallID());
    const density = new Float32Array(maxSmallID + 1);
    for (const p of players) {
      density[p.smallID()] = logScale(p.troops() / Math.max(1, p.numTilesOwned()), 1, 8);
    }
    const roleBits = new Uint32Array(maxSmallID + 1);
    const setRole = (smallID: number, seat: number, role: number) => {
      if (smallID > 0 && smallID <= maxSmallID) roleBits[smallID] |= 1 << (seat * ROLES + role);
    };
    for (let s = 0; s < S; s++) {
      const seat = seats[s];
      setRole(seat.smallID(), s, ROLE_SELF);
      for (const p of players) {
        if (p !== seat && seat.isFriendly(p)) setRole(p.smallID(), s, ROLE_ALLY);
      }
      for (const a of seat.incomingAttacks()) setRole(a.attacker().smallID(), s, ROLE_THREAT);
      for (const a of seat.outgoingAttacks()) {
        const target = a.target();
        if (target.isPlayer()) setRole(target.smallID(), s, ROLE_TARGET);
      }
    }

    // One pass over every tile, shared by all seats.
    this.unowned.fill(0);
    this.ownedAll.fill(0);
    this.troopAll.fill(0);
    this.roleCount.fill(0);
    this.allyTroop.fill(0);
    const { unowned, ownedAll, troopAll, roleCount, allyTroop } = this;
    const stateBuf = map.tileStateBuffer();
    const { cxForX, rowBaseForY, w, h } = geometry;
    const isLand = land.isLand;
    let t = 0;
    for (let y = 0; y < h; y++) {
      const rowBase = rowBaseForY[y];
      for (let x = 0; x < w; x++, t++) {
        if (isLand[t] === 0) continue;
        const cell = rowBase + cxForX[x];
        const owner = stateBuf[t] & OWNER_ID_MASK;
        if (owner === 0) {
          unowned[cell]++;
          continue;
        }
        // Dangling ownerIDs (engine bug, see docs) have no player: skip.
        if (owner > maxSmallID) continue;
        ownedAll[cell]++;
        const d = density[owner];
        troopAll[cell] += d;
        let bits = roleBits[owner];
        while (bits !== 0) {
          const b = 31 - Math.clz32(bits & -bits);
          bits &= bits - 1;
          roleCount[b * RASTER_CELLS + cell]++;
          if ((b & 3) === ROLE_ALLY) allyTroop[(b >> 2) * RASTER_CELLS + cell] += d;
        }
      }
    }

    // Structures: own vs hostile per seat.
    this.ownStructures.fill(0);
    this.enemyStructures.fill(0);
    for (const unit of game.units(Structures.types)) {
      const ownerSmall = unit.owner().smallID();
      const cell = tileCell(unit.tile(), geometry);
      const bits = ownerSmall <= maxSmallID ? roleBits[ownerSmall] : 0;
      for (let s = 0; s < S; s++) {
        if (ownerSmall === seats[s].smallID()) this.ownStructures[s * RASTER_CELLS + cell]++;
        else if ((bits & (1 << (s * ROLES + ROLE_ALLY))) === 0) this.enemyStructures[s * RASTER_CELLS + cell]++;
      }
    }

    for (let s = 0; s < S; s++) {
      this.writeRaster(s, seats[s], density, land, out[s].raster);
      this.writeFeatures(game, seats[s], players, out[s].features, geometry);
    }
  }

  /** Single-seat convenience (live play, tests). */
  encodeOne(game: Game, playerID: PlayerID, out: SeatObservation): void {
    this.encode(game, [playerID], [out]);
  }

  private writeRaster(
    s: number,
    seat: Player,
    density: Float32Array,
    land: LandCache,
    raster: Uint8Array,
  ): void {
    const base = (role: number) => (s * ROLES + role) * RASTER_CELLS;
    const selfBase = base(ROLE_SELF);
    const allyBase = base(ROLE_ALLY);
    const threatBase = base(ROLE_THREAT);
    const targetBase = base(ROLE_TARGET);
    const seatDensity = seat.smallID() < density.length ? density[seat.smallID()] : 0;
    const q = (plane: number, cell: number, v: number) => {
      raster[plane * RASTER_CELLS + cell] = Math.round(clamp01(v) * 255);
    };
    for (let cell = 0; cell < RASTER_CELLS; cell++) {
      const landT = land.landInCell[cell];
      const totalT = land.totalInCell[cell];
      q(Plane.Land, cell, totalT > 0 ? landT / totalT : 0);
      const own = this.ownStructures[s * RASTER_CELLS + cell];
      const hostile = this.enemyStructures[s * RASTER_CELLS + cell];
      q(Plane.OwnStructures, cell, own / 3);
      q(Plane.EnemyStructures, cell, hostile / 3);
      if (landT <= 0) {
        for (const p of [Plane.Self, Plane.Ally, Plane.Enemy, Plane.Unowned, Plane.SelfTroops, Plane.EnemyTroops, Plane.Threat, Plane.Targets]) {
          raster[p * RASTER_CELLS + cell] = 0;
        }
        continue;
      }
      const self = this.roleCount[selfBase + cell];
      const ally = this.roleCount[allyBase + cell];
      const enemy = Math.max(0, this.ownedAll[cell] - self - ally);
      q(Plane.Self, cell, self / landT);
      q(Plane.Ally, cell, ally / landT);
      q(Plane.Enemy, cell, enemy / landT);
      q(Plane.Unowned, cell, this.unowned[cell] / landT);
      q(Plane.SelfTroops, cell, (self * seatDensity) / landT);
      const enemyTroops = this.troopAll[cell] - self * seatDensity - this.allyTroop[s * RASTER_CELLS + cell];
      q(Plane.EnemyTroops, cell, enemyTroops / landT);
      q(Plane.Threat, cell, this.roleCount[threatBase + cell] / landT);
      q(Plane.Targets, cell, this.roleCount[targetBase + cell] / landT);
    }
  }

  private writeFeatures(
    game: Game,
    seat: Player,
    players: Player[],
    out: Float32Array,
    geometry: CellGeometry,
  ): void {
    out.fill(0);
    const config = game.config();
    const alive = seat.isAlive();
    const ourTroops = seat.troops();
    const ourTiles = seat.numTilesOwned();
    const totalLand = Math.max(1, game.map().numLandTiles());

    const incomingBy = new Map<number, number>();
    let incomingTroops = 0;
    for (const a of seat.incomingAttacks()) {
      const id = a.attacker().smallID();
      incomingBy.set(id, (incomingBy.get(id) ?? 0) + a.troops());
      incomingTroops += a.troops();
    }
    const outgoingBy = new Map<number, number>();
    let outgoingTroops = 0;
    for (const a of seat.outgoingAttacks()) {
      const target = a.target();
      const id = target.isPlayer() ? target.smallID() : 0;
      outgoingBy.set(id, (outgoingBy.get(id) ?? 0) + a.troops());
      outgoingTroops += a.troops();
    }
    const vsUs = (troops: number) => clamp01(troops / Math.max(1, ourTroops) / 2);

    // Slot table. Slot 0 is terra nullius (expansion target).
    const bordering = borderingSmallIDs(game, seat);
    const slots = assignTargetSlots(game, seat, bordering);
    const owned = players.reduce((sum, p) => sum + p.numTilesOwned(), 0);
    const unownedTiles = Math.max(0, game.map().numLandTiles() - owned);
    const slot0 = SLOT_OFFSET;
    out[slot0 + SlotField.Valid] = unownedTiles > 0 ? 1 : 0;
    out[slot0 + SlotField.Tiles] = logScale(unownedTiles, 100, 10);
    out[slot0 + SlotField.TileShareVsUs] = unownedTiles / Math.max(1, unownedTiles + ourTiles);
    out[slot0 + SlotField.SharesBorder] = bordering.has(0) ? 1 : 0;
    out[slot0 + SlotField.WeAttack] = vsUs(outgoingBy.get(0) ?? 0);
    const requesting = new Set(seat.incomingAllianceRequests().map((r) => r.requestor().smallID()));
    for (let i = 1; i < NUM_SLOTS; i++) {
      const p = slots[i];
      if (p === null) continue;
      const base = SLOT_OFFSET + i * SLOT_FIELDS;
      const tiles = p.numTilesOwned();
      const troops = p.troops();
      const type = p.type();
      out[base + SlotField.Valid] = 1;
      out[base + SlotField.Tiles] = logScale(tiles, 100, 10);
      out[base + SlotField.TileShareVsUs] = tiles / Math.max(1, tiles + ourTiles);
      out[base + SlotField.Troops] = logScale(troops, 1000, 7);
      out[base + SlotField.StrengthVsUs] = troops / Math.max(1, troops + ourTroops);
      out[base + SlotField.Density] = logScale(troops / Math.max(1, tiles), 1, 8);
      out[base + SlotField.Allied] = seat.isAlliedWith(p) ? 1 : 0;
      out[base + SlotField.SharesBorder] = bordering.has(p.smallID()) ? 1 : 0;
      out[base + SlotField.AttackingUs] = vsUs(incomingBy.get(p.smallID()) ?? 0);
      out[base + SlotField.WeAttack] = vsUs(outgoingBy.get(p.smallID()) ?? 0);
      out[base + SlotField.IsBot] = type === PlayerType.Bot ? 1 : 0;
      out[base + SlotField.IsNation] = type === PlayerType.Nation ? 1 : 0;
      out[base + SlotField.IsHuman] = type === PlayerType.Human ? 1 : 0;
      out[base + SlotField.Traitor] = p.isTraitor() ? 1 : 0;
      out[base + SlotField.EmbargoAgainstUs] = p.hasEmbargoAgainst(seat) ? 1 : 0;
      out[base + SlotField.RequestsAlliance] = requesting.has(p.smallID()) ? 1 : 0;
    }

    // Build-site table, same list the codec's site head indexes.
    if (alive) {
      const candidates = buildCandidates(game, seat);
      for (let i = 0; i < candidates.length && i < MAX_BUILD_SITES; i++) {
        const c = candidates[i];
        const base = SITE_OFFSET + i * SITE_FIELDS;
        out[base + SiteField.Valid] = 1;
        out[base + SiteField.TypeStart + c.typeIndex] = 1;
        out[base + SiteField.X] = (c.tile % geometry.w) / geometry.w;
        out[base + SiteField.Y] = Math.floor(c.tile / geometry.w) / geometry.h;
        out[base + SiteField.Border] = c.border ? 1 : 0;
        out[base + SiteField.RailLink] = c.rail ? 1 : 0;
      }
    }

    // Globals.
    const g = GLOBAL_OFFSET;
    const spawnTicks = config.numSpawnPhaseTurns();
    out[g + GlobalField.TimeFraction] = clamp01((game.ticks() - spawnTicks) / this.horizonTicks);
    // game.players() is living players only; allPlayers() includes the dead.
    const living = players;
    const everyone = Math.max(1, game.allPlayers().length);
    out[g + GlobalField.AliveFraction] = living.length / everyone;
    const larger = living.filter((p) => p.numTilesOwned() > ourTiles).length;
    out[g + GlobalField.Rank] = living.length > 1 ? larger / (living.length - 1) : 0;
    out[g + GlobalField.Tiles] = logScale(ourTiles, 100, 10);
    out[g + GlobalField.LandShare] = clamp01((ourTiles / totalLand) * 10);
    out[g + GlobalField.Troops] = logScale(ourTroops, 1000, 7);
    if (alive) {
      const maxTroops = Math.max(1, config.maxTroops(seat));
      out[g + GlobalField.TroopRatio] = clamp01(ourTroops / maxTroops);
      out[g + GlobalField.GrowthRate] = clamp01((config.troopIncreaseRate(seat) / maxTroops) * 100);
      const cityCost = Number(game.unitInfo(UnitType.City).cost(game, seat));
      out[g + GlobalField.CityAffordable] = cityCost > 0 ? clamp01(Number(seat.gold()) / cityCost) : 0;
    }
    out[g + GlobalField.OutgoingTroops] = clamp01(outgoingTroops / Math.max(1, ourTroops + outgoingTroops));
    out[g + GlobalField.IncomingTroops] = vsUs(incomingTroops);
    out[g + GlobalField.IncomingCount] = clamp01(seat.incomingAttacks().length / 4);
    out[g + GlobalField.OutgoingCount] = clamp01(seat.outgoingAttacks().length / 4);
    out[g + GlobalField.Gold] = logScale(Number(seat.gold()), 1000, 10);
    out[g + GlobalField.Cities] = clamp01(seat.unitCount(UnitType.City) / 10);
    out[g + GlobalField.Ports] = clamp01(seat.unitCount(UnitType.Port) / 5);
    out[g + GlobalField.Factories] = clamp01(seat.unitCount(UnitType.Factory) / 5);
    out[g + GlobalField.DefensePosts] = clamp01(seat.unitCount(UnitType.DefensePost) / 10);
    out[g + GlobalField.SAMs] = clamp01(seat.unitCount(UnitType.SAMLauncher) / 5);
    out[g + GlobalField.Silos] = clamp01(seat.unitCount(UnitType.MissileSilo) / 5);
    out[g + GlobalField.Warships] = clamp01(seat.unitCount(UnitType.Warship) / 5);
    out[g + GlobalField.BoatsAtSea] = clamp01(seat.unitCount(UnitType.TransportShip) / 3);
    out[g + GlobalField.Alive] = alive ? 1 : 0;
    out[g + GlobalField.Traitor] = seat.isTraitor() ? 1 : 0;
    const others = players.filter((p) => p !== seat);
    out[g + GlobalField.Allies] = clamp01(others.filter((p) => seat.isAlliedWith(p)).length / 16);
    out[g + GlobalField.Embargoed] = clamp01(others.filter((p) => p.hasEmbargoAgainst(seat)).length / 16);
    out[g + GlobalField.InSpawnPhase] = game.inSpawnPhase() ? 1 : 0;
    out[g + GlobalField.AllianceRequests] = clamp01(requesting.size / 4);
  }

  private ensureScratch(S: number): void {
    if (this.roleCount.length === S * ROLES * RASTER_CELLS) return;
    this.roleCount = new Float32Array(S * ROLES * RASTER_CELLS);
    this.allyTroop = new Float32Array(S * RASTER_CELLS);
    this.ownStructures = new Float32Array(S * RASTER_CELLS);
    this.enemyStructures = new Float32Array(S * RASTER_CELLS);
  }

  /** Land membership per tile and land/total tile counts per cell, cached per
   *  GameMap and rebuilt when nukes bump waterVersion. */
  private landCache(
    map: { isLand(ref: number): boolean; waterVersion(): number },
    geometry: CellGeometry,
  ): LandCache {
    let cache = this.landByMap.get(map);
    if (cache !== undefined && cache.waterVersion === map.waterVersion()) return cache;
    const { w, h, cxForX, rowBaseForY } = geometry;
    cache ??= {
      waterVersion: -1,
      isLand: new Uint8Array(w * h),
      landInCell: new Float32Array(RASTER_CELLS),
      totalInCell: new Float32Array(RASTER_CELLS),
    };
    cache.landInCell.fill(0);
    cache.totalInCell.fill(0);
    let t = 0;
    for (let y = 0; y < h; y++) {
      const rowBase = rowBaseForY[y];
      for (let x = 0; x < w; x++, t++) {
        const cell = rowBase + cxForX[x];
        cache.totalInCell[cell]++;
        const isLand = map.isLand(t) ? 1 : 0;
        cache.isLand[t] = isLand;
        cache.landInCell[cell] += isLand;
      }
    }
    cache.waterVersion = map.waterVersion();
    this.landByMap.set(map, cache);
    return cache;
  }
}

export { NUM_PLANES, RASTER_SIZE };
