import type {
  Game,
  Player,
  PlayerBuildableUnitType,
  PlayerID,
} from "../../OpenFrontIO/src/core/game/Game";
import { UnitType as UnitTypeEnum } from "../../OpenFrontIO/src/core/game/Game";
import type { TileRef } from "../../OpenFrontIO/src/core/game/GameMap";
import type { Intent } from "../../OpenFrontIO/src/core/Schemas";
import {
  letterboxTransform,
  RASTER_CELLS,
  RASTER_HEIGHT,
  RASTER_WIDTH,
} from "./ObservationEncoder";
import { assignTargetSlots, borderingSmallIDs, slotPlayer } from "./TargetSlots";
import type { Action, ActionCodec, MaskBuffers } from "./types";

/**
 * The five action heads (spec §5.4), with exact sizes. Head values index
 * these in order; IntentType's order is the head's order.
 */
export enum IntentType {
  Noop = 0,
  Attack = 1,
  Boat = 2,
  BuildUnit = 3,
  UpgradeStructure = 4,
  MoveWarship = 5,
  TargetPlayer = 6,
  AllianceRequest = 7,
  AllianceReject = 8,
  BreakAlliance = 9,
  Embargo = 10,
  DonateGold = 11,
  DonateTroops = 12,
  CancelAttack = 13,
}
export const NUM_INTENT_TYPES = 14;

export const NUM_TARGET_SLOTS = 17; // slot 0 = terra nullius, 1-16 = players

/** Troop/gold fraction per ratio bucket: 5/10/20/30/50/70/90/100%. */
export const RATIO_BUCKETS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
export const NUM_RATIO_BUCKETS = RATIO_BUCKETS.length;

export const NUM_SPATIAL_CELLS = RASTER_CELLS; // 128x64, shared with encoder

/** Bounded build-site candidate list (spec §11.1: dense masking rejected). */
export const MAX_BUILD_SITES = 32;

/**
 * Unit types competing for build-site slots. Structures plus Warship — the
 * economy and coastal-defense game. Nukes (Atom/Hydrogen/MIRV) are cheap to
 * query but need silos and late-game gold; TransportShip is covered by the
 * boat intent instead. Deliberate v1 scope cut, not an oversight.
 */
export const BUILD_CANDIDATE_TYPES: PlayerBuildableUnitType[] = [
  UnitTypeEnum.City,
  UnitTypeEnum.DefensePost,
  UnitTypeEnum.Port,
  UnitTypeEnum.SAMLauncher,
  UnitTypeEnum.MissileSilo,
  UnitTypeEnum.Factory,
  UnitTypeEnum.Warship,
];
/** How many (rotated) types each candidate site contributes. */
const TYPES_PER_SITE = 3;
export function createMaskBuffers(): MaskBuffers {
  return {
    intentType: new Uint8Array(NUM_INTENT_TYPES),
    target: new Uint8Array(NUM_TARGET_SLOTS),
    ratioBucket: new Uint8Array(NUM_RATIO_BUCKETS),
    spatialCell: new Uint8Array(NUM_SPATIAL_CELLS),
    buildSite: new Uint8Array(MAX_BUILD_SITES),
  };
}

/** How many build-site slots are populated for this state (≤ MAX_BUILD_SITES). */
export function countBuildCandidates(game: Game, playerID: PlayerID): number {
  return buildCandidates(buildContext(game, playerID)).length;
}

export interface BuildCandidate {
  tile: TileRef;
  type: PlayerBuildableUnitType;
}

interface CodecContext {
  game: Game;
  agent: Player;
  agentID: PlayerID;
  /** Neighboring smallIDs including 0 (unowned land). */
  bordering: Set<number>;
  slots: (Player | null)[];
  bySmallID: Map<number, Player>;
}

function buildContext(game: Game, playerID: PlayerID): CodecContext {
  const agent = game.player(playerID);
  const bordering = borderingSmallIDs(game, agent);
  const slots = assignTargetSlots(game, agent, bordering);
  const bySmallID = new Map<number, Player>();
  for (const p of game.players()) bySmallID.set(p.smallID(), p);
  return { game, agent, agentID: playerID, bordering, slots, bySmallID };
}

function checkAction(action: Action): void {
  if (action.intentType < 0 || action.intentType >= NUM_INTENT_TYPES) {
    throw new RangeError(`intentType ${action.intentType} out of range`);
  }
  if (action.target < 0 || action.target >= NUM_TARGET_SLOTS) {
    throw new RangeError(`target ${action.target} out of range`);
  }
  if (action.ratioBucket < 0 || action.ratioBucket >= NUM_RATIO_BUCKETS) {
    throw new RangeError(`ratioBucket ${action.ratioBucket} out of range`);
  }
  if (action.spatialCell < 0 || action.spatialCell >= NUM_SPATIAL_CELLS) {
    throw new RangeError(`spatialCell ${action.spatialCell} out of range`);
  }
  if (action.buildSite < 0 || action.buildSite >= MAX_BUILD_SITES) {
    throw new RangeError(`buildSite ${action.buildSite} out of range`);
  }
}

function checkMaskBuffers(out: MaskBuffers): void {
  if (out.intentType.length !== NUM_INTENT_TYPES) {
    throw new Error(`intentType mask must have length ${NUM_INTENT_TYPES}`);
  }
  if (out.target.length !== NUM_TARGET_SLOTS) {
    throw new Error(`target mask must have length ${NUM_TARGET_SLOTS}`);
  }
  if (out.ratioBucket.length !== NUM_RATIO_BUCKETS) {
    throw new Error(`ratioBucket mask must have length ${NUM_RATIO_BUCKETS}`);
  }
  if (out.spatialCell.length !== NUM_SPATIAL_CELLS) {
    throw new Error(`spatialCell mask must have length ${NUM_SPATIAL_CELLS}`);
  }
  if (out.buildSite.length !== MAX_BUILD_SITES) {
    throw new Error(`buildSite mask must have length ${MAX_BUILD_SITES}`);
  }
}

/**
 * Bounded build-site candidates (spec §5.4/§11.1): spread-sampled border
 * tiles plus a few interior sites, each contributing TYPES_PER_SITE rotated
 * unit types, truncated to MAX_BUILD_SITES. Deterministic — sorted numerics
 * throughout — so the same game state always yields the same slots.
 */
export function buildCandidates(ctx: CodecContext): BuildCandidate[] {
  const border = [...ctx.agent.borderTiles()].sort((a, b) => a - b);
  const maxBorderSites = 10;
  const borderSites: TileRef[] = [];
  if (border.length > 0) {
    const stride = Math.max(1, Math.floor(border.length / maxBorderSites));
    for (let i = 0; i < border.length && borderSites.length < maxBorderSites; i += stride) {
      borderSites.push(border[i]);
    }
  }

  // A few interior sites for structures that want depth. One strided pass
  // over owned tiles, no per-tile engine calls.
  const interiorSites: TileRef[] = [];
  const owned = ctx.agent.numTilesOwned();
  if (owned > border.length) {
    const borderSet = new Set<TileRef>(border);
    const step = Math.max(1, Math.floor(owned / 4));
    let i = 0;
    for (const tile of ctx.agent.tiles()) {
      if (!borderSet.has(tile) && i++ % step === 0) {
        interiorSites.push(tile);
        if (interiorSites.length >= 4) break;
      }
    }
    interiorSites.sort((a, b) => a - b);
  }

  const sites = [...borderSites, ...interiorSites];
  const candidates: BuildCandidate[] = [];
  const nTypes = BUILD_CANDIDATE_TYPES.length;
  for (let s = 0; s < sites.length && candidates.length < MAX_BUILD_SITES; s++) {
    for (let j = 0; j < TYPES_PER_SITE && candidates.length < MAX_BUILD_SITES; j++) {
      candidates.push({ tile: sites[s], type: BUILD_CANDIDATE_TYPES[(s + j) % nTypes] });
    }
  }
  return candidates;
}

interface CellRect {
  x0: number;
  x1: number; // exclusive
  y0: number;
  y1: number; // exclusive
}

function cellRect(cell: number, w: number, h: number): CellRect {
  const cx = cell % RASTER_WIDTH;
  const cy = Math.floor(cell / RASTER_WIDTH);
  const { scale, offsetX, offsetY } = letterboxTransform(w, h);
  return {
    x0: Math.max(0, Math.floor((cx - offsetX) * scale)),
    x1: Math.min(w, Math.ceil((cx - offsetX + 1) * scale)),
    y0: Math.max(0, Math.floor((cy - offsetY) * scale)),
    y1: Math.min(h, Math.ceil((cy - offsetY + 1) * scale)),
  };
}

/** Representative tile at a cell's centre, always a valid ref. Exported: tests
 *  and the policy use the same sampling the codec resolves from. */
export function cellCenterTile(cell: number, w: number, h: number): TileRef {
  const cx = cell % RASTER_WIDTH;
  const cy = Math.floor(cell / RASTER_WIDTH);
  const { scale, offsetX, offsetY } = letterboxTransform(w, h);
  const x = Math.min(w - 1, Math.max(0, Math.floor((cx - offsetX + 0.5) * scale)));
  const y = Math.min(h - 1, Math.max(0, Math.floor((cy - offsetY + 0.5) * scale)));
  return y * w + x;
}

/** True when the cell's centre falls outside the map — letterbox padding. Exported: the policy must ignore these cells too. */
export function isPaddingCell(cell: number, w: number, h: number): boolean {
  const cx = cell % RASTER_WIDTH;
  const cy = Math.floor(cell / RASTER_WIDTH);
  const { scale, offsetX, offsetY } = letterboxTransform(w, h);
  const x = (cx - offsetX + 0.5) * scale;
  const y = (cy - offsetY + 0.5) * scale;
  return x < 0 || x >= w || y < 0 || y >= h;
}

/**
 * Resolve a spatial cell to a concrete boat destination: enemy-owned land
 * first, then unowned land, else the cell centre. A bounded scan of one
 * cell's tiles (~250 on World), not a map-wide search.
 */
function resolveBoatDst(ctx: CodecContext, cell: number): TileRef {
  const map = ctx.game.map();
  const w = map.width();
  const h = map.height();
  const r = cellRect(cell, w, h);
  let fallback = cellCenterTile(cell, w, h);
  let unownedFallback: TileRef | null = null;
  for (let y = r.y0; y < r.y1; y++) {
    for (let x = r.x0; x < r.x1; x++) {
      const tile = y * w + x;
      if (!map.isLand(tile) || map.isImpassable(tile)) continue;
      const owner = map.ownerID(tile);
      if (owner === ctx.agent.smallID()) continue;
      if (owner === 0) {
        unownedFallback ??= tile;
        continue;
      }
      return tile;
    }
  }
  return unownedFallback ?? fallback;
}

/** Resolve a spatial cell to a warship patrol tile: water first, else centre. */
function resolveWaterDst(ctx: CodecContext, cell: number): TileRef {
  const map = ctx.game.map();
  const w = map.width();
  const h = map.height();
  const r = cellRect(cell, w, h);
  for (let y = r.y0; y < r.y1; y++) {
    for (let x = r.x0; x < r.x1; x++) {
      const tile = y * w + x;
      if (!map.isLand(tile)) return tile;
    }
  }
  return cellCenterTile(cell, w, h);
}

/** Exact bigint math for the gold fraction — gold exceeds float precision. */
function goldFraction(gold: bigint, ratio: number): bigint {
  return (gold * BigInt(Math.round(ratio * 100))) / 100n;
}

/**
 * Decode a five-head action into engine intents (spec §5.4).
 *
 * Contract: out-of-range head values throw (caller bug, fail fast), but an
 * in-range combination that is illegal in the current state returns [] — the
 * factorized heads cannot express every cross-head condition in the marginal
 * masks, so decode re-validates the specific combination and drops it rather
 * than emitting an intent the engine would no-op or, worse, half-execute
 * (attacking an ally embargoes them before the execution deactivates — that
 * must never fire from a masked-out combo).
 */
export class ActionCodecImpl implements ActionCodec {
  decode(action: Action, game: Game, playerID: PlayerID): Intent[] {
    checkAction(action);
    if (action.intentType === IntentType.Noop) return [];

    const ctx = buildContext(game, playerID);
    // The episode is over; nothing below can legally progress it.
    if (!ctx.agent.isAlive()) return [];
    const ratio = RATIO_BUCKETS[action.ratioBucket];

    switch (action.intentType) {
      case IntentType.Attack:
        return this.decodeAttack(ctx, action, ratio);
      case IntentType.Boat:
        return this.decodeBoat(ctx, action, ratio);
      case IntentType.BuildUnit:
        return this.decodeBuild(ctx, action, false);
      case IntentType.UpgradeStructure:
        return this.decodeBuild(ctx, action, true);
      case IntentType.MoveWarship:
        return this.decodeMoveWarship(ctx, action);
      case IntentType.TargetPlayer:
        return this.decodeTargetPlayer(ctx, action);
      case IntentType.AllianceRequest:
        return this.decodeDiplomacy(ctx, action, "allianceRequest");
      case IntentType.AllianceReject:
        return this.decodeDiplomacy(ctx, action, "allianceReject");
      case IntentType.BreakAlliance:
        return this.decodeDiplomacy(ctx, action, "breakAlliance");
      case IntentType.Embargo:
        return this.decodeDiplomacy(ctx, action, "embargo");
      case IntentType.DonateGold:
        return this.decodeDiplomacy(ctx, action, "donateGold", ratio);
      case IntentType.DonateTroops:
        return this.decodeDiplomacy(ctx, action, "donateTroops", ratio);
      case IntentType.CancelAttack:
        return this.decodeCancelAttack(ctx);
      default:
        throw new RangeError(`unknown intentType ${action.intentType}`);
    }
  }

  private decodeAttack(ctx: CodecContext, action: Action, ratio: number): Intent[] {
    const troops = ratio * ctx.agent.troops();
    if (troops <= 0) return [];
    if (action.target === 0) {
      // Terra nullius: expand into adjacent unowned land (spec §2.4/§2.7).
      if (!ctx.bordering.has(0)) return [];
      return [{ type: "attack", targetID: null, troops } as Intent];
    }
    const raw = ctx.slots[action.target];
    if (raw === null) return [];
    const target = slotPlayer(ctx.slots, ctx.agentID, ctx.game, action.target);
    if (target === null) return [];
    // Guarded here, not just in the mask: attacking an ally still embargoes
    // them and rejects alliance requests before the execution deactivates.
    if (!ctx.agent.canAttackPlayer(target)) return [];
    if (!ctx.agent.sharesBorderWith(target)) return [];
    return [{ type: "attack", targetID: target.id(), troops } as Intent];
  }

  private decodeBoat(ctx: CodecContext, action: Action, ratio: number): Intent[] {
    const troops = ratio * ctx.agent.troops();
    if (troops <= 0) return [];
    const dst = resolveBoatDst(ctx, action.spatialCell);
    const map = ctx.game.map();
    if (!map.isLand(dst) || map.isImpassable(dst)) return [];
    const ownerID = map.ownerID(dst);
    if (ownerID === ctx.agent.smallID()) return [];
    if (ownerID !== 0) {
      const owner = ctx.bySmallID.get(ownerID);
      if (owner === undefined || !ctx.agent.canAttackPlayer(owner)) return [];
    }
    // The exact predicate TransportShipExecution.init uses to stand down —
    // covers the boat cap, gold, and shore reachability in one call.
    if (ctx.agent.canBuild(UnitTypeEnum.TransportShip, dst) === false) return [];
    return [{ type: "boat", troops, dst } as Intent];
  }

  private decodeBuild(ctx: CodecContext, action: Action, upgrade: boolean): Intent[] {
    const candidates = buildCandidates(ctx);
    if (action.buildSite >= candidates.length) return [];
    const candidate = candidates[action.buildSite];
    const query = ctx.agent.buildableUnits(candidate.tile, [candidate.type])[0];
    if (upgrade) {
      if (query.canUpgrade === false) return [];
      return [
        { type: "upgrade_structure", unit: candidate.type, unitId: query.canUpgrade } as Intent,
      ];
    }
    // Send the engine-resolved spawn tile, not the candidate tile — the two
    // differ when the resolver adjusts placement.
    if (query.canBuild === false) return [];
    return [{ type: "build_unit", unit: candidate.type, tile: query.canBuild } as Intent];
  }

  private decodeMoveWarship(ctx: CodecContext, action: Action): Intent[] {
    const warships = ctx.agent.units(UnitTypeEnum.Warship).filter((u) => u.isActive());
    if (warships.length === 0) return [];
    const dst = resolveWaterDst(ctx, action.spatialCell);
    const map = ctx.game.map();
    if (dst < 0 || dst >= map.width() * map.height()) return [];
    // Patrol tiles are water by definition. A land fallback would name ships
    // that all silently ignore the order — a no-op intent decode must not
    // emit, same contract as the attack guards above.
    if (map.isLand(dst)) return [];
    // Sail only the warships sharing the destination's water — others would
    // silently ignore the order. When none match, send them all anyway: the
    // schema needs a nonempty array and the execution skips mismatches.
    const component = ctx.game.getWaterComponent(dst);
    const same =
      component === null
        ? []
        : warships.filter((w) => ctx.game.hasWaterComponent(w.tile(), component));
    const ids = (same.length > 0 ? same : warships).map((w) => w.id());
    return [{ type: "move_warship", unitIds: ids, tile: dst } as Intent];
  }

  private decodeTargetPlayer(ctx: CodecContext, action: Action): Intent[] {
    if (ctx.slots[action.target] === null) return [];
    const target = slotPlayer(ctx.slots, ctx.agentID, ctx.game, action.target);
    if (target === null || !ctx.agent.canTarget(target)) return [];
    return [{ type: "targetPlayer", target: target.id() } as Intent];
  }

  private decodeDiplomacy(
    ctx: CodecContext,
    action: Action,
    kind: "allianceRequest" | "allianceReject" | "breakAlliance" | "embargo" | "donateGold" | "donateTroops",
    ratio?: number,
  ): Intent[] {
    const raw = ctx.slots[action.target];
    if (raw === null) return [];
    const target = slotPlayer(ctx.slots, ctx.agentID, ctx.game, action.target);
    if (target === null) return [];
    switch (kind) {
      case "allianceRequest":
        if (!ctx.agent.canSendAllianceRequest(target)) return [];
        return [{ type: "allianceRequest", recipient: target.id() } as Intent];
      case "allianceReject": {
        const pending = ctx.agent
          .incomingAllianceRequests()
          .some((r) => r.requestor() === target);
        if (!pending) return [];
        return [{ type: "allianceReject", requestor: target.id() } as Intent];
      }
      case "breakAlliance":
        if (ctx.agent.allianceWith(target) === null) return [];
        return [{ type: "breakAlliance", recipient: target.id() } as Intent];
      case "embargo": {
        const toggle = ctx.agent.hasEmbargoAgainst(target) ? "stop" : "start";
        return [{ type: "embargo", targetID: target.id(), action: toggle } as Intent];
      }
      case "donateGold": {
        if (!ctx.agent.canDonateGold(target)) return [];
        const gold = goldFraction(ctx.agent.gold(), ratio ?? 0);
        if (gold <= 0n) return [];
        return [{ type: "donate_gold", recipient: target.id(), gold: Number(gold) } as Intent];
      }
      case "donateTroops": {
        if (!ctx.agent.canDonateTroops(target)) return [];
        const troops = (ratio ?? 0) * ctx.agent.troops();
        if (troops <= 0) return [];
        return [{ type: "donate_troops", recipient: target.id(), troops } as Intent];
      }
    }
  }

  private decodeCancelAttack(ctx: CodecContext): Intent[] {
    const attacks = ctx.agent.outgoingAttacks();
    if (attacks.length === 0) return [];
    return [{ type: "cancel_attack", attackID: attacks[0].id() } as Intent];
  }

  /**
   * Per-head legality masks. Each head is marginal — a value is unmasked when
   * SOME combination using it is legal — because masks() cannot see the other
   * heads' selections. decode() re-validates the actual combination and drops
   * illegal ones as [].
   */
  masks(game: Game, playerID: PlayerID, out: MaskBuffers): void {
    checkMaskBuffers(out);
    const ctx = buildContext(game, playerID);
    const alive = ctx.agent.isAlive();
    const troops = ctx.agent.troops();

    // Bordering players the agent can legally strike. Membership in the
    // bordering set already implies a shared border — no per-player engine
    // call needed for that half.
    let anyBorderTarget = false;
    for (const id of ctx.bordering) {
      if (id === 0) continue;
      const p = ctx.bySmallID.get(id);
      if (p !== undefined && p.id() !== ctx.agentID && p.isAlive() && ctx.agent.canAttackPlayer(p)) {
        anyBorderTarget = true;
        break;
      }
    }
    const hasUnownedNeighbor = ctx.bordering.has(0);

    // Spatial pass first: the boat marginal needs to know a launchable cell
    // exists, which this loop determines.
    const boatTiles: TileRef[] = [];
    this.maskSpatial(ctx, out.spatialCell, (tile) => {
      if (boatTiles.length < 8) boatTiles.push(tile);
    });

    // Build pass: one single-type query per candidate; shared by the
    // intent-type marginals and the build-site mask.
    const candidates = alive ? buildCandidates(ctx) : [];
    const buildable = new Array<boolean>(MAX_BUILD_SITES).fill(false);
    const upgradeable = new Array<boolean>(MAX_BUILD_SITES).fill(false);
    let anyBuildable = false;
    let anyUpgradeable = false;
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      const q = ctx.agent.buildableUnits(c.tile, [c.type])[0];
      if (q.canBuild !== false) {
        buildable[i] = true;
        anyBuildable = true;
      }
      if (q.canUpgrade !== false) {
        upgradeable[i] = true;
        anyUpgradeable = true;
      }
    }

    const occupied = ctx.slots
      .map((p, i) => (i > 0 && p !== null ? p : null))
      .filter((p): p is Player => p !== null);

    const intent = out.intentType;
    intent[IntentType.Noop] = 1;
    intent[IntentType.Attack] =
      alive && troops > 0 && (hasUnownedNeighbor || anyBorderTarget) ? 1 : 0;
    intent[IntentType.Boat] = this.boatUsable(ctx, boatTiles) ? 1 : 0;
    intent[IntentType.BuildUnit] = anyBuildable ? 1 : 0;
    intent[IntentType.UpgradeStructure] = anyUpgradeable ? 1 : 0;
    intent[IntentType.MoveWarship] =
      alive && ctx.agent.units(UnitTypeEnum.Warship).some((u) => u.isActive()) ? 1 : 0;
    intent[IntentType.TargetPlayer] = occupied.some((p) => ctx.agent.canTarget(p)) ? 1 : 0;
    intent[IntentType.AllianceRequest] = occupied.some((p) =>
      ctx.agent.canSendAllianceRequest(p),
    )
      ? 1
      : 0;
    intent[IntentType.AllianceReject] =
      ctx.agent.incomingAllianceRequests().length > 0 ? 1 : 0;
    intent[IntentType.BreakAlliance] = ctx.agent.alliances().length > 0 ? 1 : 0;
    // Any occupied slot accepts at least an embargo toggle, so the target
    // mask below is occupancy and this marginal matches it.
    intent[IntentType.Embargo] = occupied.length > 0 ? 1 : 0;
    intent[IntentType.DonateGold] = occupied.some((p) => ctx.agent.canDonateGold(p)) ? 1 : 0;
    intent[IntentType.DonateTroops] = occupied.some((p) => ctx.agent.canDonateTroops(p))
      ? 1
      : 0;
    intent[IntentType.CancelAttack] = ctx.agent.outgoingAttacks().length > 0 ? 1 : 0;

    const target = out.target;
    target[0] = hasUnownedNeighbor ? 1 : 0;
    for (let i = 1; i < NUM_TARGET_SLOTS; i++) {
      target[i] = ctx.slots[i] !== null ? 1 : 0;
    }

    out.ratioBucket.fill(1);

    const site = out.buildSite;
    for (let i = 0; i < MAX_BUILD_SITES; i++) {
      site[i] = i < candidates.length && (buildable[i] || upgradeable[i]) ? 1 : 0;
    }
  }

  /**
   * Boat marginal: troops to send, a boatable cell, launch capacity, and at
   * least one probe tile the strict spawn query accepts. The spawn query
   * only takes water/shore targets while the execution resolves inland
   * targets through a shore BFS, so probes run shore-first and stop at the
   * first pass — a bounded approximation, with decode's full `canBuild`
   * check as the exact gate.
   */
  private boatUsable(ctx: CodecContext, boatTiles: TileRef[]): boolean {
    if (!ctx.agent.isAlive() || ctx.agent.troops() <= 0 || boatTiles.length === 0) {
      return false;
    }
    if (
      ctx.agent.unitCount(UnitTypeEnum.TransportShip) >= ctx.game.config().boatMaxNumber()
    ) {
      return false;
    }
    const game = ctx.game;
    const ordered = boatTiles
      .filter((t) => game.isShore(t))
      .concat(boatTiles.filter((t) => !game.isShore(t)));
    for (const tile of ordered) {
      if (ctx.agent.bestTransportShipSpawn(tile) !== false) return true;
    }
    return false;
  }

  /**
   * Spatial-cell mask: land not owned by the agent (boat destinations) and
   * water (warship patrol) stay open; own land, impassable terrain, and the
   * letterbox padding stay shut. One cheap center-tile probe per cell —
   * never a pathfinding query at this scale. Calls onBoatable once per
   * boatable cell so the caller learns launchability without a second pass.
   */
  private maskSpatial(
    ctx: CodecContext,
    mask: Uint8Array,
    onBoatable: (tile: TileRef) => void,
  ): void {
    const map = ctx.game.map();
    const w = map.width();
    const h = map.height();
    const agentSmallID = ctx.agent.smallID();
    for (let cell = 0; cell < NUM_SPATIAL_CELLS; cell++) {
      if (isPaddingCell(cell, w, h)) {
        mask[cell] = 0;
        continue;
      }
      const tile = cellCenterTile(cell, w, h);
      if (!map.isLand(tile)) {
        mask[cell] = 1;
        continue;
      }
      if (map.isImpassable(tile)) {
        mask[cell] = 0;
        continue;
      }
      const ownerID = map.ownerID(tile);
      if (ownerID === agentSmallID || ownerID === 0) {
        const open = ownerID === 0 ? 1 : 0;
        mask[cell] = open;
        if (open === 1) onBoatable(tile);
        continue;
      }
      const owner = ctx.bySmallID.get(ownerID);
      const open = owner !== undefined && ctx.agent.canAttackPlayer(owner) ? 1 : 0;
      mask[cell] = open;
      if (open === 1) onBoatable(tile);
    }
  }
}
