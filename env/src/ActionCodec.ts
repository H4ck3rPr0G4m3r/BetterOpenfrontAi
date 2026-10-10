import type { Game, Player, PlayerID } from "../../OpenFrontIO/src/core/game/Game";
import { UnitType } from "../../OpenFrontIO/src/core/game/Game";
import type { TileRef } from "../../OpenFrontIO/src/core/game/GameMap";
import type { Intent } from "../../OpenFrontIO/src/core/Schemas";
import { BehaviorConfig, NormalizedBehavior, normalizeBehavior } from "./BehaviorConfig";
import { buildCandidates } from "./BuildSites";
import { cellGeometry, cellRect } from "./Geometry";
import {
  IntentType,
  MAX_BUILD_SITES,
  Masks,
  NUM_INTENTS,
  NUM_RATIO_BUCKETS,
  NUM_SLOTS,
  RASTER_CELLS,
  RATIO_BUCKETS,
  siteRow,
  spatialRow,
  targetRow,
} from "./Layout";
import { assignTargetSlots, borderingSmallIDs, slotPlayer } from "./TargetSlots";
import type { Action } from "./types";

interface Context {
  game: Game;
  agent: Player;
  agentID: PlayerID;
  /** Neighbouring smallIDs; 0 means unowned land touches our border. */
  bordering: Set<number>;
  slots: (Player | null)[];
  bySmallID: Map<number, Player>;
}

function context(game: Game, playerID: PlayerID): Context {
  const agent = game.player(playerID);
  const bordering = borderingSmallIDs(game, agent);
  const bySmallID = new Map<number, Player>();
  for (const p of game.players()) bySmallID.set(p.smallID(), p);
  return { game, agent, agentID: playerID, bordering, slots: assignTargetSlots(game, agent, bordering), bySmallID };
}

/** Intents whose target head picks a player slot. */
const TARGETED_INTENTS = [
  IntentType.Attack,
  IntentType.TargetPlayer,
  IntentType.AllianceRequest,
  IntentType.AllianceReject,
  IntentType.BreakAlliance,
  IntentType.Embargo,
  IntentType.DonateGold,
  IntentType.DonateTroops,
];

function checkAction(a: Action): void {
  const ranges: [keyof Action, number][] = [
    ["intentType", NUM_INTENTS],
    ["target", NUM_SLOTS],
    ["ratioBucket", NUM_RATIO_BUCKETS],
    ["spatialCell", RASTER_CELLS],
    ["buildSite", MAX_BUILD_SITES],
  ];
  for (const [key, size] of ranges) {
    const v = a[key];
    if (!Number.isInteger(v) || v < 0 || v >= size) throw new RangeError(`${key} ${v} out of range`);
  }
}

/**
 * Action codec, schema v2: intent-conditioned masks plus strict decoding.
 *
 * masks() writes, per intent, exactly which target slots / cells / sites
 * that intent can use (Layout.ts). The policy samples the intent first and
 * then each head with that intent's row, so a sampled action is legal by
 * construction and decode() never needs to guess. v1 used marginal masks
 * and "retargeted" any invalid attack onto the first attackable slot —
 * which, by slot order, is the largest bordering neighbour. Random and
 * early policies were systematically picking fights with the strongest
 * player next door (and every attack on a nation costs -70 relation).
 * That fallback is gone: an invalid combination decodes to [].
 *
 * Boat landings are the one approximate mask (a cell probe, not a
 * pathfind); decode() runs the engine's exact launch check.
 */
export class ActionCodecImpl {
  private readonly behavior: NormalizedBehavior;

  constructor(behavior?: BehaviorConfig) {
    this.behavior = normalizeBehavior(behavior);
  }

  decode(action: Action, game: Game, playerID: PlayerID): Intent[] {
    checkAction(action);
    const intent = action.intentType;
    if (intent === IntentType.Noop || this.behavior.disabledIntents.has(intent)) return [];
    const ctx = context(game, playerID);
    if (!ctx.agent.isAlive()) return [];
    const intents = this.decodeInner(action, ctx);
    // The engine resolves the acting player through the intent's clientID;
    // an unstamped intent silently does nothing.
    const clientID = ctx.agent.clientID();
    if (clientID === null) throw new Error("decode() needs a seated human player with a clientID");
    return intents.map((i) => ({ ...i, clientID }));
  }

  private decodeInner(action: Action, ctx: Context): Intent[] {
    const ratio = RATIO_BUCKETS[action.ratioBucket];
    const intent = action.intentType;
    switch (intent) {
      case IntentType.Boat:
        return this.decodeBoat(ctx, action.spatialCell, ratio);
      case IntentType.BuildUnit:
      case IntentType.UpgradeStructure:
        return this.decodeBuild(ctx, action.buildSite, intent === IntentType.UpgradeStructure);
      case IntentType.MoveWarship:
        return this.decodeMoveWarship(ctx, action.spatialCell);
      case IntentType.CancelAttack: {
        const attacks = ctx.agent.outgoingAttacks();
        return attacks.length > 0 ? [{ type: "cancel_attack", attackID: attacks[0].id() } as Intent] : [];
      }
      default:
        break;
    }
    if (!TARGETED_INTENTS.includes(intent)) throw new RangeError(`unknown intent ${intent}`);
    if (!this.targetValid(ctx, intent, action.target, ratio)) return [];
    const target = action.target === 0 ? null : slotPlayer(ctx.slots, ctx.agentID, ctx.game, action.target);
    switch (intent) {
      case IntentType.Attack:
        return [{ type: "attack", targetID: target?.id() ?? null, troops: ratio * ctx.agent.troops() } as Intent];
      case IntentType.TargetPlayer:
        return [{ type: "targetPlayer", target: target!.id() } as Intent];
      case IntentType.AllianceRequest:
        return [{ type: "allianceRequest", recipient: target!.id() } as Intent];
      case IntentType.AllianceReject:
        return [{ type: "allianceReject", requestor: target!.id() } as Intent];
      case IntentType.BreakAlliance:
        return [{ type: "breakAlliance", recipient: target!.id() } as Intent];
      case IntentType.Embargo: {
        const toggle = ctx.agent.hasEmbargoAgainst(target!) ? "stop" : "start";
        return [{ type: "embargo", targetID: target!.id(), action: toggle } as Intent];
      }
      case IntentType.DonateGold:
        return [{ type: "donate_gold", recipient: target!.id(), gold: Number(goldFraction(ctx.agent.gold(), ratio)) } as Intent];
      case IntentType.DonateTroops:
        return [{ type: "donate_troops", recipient: target!.id(), troops: ratio * ctx.agent.troops() } as Intent];
      default:
        return [];
    }
  }

  /** Whether `intent` can legally use target `slot` in this state. */
  private targetValid(ctx: Context, intent: number, slot: number, ratio = 1): boolean {
    const agent = ctx.agent;
    if (slot === 0) {
      return intent === IntentType.Attack && agent.troops() > 0 && ctx.bordering.has(0);
    }
    const p = slotPlayer(ctx.slots, ctx.agentID, ctx.game, slot);
    if (p === null) return false;
    switch (intent) {
      case IntentType.Attack:
        return agent.troops() > 0 && ctx.bordering.has(p.smallID()) && agent.canAttackPlayer(p);
      case IntentType.TargetPlayer:
        return agent.canTarget(p);
      case IntentType.AllianceRequest:
        return agent.canSendAllianceRequest(p);
      case IntentType.AllianceReject:
        return agent.incomingAllianceRequests().some((r) => r.requestor() === p);
      case IntentType.BreakAlliance:
        return agent.allianceWith(p) !== null;
      case IntentType.Embargo:
        return true;
      case IntentType.DonateGold:
        return agent.canDonateGold(p) && goldFraction(agent.gold(), ratio) > 0n;
      case IntentType.DonateTroops:
        return agent.canDonateTroops(p) && agent.troops() > 0;
      default:
        return false;
    }
  }

  private decodeBoat(ctx: Context, cell: number, ratio: number): Intent[] {
    const troops = ratio * ctx.agent.troops();
    if (troops <= 0) return [];
    const dst = resolveBoatDst(ctx, cell);
    const map = ctx.game.map();
    if (dst < 0 || !map.isLand(dst) || map.isImpassable(dst)) return [];
    const owner = map.ownerID(dst);
    if (owner === ctx.agent.smallID()) return [];
    if (owner !== 0) {
      const p = ctx.bySmallID.get(owner);
      if (p === undefined || !ctx.agent.canAttackPlayer(p)) return [];
    }
    // Same predicate TransportShipExecution uses: boat cap, gold, reachability.
    if (ctx.agent.canBuild(UnitType.TransportShip, dst) === false) return [];
    return [{ type: "boat", troops, dst } as Intent];
  }

  private decodeBuild(ctx: Context, site: number, upgrade: boolean): Intent[] {
    const candidates = buildCandidates(ctx.game, ctx.agent);
    if (site >= candidates.length) return [];
    const c = candidates[site];
    if (this.behavior.disabledBuilds.has(c.type as string)) return [];
    const q = ctx.agent.buildableUnits(c.tile, [c.type])[0];
    if (upgrade) {
      return q.canUpgrade === false ? [] : [{ type: "upgrade_structure", unit: c.type, unitId: q.canUpgrade } as Intent];
    }
    // The engine-resolved spawn tile, not the candidate tile.
    return q.canBuild === false ? [] : [{ type: "build_unit", unit: c.type, tile: q.canBuild } as Intent];
  }

  private decodeMoveWarship(ctx: Context, cell: number): Intent[] {
    const warships = ctx.agent.units(UnitType.Warship).filter((u) => u.isActive());
    if (warships.length === 0) return [];
    const dst = resolveWaterDst(ctx, cell);
    const map = ctx.game.map();
    if (dst < 0 || map.isLand(dst)) return [];
    // Only ships sharing the destination's water can sail there.
    const component = ctx.game.getWaterComponent(dst);
    const same = component === null ? [] : warships.filter((w) => ctx.game.hasWaterComponent(w.tile(), component));
    if (same.length === 0) return [];
    return [{ type: "move_warship", unitIds: same.map((w) => w.id()), tile: dst } as Intent];
  }

  masks(game: Game, playerID: PlayerID, out: Masks): void {
    out.flat.fill(0);
    out.intent[IntentType.Noop] = 1;
    const ctx = context(game, playerID);
    const agent = ctx.agent;
    if (!agent.isAlive()) return;
    const disabled = this.behavior.disabledIntents;
    const open = (intent: number) => !disabled.has(intent);

    // Target rows, one per targeted intent.
    for (const intent of TARGETED_INTENTS) {
      if (!open(intent)) continue;
      const row = targetRow(out, intent);
      let any = 0;
      for (let slot = 0; slot < NUM_SLOTS; slot++) {
        if (this.targetValid(ctx, intent, slot)) {
          row[slot] = 1;
          any = 1;
        }
      }
      out.intent[intent] = any;
    }
    out.ratio.fill(1);

    // Spatial rows: boat landings (row 0), warship water (row 1).
    const map = game.map();
    const geometry = cellGeometry(map.width(), map.height());
    const boatRow = spatialRow(out, 0);
    const waterRow = spatialRow(out, 1);
    const hasWarships = agent.units(UnitType.Warship).some((u) => u.isActive());
    const agentSmall = agent.smallID();
    const boatProbes: TileRef[] = [];
    for (let cell = 0; cell < RASTER_CELLS; cell++) {
      const tile = geometry.centerTile[cell];
      if (tile < 0) continue;
      if (!map.isLand(tile)) {
        if (hasWarships) waterRow[cell] = 1;
        continue;
      }
      if (map.isImpassable(tile)) continue;
      const owner = map.ownerID(tile);
      if (owner === agentSmall) continue;
      if (owner !== 0) {
        const p = ctx.bySmallID.get(owner);
        if (p === undefined || !agent.canAttackPlayer(p)) continue;
      }
      boatRow[cell] = 1;
      if (boatProbes.length < 8) boatProbes.push(tile);
    }
    out.intent[IntentType.Boat] = open(IntentType.Boat) && this.boatUsable(ctx, boatProbes) ? 1 : 0;
    if (out.intent[IntentType.Boat] === 0) boatRow.fill(0);
    out.intent[IntentType.MoveWarship] = open(IntentType.MoveWarship) && hasWarships ? 1 : 0;
    if (out.intent[IntentType.MoveWarship] === 0) waterRow.fill(0);

    // Build sites: one single-type engine query per candidate.
    const buildRow = siteRow(out, 0);
    const upgradeRow = siteRow(out, 1);
    const candidates = buildCandidates(game, agent);
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      if (this.behavior.disabledBuilds.has(c.type as string)) continue;
      const q = agent.buildableUnits(c.tile, [c.type])[0];
      if (q.canBuild !== false) buildRow[i] = 1;
      if (q.canUpgrade !== false) upgradeRow[i] = 1;
    }
    out.intent[IntentType.BuildUnit] = open(IntentType.BuildUnit) && buildRow.some((v) => v === 1) ? 1 : 0;
    out.intent[IntentType.UpgradeStructure] = open(IntentType.UpgradeStructure) && upgradeRow.some((v) => v === 1) ? 1 : 0;
    if (!out.intent[IntentType.BuildUnit]) buildRow.fill(0);
    if (!out.intent[IntentType.UpgradeStructure]) upgradeRow.fill(0);

    out.intent[IntentType.CancelAttack] = open(IntentType.CancelAttack) && agent.outgoingAttacks().length > 0 ? 1 : 0;
  }

  /**
   * Boat marginal: troops, launch capacity, and a probe tile the engine's
   * spawn query accepts (shore probes first). Bounded approximation; decode
   * runs the exact check.
   */
  private boatUsable(ctx: Context, probes: TileRef[]): boolean {
    const agent = ctx.agent;
    if (agent.troops() <= 0 || probes.length === 0) return false;
    if (agent.unitCount(UnitType.TransportShip) >= ctx.game.config().boatMaxNumber()) return false;
    const ordered = probes.filter((t) => ctx.game.isShore(t)).concat(probes.filter((t) => !ctx.game.isShore(t)));
    return ordered.some((t) => agent.bestTransportShipSpawn(t) !== false);
  }
}

/** Exact bigint math for gold fractions (gold exceeds float precision). */
function goldFraction(gold: bigint, ratio: number): bigint {
  return (gold * BigInt(Math.round(ratio * 100))) / 100n;
}

/** Boat landing inside a cell: enemy land first, then unowned land. */
function resolveBoatDst(ctx: Context, cell: number): TileRef {
  const map = ctx.game.map();
  const r = cellRect(cell, map.width(), map.height());
  let unowned: TileRef = -1;
  for (let y = r.y0; y < r.y1; y++) {
    for (let x = r.x0; x < r.x1; x++) {
      const tile = y * map.width() + x;
      if (!map.isLand(tile) || map.isImpassable(tile)) continue;
      const owner = map.ownerID(tile);
      if (owner === ctx.agent.smallID()) continue;
      if (owner === 0) {
        if (unowned < 0) unowned = tile;
        continue;
      }
      return tile;
    }
  }
  return unowned;
}

/** Warship patrol tile inside a cell: its first water tile. */
function resolveWaterDst(ctx: Context, cell: number): TileRef {
  const map = ctx.game.map();
  const r = cellRect(cell, map.width(), map.height());
  for (let y = r.y0; y < r.y1; y++) {
    for (let x = r.x0; x < r.x1; x++) {
      const tile = y * map.width() + x;
      if (!map.isLand(tile)) return tile;
    }
  }
  return -1;
}
