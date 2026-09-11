import { describe, expect, it } from "vitest";
import type { Turn } from "../../OpenFrontIO/src/core/Schemas";
import {
  ActionCodecImpl,
  IntentType,
  MAX_BUILD_SITES,
  NUM_INTENT_TYPES,
  NUM_RATIO_BUCKETS,
  NUM_SPATIAL_CELLS,
  NUM_TARGET_SLOTS,
  RATIO_BUCKETS,
  cellCenterTile,
  countBuildCandidates,
  createMaskBuffers,
  isPaddingCell,
} from "../src/ActionCodec";
import {
  NUM_SLOT_FIELDS,
  ObservationEncoder,
  PLAYER_TABLE_SIZE,
  RASTER_SIZE,
  SlotField,
} from "../src/ObservationEncoder";
import { TROOP_SCALE } from "../src/Reward";
import { TurnFeeder } from "../src/TurnFeeder";
import { assignTargetSlots, borderingSmallIDs } from "../src/TargetSlots";
import type { Action, MaskBuffers } from "../src/types";
import { AGENT_ID, buildPostSpawnGame, buildWorldGame } from "./helpers/buildWorldGame";
import { buildTestRunner } from "./helpers/buildTestRunner";

const codec = new ActionCodecImpl();

function baseAction(overrides: Partial<Action> = {}): Action {
  return {
    intentType: IntentType.Noop,
    target: 0,
    ratioBucket: 4, // 50%
    spatialCell: 0,
    buildSite: 0,
    ...overrides,
  };
}

function computeMasks(game: Parameters<ActionCodecImpl["masks"]>[0]): MaskBuffers {
  const masks = createMaskBuffers();
  codec.masks(game, AGENT_ID, masks);
  return masks;
}

function firstOpen(mask: Uint8Array): number {
  return mask.indexOf(1);
}

function expectBinary(mask: Uint8Array): void {
  for (const v of mask) expect(v === 0 || v === 1).toBe(true);
}

describe("ActionCodec masks", () => {
  it("writes correctly-sized 0/1 buffers with noop and ratio always open", async () => {
    const { game } = await buildPostSpawnGame("codec3-0");
    const masks = computeMasks(game);

    expect(masks.intentType.length).toBe(NUM_INTENT_TYPES);
    expect(masks.target.length).toBe(NUM_TARGET_SLOTS);
    expect(masks.ratioBucket.length).toBe(NUM_RATIO_BUCKETS);
    expect(masks.spatialCell.length).toBe(NUM_SPATIAL_CELLS);
    expect(masks.buildSite.length).toBe(MAX_BUILD_SITES);
    for (const buf of Object.values(masks)) expectBinary(buf);

    expect(masks.intentType[IntentType.Noop]).toBe(1);
    expect([...masks.ratioBucket].every((v) => v === 1)).toBe(true);
  });

  it("opens target slots exactly for living players, slot 0 for adjacent unowned land", async () => {
    const { game } = await buildPostSpawnGame("codec3-0");
    const agent = game.player(AGENT_ID);
    const masks = computeMasks(game);
    const slots = assignTargetSlots(game, agent, borderingSmallIDs(game, agent));

    for (let i = 1; i < NUM_TARGET_SLOTS; i++) {
      expect(masks.target[i]).toBe(slots[i] !== null ? 1 : 0);
    }

    // Independent expectation for slot 0: scan the agent's own border for
    // unowned neighboring land rather than reusing the codec's set.
    let adjacentUnowned = false;
    const map = game.map();
    for (const tile of agent.borderTiles()) {
      for (const n of map.neighbors(tile)) {
        if (map.isLand(n) && !map.hasOwner(n)) {
          adjacentUnowned = true;
          break;
        }
      }
      if (adjacentUnowned) break;
    }
    expect(masks.target[0]).toBe(adjacentUnowned ? 1 : 0);
  });

  it("masks the letterbox padding on a non-2:1 map", async () => {
    // Plains is 100x100: offsetX works out to 32, so the outer 64 columns
    // of the raster are padding with no map behind them.
    const { runner } = await buildTestRunner("codec-plains");
    const game = runner.game;
    const map = game.map();
    expect(map.width()).toBe(100);
    expect(map.height()).toBe(100);

    let turnNumber = 0;
    const runTick = (intents: Turn["intents"] = []): void => {
      runner.addTurn({ turnNumber: turnNumber++, intents });
      runner.executeNextTick();
    };
    runTick([
      { clientID: "client0", playerID: AGENT_ID, type: "spawn", tile: 5050 },
    ] as unknown as Turn["intents"]);
    let guard = 0;
    while (game.inSpawnPhase() && guard++ < 2000) runTick();

    const masks = computeMasks(game);
    const agentSmallID = game.player(AGENT_ID).smallID();
    let paddingCells = 0;
    let openCells = 0;
    let shutCells = 0;
    for (let cell = 0; cell < NUM_SPATIAL_CELLS; cell++) {
      if (isPaddingCell(cell, map.width(), map.height())) {
        paddingCells++;
        expect(masks.spatialCell[cell]).toBe(0);
      } else if (masks.spatialCell[cell] === 1) {
        openCells++;
      } else {
        // Reverse direction: on this bot-free map every shut non-padding
        // cell must be genuinely unusable — agent-owned or impassable land.
        shutCells++;
        const tile = cellCenterTile(cell, map.width(), map.height());
        const unusable =
          map.isImpassable(tile) || map.ownerID(tile) === agentSmallID;
        expect(unusable).toBe(true);
      }
    }
    expect(paddingCells).toBeGreaterThan(0);
    expect(openCells).toBeGreaterThan(0);
    expect(shutCells).toBeGreaterThan(0);
  });

  it("closes intent types with nothing pending on an idle game", async () => {
    const { game } = await buildPostSpawnGame("codec3-1");
    const masks = computeMasks(game);

    expect(masks.intentType[IntentType.CancelAttack]).toBe(0);
    expect(masks.intentType[IntentType.BreakAlliance]).toBe(0);
    expect(masks.intentType[IntentType.AllianceReject]).toBe(0);
    expect(masks.intentType[IntentType.MoveWarship]).toBe(0);
    // This seed was scanned for adjacent unowned land with troops in hand.
    expect(masks.intentType[IntentType.Attack]).toBe(1);
  });

  it("populates at most MAX_BUILD_SITES candidates and masks past the end", async () => {
    const { game } = await buildPostSpawnGame("codec3-0");
    const masks = computeMasks(game);
    const count = countBuildCandidates(game, AGENT_ID);

    expect(count).toBeGreaterThan(0);
    expect(count).toBeLessThanOrEqual(MAX_BUILD_SITES);
    for (let i = count; i < MAX_BUILD_SITES; i++) {
      expect(masks.buildSite[i]).toBe(0);
    }
  });

  it("shares its target-slot order with the observation encoder", async () => {
    const { game } = await buildPostSpawnGame("codec3-0");
    const agent = game.player(AGENT_ID);
    const encoder = new ObservationEncoder();
    const out = new Float32Array(encoder.size);
    encoder.encode(game, AGENT_ID, out);
    const table = out.subarray(RASTER_SIZE, RASTER_SIZE + PLAYER_TABLE_SIZE);
    const totalLand = game.map().numLandTiles();

    const slots = assignTargetSlots(game, agent, borderingSmallIDs(game, agent));
    for (let i = 1; i < NUM_TARGET_SLOTS; i++) {
      const base = i * NUM_SLOT_FIELDS;
      const p = slots[i];
      expect(table[base + SlotField.Valid]).toBe(p !== null ? 1 : 0);
      if (p !== null) {
        // Tiles pins the ordering; Troops pins the identity (a rotation bug
        // preserving tile order would still move troop counts).
        expect(table[base + SlotField.Tiles]).toBeCloseTo(p.numTilesOwned() / totalLand, 6);
        expect(table[base + SlotField.Troops]).toBeCloseTo(p.troops() / TROOP_SCALE, 6);
      }
    }
  });
});

describe("ActionCodec decode", () => {
  it("decodes a null-target attack with ratio-scaled troops", async () => {
    const { game } = await buildPostSpawnGame("codec3-0");
    const troopsBefore = game.player(AGENT_ID).troops();

    const intents = codec.decode(baseAction({ intentType: IntentType.Attack, target: 0 }), game, AGENT_ID);

    expect(intents.length).toBe(1);
    const intent = intents[0] as { type: string; targetID: null; troops: number };
    expect(intent.type).toBe("attack");
    expect(intent.targetID).toBeNull();
    expect(intent.troops).toBeCloseTo(troopsBefore * RATIO_BUCKETS[4], 6);
  });

  it("stamps every decoded intent with the agent's clientID", async () => {
    // Regression guard for the dropped-action defect: the engine resolves
    // each intent's acting player through playerByClientID, so an intent
    // without a clientID resolves to nobody and silently becomes a noop.
    // Every policy action trained before this line existed was dropped.
    const { game } = await buildPostSpawnGame("codec3-0");
    for (const action of [
      baseAction({ intentType: IntentType.Attack, target: 0 }),
      baseAction({ intentType: IntentType.TargetPlayer, target: 0 }),
    ]) {
      const intents = codec.decode(action, game, AGENT_ID);
      for (const intent of intents) {
        expect((intent as { clientID?: unknown }).clientID).toBe("client0");
      }
    }
  });

  it("executes a decoded attack: tiles actually grow", async () => {    // End-to-end proof the stamped intent resolves: feed it through the
    // real runner and watch territory expand. Pre-stamp this silently
    // no-op'd and tiles never moved.
    const { game, runner } = await buildPostSpawnGame("codec3-exec");
    const before = game.player(AGENT_ID).numTilesOwned();
    const intents = codec.decode(
      baseAction({ intentType: IntentType.Attack, target: 0 }),
      game,
      AGENT_ID,
    );
    expect(intents.length).toBe(1);
    const feeder = new TurnFeeder(runner);
    // Conquest takes a while (troops must march); poll until the frontier
    // moves, well past any single-tick timing. Pre-stamp this silently
    // no-op'd and tiles never moved at any horizon.
    let grown = game.player(AGENT_ID).numTilesOwned();
    for (let i = 0; i < 20 && grown <= before; i++) {
      feeder.step(10, intents as never);
      grown = game.player(AGENT_ID).numTilesOwned();
    }
    expect(grown).toBeGreaterThan(before);
  });

  it("resolves sampled open intents instead of voiding them", async () => {
    // Regression guard for the passive-bot defect: the target head is
    // occupancy-marginal, so sampled attack/donate/break combos usually
    // named the wrong slot and decoded to [], teaching the policy that
    // only diplomacy does anything. Fallback retargeting must hold the
    // void rate at zero for every intent whose mask guarantees a target.
    const { game, runner } = await buildPostSpawnGame("void-probe-1");
    const feeder = new TurnFeeder(runner);
    let rngState = 0x12345678;
    const rand = (): number => {
      rngState = (Math.imul(rngState, 1664525) + 1013904223) | 0;
      return (rngState >>> 0) / 4294967296;
    };
    const voids: Record<number, number> = {};
    for (let step = 0; step < 200; step++) {
      const masks = createMaskBuffers();
      codec.masks(game, AGENT_ID, masks);
      const pick = (mask: Uint8Array, size: number): number => {
        const open: number[] = [];
        for (let i = 0; i < size; i++) if (mask[i] === 1) open.push(i);
        return open.length > 0 ? open[Math.floor(rand() * open.length)] : 0;
      };
      const action: Action = {
        intentType: pick(masks.intentType, NUM_INTENT_TYPES),
        target: pick(masks.target, NUM_TARGET_SLOTS),
        ratioBucket: pick(masks.ratioBucket, NUM_RATIO_BUCKETS),
        spatialCell: pick(masks.spatialCell, NUM_SPATIAL_CELLS),
        buildSite: pick(masks.buildSite, MAX_BUILD_SITES),
      };
      const intents = codec.decode(action, game, AGENT_ID);
      if (intents.length === 0 && action.intentType !== IntentType.Noop) {
        voids[action.intentType] = (voids[action.intentType] ?? 0) + 1;
      }
      feeder.step(10, intents as never);
      if (!game.player(AGENT_ID).isAlive()) break;
    }
    expect(voids).toEqual({});
  });

  it("decodes a player-targeted attack to the slotted player's id", async () => {
    // This seed was scanned for a bordering attackable player at fixture
    // end; most early-game seeds border only unowned land so far.
    const { game } = await buildPostSpawnGame("codec3-19");
    const agent = game.player(AGENT_ID);
    const slots = assignTargetSlots(game, agent, borderingSmallIDs(game, agent));
    const slot = slots.findIndex(
      (p, i) => i > 0 && p !== null && agent.canAttackPlayer(p),
    );
    expect(slot).toBeGreaterThan(0);

    const intents = codec.decode(
      baseAction({ intentType: IntentType.Attack, target: slot }),
      game,
      AGENT_ID,
    );

    expect(intents.length).toBe(1);
    const intent = intents[0] as { type: string; targetID: string };
    expect(intent.type).toBe("attack");
    expect(intent.targetID).toBe(slots[slot]!.id());
  });

  it("drops illegal combinations as [] instead of emitting", async () => {
    const { game } = await buildWorldGame("codec3-1");
    const agent = game.player(AGENT_ID);
    const slots = assignTargetSlots(game, agent, borderingSmallIDs(game, agent));
    const occupied = slots.findIndex((p) => p !== null);

    // Slot 0 is terra nullius: targetPlayer retargets to a real player
    // through fallback (or voids if nobody is targetable).
    const retargeted = codec.decode(baseAction({ intentType: IntentType.TargetPlayer, target: 0 }), game, AGENT_ID);
    expect(retargeted.length).toBeLessThanOrEqual(1);
    if (retargeted.length === 1) {
      const intent = retargeted[0] as { type: string; target: string };
      expect(intent.type).toBe("targetPlayer");
      expect(game.player(intent.target).isAlive()).toBe(true);
    }
    // Fresh game: no allies, no pending requests, no structures, no attacks.
    expect(
      codec.decode(baseAction({ intentType: IntentType.BreakAlliance, target: occupied }), game, AGENT_ID),
    ).toEqual([]);
    expect(
      codec.decode(baseAction({ intentType: IntentType.AllianceReject, target: occupied }), game, AGENT_ID),
    ).toEqual([]);
    expect(
      codec.decode(baseAction({ intentType: IntentType.DonateGold, target: occupied }), game, AGENT_ID),
    ).toEqual([]);
    expect(codec.decode(baseAction({ intentType: IntentType.CancelAttack }), game, AGENT_ID)).toEqual([]);
    expect(codec.decode(baseAction({ intentType: IntentType.UpgradeStructure }), game, AGENT_ID)).toEqual([]);
    expect(codec.decode(baseAction({ intentType: IntentType.MoveWarship }), game, AGENT_ID)).toEqual([]);
    expect(codec.decode(baseAction({ intentType: IntentType.Noop }), game, AGENT_ID)).toEqual([]);
  });

  it("throws RangeError on out-of-range head values", async () => {
    const { game } = await buildWorldGame("codec3-1");
    expect(() => codec.decode(baseAction({ intentType: 99 }), game, AGENT_ID)).toThrow(RangeError);
    expect(() => codec.decode(baseAction({ target: 99 }), game, AGENT_ID)).toThrow(RangeError);
    expect(() => codec.decode(baseAction({ ratioBucket: 99 }), game, AGENT_ID)).toThrow(RangeError);
    expect(() => codec.decode(baseAction({ spatialCell: 99999 }), game, AGENT_ID)).toThrow(RangeError);
    expect(() => codec.decode(baseAction({ buildSite: 99 }), game, AGENT_ID)).toThrow(RangeError);
  });

  it("decodes alliance requests and embargo toggles with correct shapes", async () => {
    const { game } = await buildPostSpawnGame("codec3-2");
    const agent = game.player(AGENT_ID);
    const slots = assignTargetSlots(game, agent, borderingSmallIDs(game, agent));
    const slot = slots.findIndex((p, i) => i > 0 && p !== null && agent.canSendAllianceRequest(p));
    expect(slot).toBeGreaterThan(0);

    const request = codec.decode(
      baseAction({ intentType: IntentType.AllianceRequest, target: slot }),
      game,
      AGENT_ID,
    );
    expect(request).toEqual([{ type: "allianceRequest", recipient: slots[slot]!.id(), clientID: "client0" }]);

    const embargo = codec.decode(
      baseAction({ intentType: IntentType.Embargo, target: slot }),
      game,
      AGENT_ID,
    );
    // No embargo yet against a fresh counterpart: the codec opens with start.
    expect(embargo).toEqual([{ type: "embargo", targetID: slots[slot]!.id(), action: "start", clientID: "client0" }]);
  });

  it("builds with bonus gold and the engine accepts the intent", async () => {
    const { game, feeder } = await buildPostSpawnGame("codec3-2");
    const agent = game.player(AGENT_ID);
    agent.addGold(1_000_000_000n);

    const masks = computeMasks(game);
    const site = firstOpen(masks.buildSite);
    expect(site).toBeGreaterThanOrEqual(0);

    const intents = codec.decode(
      baseAction({ intentType: IntentType.BuildUnit, buildSite: site }),
      game,
      AGENT_ID,
    );
    expect(intents.length).toBe(1);
    const intent = intents[0] as { type: string; unit: string; tile: number };
    expect(intent.type).toBe("build_unit");
    expect(game.map().isValidRef(intent.tile)).toBe(true);

    // The engine takes it without complaint (the fixture's callback throws
    // on any ErrorUpdate, so surviving the tick is the assertion).
    feeder.step(1, intents as unknown as Turn["intents"]);
  });
});

describe("ActionCodec soundness (spec §8)", () => {
  it("every unmasked intent executes a live turn without throwing", async () => {
    // Scanned seed: borders an attackable player at fixture end (rare this
    // early — most seeds border only unowned land so far).
    const { game, feeder } = await buildPostSpawnGame("codec3-23");
    const agent = game.player(AGENT_ID);
    agent.addGold(1_000_000_000n);
    const slots = assignTargetSlots(game, agent, borderingSmallIDs(game, agent));

    const attackTarget = slots.findIndex(
      (p, i) => i > 0 && p !== null && agent.canAttackPlayer(p),
    );
    const allyTarget = slots.findIndex((p, i) => i > 0 && p !== null && agent.canTarget(p));
    const masks = computeMasks(game);
    const site = firstOpen(masks.buildSite);

    // The spatial mask is boat/warship-marginal: the first open cell may be
    // water (a warship patrol spot), which boat decode correctly refuses.
    // Search open cells for one that actually decodes, bounded so a
    // landlocked map stays fast.
    let boatCell = -1;
    if (masks.intentType[IntentType.Boat] === 1) {
      let tried = 0;
      for (let cell = 0; cell < NUM_SPATIAL_CELLS && tried < 64 && boatCell < 0; cell++) {
        if (masks.spatialCell[cell] !== 1) continue;
        tried++;
        const probe = baseAction({ intentType: IntentType.Boat, spatialCell: cell });
        if (codec.decode(probe, game, AGENT_ID).length === 1) boatCell = cell;
      }
    }

    const cases: { type: IntentType; action: Action }[] = [
      { type: IntentType.Attack, action: baseAction({ intentType: IntentType.Attack, target: 0 }) },
      {
        type: IntentType.Attack,
        action: baseAction({ intentType: IntentType.Attack, target: attackTarget }),
      },
      {
        type: IntentType.TargetPlayer,
        action: baseAction({ intentType: IntentType.TargetPlayer, target: allyTarget }),
      },
      {
        type: IntentType.AllianceRequest,
        action: baseAction({ intentType: IntentType.AllianceRequest, target: allyTarget }),
      },
      {
        type: IntentType.Embargo,
        action: baseAction({ intentType: IntentType.Embargo, target: allyTarget }),
      },
    ];
    if (boatCell >= 0) {
      cases.push({
        type: IntentType.Boat,
        action: baseAction({ intentType: IntentType.Boat, spatialCell: boatCell }),
      });
    }
    if (site >= 0 && masks.intentType[IntentType.BuildUnit] === 1) {
      cases.push({
        type: IntentType.BuildUnit,
        action: baseAction({ intentType: IntentType.BuildUnit, buildSite: site }),
      });
    }

    const batch: Turn["intents"] = [];
    for (const { type, action } of cases) {
      const intents = codec.decode(action, game, AGENT_ID);
      // Favorable heads were chosen above: an unmasked intent type must
      // decode to exactly one intent here, never [] and never throw.
      expect(intents.length).toBe(1);
      expect((intents[0] as { type: string }).type).toBe(
        intentTypeName(type),
      );
      batch.push(...(intents as unknown as Turn["intents"]));
    }
    feeder.step(1, batch);

    // The attack above is now in flight: cancel_attack must decode and the
    // mask must agree it is available.
    const outgoing = agent.outgoingAttacks();
    const cancel = codec.decode(baseAction({ intentType: IntentType.CancelAttack }), game, AGENT_ID);
    const after = computeMasks(game);
    if (outgoing.length > 0) {
      expect(cancel.length).toBe(1);
      expect(after.intentType[IntentType.CancelAttack]).toBe(1);
      feeder.step(1, cancel as unknown as Turn["intents"]);
    } else {
      expect(cancel).toEqual([]);
      expect(after.intentType[IntentType.CancelAttack]).toBe(0);
    }
  });
});

describe("ActionCodec determinism", () => {
  it("produces identical masks and decodes for identical seeds", async () => {
    const first = await buildPostSpawnGame("codec3-16");
    const second = await buildPostSpawnGame("codec3-16");
    const a = computeMasks(first.game);
    const b = computeMasks(second.game);
    for (const key of Object.keys(a) as (keyof MaskBuffers)[]) {
      expect([...a[key]]).toEqual([...b[key]]);
    }
    // Same state in, same intents out — including troop-scaled amounts.
    const probe = baseAction({ intentType: IntentType.Attack, target: 0 });
    expect(codec.decode(probe, second.game, AGENT_ID)).toEqual(
      codec.decode(probe, first.game, AGENT_ID),
    );
  });
});

function intentTypeName(type: IntentType): string {
  switch (type) {
    case IntentType.Attack:
      return "attack";
    case IntentType.Boat:
      return "boat";
    case IntentType.BuildUnit:
      return "build_unit";
    case IntentType.UpgradeStructure:
      return "upgrade_structure";
    case IntentType.MoveWarship:
      return "move_warship";
    case IntentType.TargetPlayer:
      return "targetPlayer";
    case IntentType.AllianceRequest:
      return "allianceRequest";
    case IntentType.AllianceReject:
      return "allianceReject";
    case IntentType.BreakAlliance:
      return "breakAlliance";
    case IntentType.Embargo:
      return "embargo";
    case IntentType.DonateGold:
      return "donate_gold";
    case IntentType.DonateTroops:
      return "donate_troops";
    case IntentType.CancelAttack:
      return "cancel_attack";
    case IntentType.Noop:
      return "noop";
  }
}
