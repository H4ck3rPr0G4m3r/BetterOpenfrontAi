/**
 * In-page agent for real openfront.io games, bundled into
 * env/dist/page_agent.js (npm run build:page) and injected by
 * train/openfront/lobby.py through Playwright.
 *
 * It exposes `window.__openfrontAgent` with:
 *   status()   cheap snapshot: in game? spawned? alive? ticks, my stats
 *   observe()  the training observation + masks for my player, base64
 *   act(a)     decode a 5-head action with the training codec and send it
 *   spawn()    pick a spawn tile (training heuristic) and send it
 *
 * State comes from the client's own GameView (`build-menu.game`, the hook
 * the upstream test driver documents) and intents leave through the
 * client's own EventBus handlers in Transport — the same path a human's
 * click takes, so the server sees ordinary player intents. Nothing here
 * hides that a bot is playing; it plays as a normal, visible player.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { ActionCodecImpl } from "../ActionCodec";
import { createMasks } from "../Layout";
import { createSeatObservation, ObservationEncoder } from "../ObservationEncoder";
import type { Action } from "../types";
import { QueryCache, scanOwnership, ViewGame } from "./GameViewAdapter";

const MAX_PASSES = 3;

function hud(): any {
  return document.querySelector("build-menu") as any;
}
function gameView(): any {
  return hud()?.game ?? null;
}

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/** Transport handler method names per intent type. */
const HANDLERS: Record<string, string> = {
  spawn: "onSendSpawnIntentEvent",
  attack: "onSendAttackIntent",
  boat: "onSendBoatAttackIntent",
  build_unit: "onBuildUnitIntent",
  upgrade_structure: "onSendUpgradeStructureIntent",
  move_warship: "onMoveWarshipEvent",
  targetPlayer: "onSendTargetPlayerIntent",
  allianceRequest: "onSendAllianceRequest",
  allianceReject: "onAllianceRejectUIEvent",
  breakAlliance: "onBreakAllianceRequestUIEvent",
  embargo: "onSendEmbargoIntent",
  donate_gold: "onSendDonateGoldIntent",
  donate_troops: "onSendDonateTroopIntent",
  cancel_attack: "onCancelAttackIntentEvent",
};

/** The event object each Transport handler reads, built from an intent. */
function eventFor(intent: any): any {
  const ref = (id: string) => ({ id: () => id });
  switch (intent.type) {
    case "spawn":
      return { tile: intent.tile };
    case "attack":
      return { targetID: intent.targetID, troops: intent.troops };
    case "boat":
      return { troops: intent.troops, dst: intent.dst };
    case "build_unit":
      return { unit: intent.unit, tile: intent.tile };
    case "upgrade_structure":
      return { unitType: intent.unit, unitId: intent.unitId };
    case "move_warship":
      return { unitIds: intent.unitIds, tile: intent.tile };
    case "targetPlayer":
      return { targetID: intent.target };
    case "allianceRequest":
      return { recipient: ref(intent.recipient) };
    case "allianceReject":
      return { requestor: ref(intent.requestor) };
    case "breakAlliance":
      return { recipient: ref(intent.recipient) };
    case "embargo":
      return { target: ref(intent.targetID), action: intent.action };
    case "donate_gold":
      return { recipient: ref(intent.recipient), gold: intent.gold };
    case "donate_troops":
      return { recipient: ref(intent.recipient), troops: intent.troops };
    case "cancel_attack":
      return { attackID: intent.attackID };
    default:
      throw new Error(`no handler mapping for intent ${intent.type}`);
  }
}

class PageAgent {
  private readonly encoder = new ObservationEncoder(6000);
  private readonly codec = new ActionCodecImpl();
  private handlers = new Map<string, (event: unknown) => void>();
  private last: ViewGame | null = null;
  readonly version = 2;

  status() {
    const v = gameView();
    if (v === null) return { inGame: false, url: location.href };
    const me = v.myPlayer();
    return {
      inGame: true,
      url: location.href,
      gameID: v.gameID?.(),
      ticks: v.ticks(),
      inSpawnPhase: v.inSpawnPhase(),
      gameOver: v.gameOver?.() ?? false,
      hasPlayer: me !== null,
      spawned: me?.hasSpawned?.() ?? (me?.numTilesOwned() ?? 0) > 0,
      alive: me?.isAlive() ?? false,
      me: me === null ? null : { name: me.name(), tiles: me.numTilesOwned(), troops: me.troops(), gold: Number(me.gold()) },
      players: v.players().filter((p: any) => p.isAlive()).length,
    };
  }

  private async world(): Promise<ViewGame> {
    const v = gameView();
    if (v === null) throw new Error("not in a game");
    const me = v.myPlayer();
    if (me === null) throw new Error("no player for this client");
    const border = me.isAlive() ? (await me.borderTiles()).borderTiles : new Set<number>();
    return new ViewGame(v, new QueryCache(), scanOwnership(v, me.smallID()), border);
  }

  /** The training observation and masks for my player. */
  async observe() {
    const game = await this.world();
    const id = game.view.myPlayer().id();
    const obs = createSeatObservation();
    const masks = createMasks();
    for (let pass = 0; pass < MAX_PASSES; pass++) {
      this.encoder.encodeOne(game as any, id, obs);
      this.codec.masks(game as any, id, masks);
      if (game.cache.missing === 0) break;
      await game.cache.resolve();
    }
    this.last = game;
    return {
      tick: game.ticks(),
      raster: toBase64(obs.raster),
      features: Array.from(obs.features),
      masks: toBase64(masks.flat),
    };
  }

  /** Decode a 5-head action with the training codec and send it. */
  async act(action: Action) {
    const game = this.last ?? (await this.world());
    const id = game.view.myPlayer().id();
    let intents: any[] = [];
    for (let pass = 0; pass < MAX_PASSES; pass++) {
      intents = this.codec.decode(action, game as any, id) as any[];
      if (game.cache.missing === 0) break;
      await game.cache.resolve();
    }
    for (const intent of intents) this.emit(intent);
    return { sent: intents.map((i) => i.type), intents: JSON.parse(JSON.stringify(intents, (_k, v) => (typeof v === "bigint" ? Number(v) : v))) };
  }

  /** Choose a spawn tile like training does (free land, room to grow, away from others). */
  spawn(seed = Date.now()) {
    const v = gameView();
    if (v === null) throw new Error("not in a game");
    const w = v.width();
    const h = v.height();
    let state = seed >>> 0 || 1;
    const rand = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 0x100000000);
    const owned: number[] = [];
    const buf: Uint16Array = v.tileStateBuffer();
    for (let t = 0; t < buf.length; t += 97) if ((buf[t] & 0xfff) !== 0) owned.push(t);
    let best = -1;
    let bestScore = -1;
    for (let i = 0; i < 6000; i++) {
      const t = Math.floor(rand() * w * h);
      if (!v.isLand(t) || v.hasOwner(t) || v.isImpassable?.(t)) continue;
      if (!v.neighbors(t).every((n: number) => v.isLand(n) && !v.hasOwner(n))) continue;
      let nearest = Infinity;
      for (const o of owned) nearest = Math.min(nearest, Math.hypot((o % w) - (t % w), Math.floor(o / w) - Math.floor(t / w)));
      const score = Math.min(nearest, 200);
      if (score > bestScore) {
        bestScore = score;
        best = t;
      }
    }
    if (best < 0) throw new Error("no free spawn tile found");
    this.emit({ type: "spawn", tile: best });
    return { tile: best, x: best % w, y: Math.floor(best / w) };
  }

  private emit(intent: any): void {
    const handler = this.handler(HANDLERS[intent.type]);
    handler(eventFor(intent));
  }

  /** Find a Transport listener by the method it forwards to. Method names
   *  survive the client's minification (properties are not mangled). */
  private handler(name: string): (event: unknown) => void {
    const cached = this.handlers.get(name);
    if (cached !== undefined) return cached;
    const bus = hud()?.eventBus;
    const listeners: Map<unknown, ((e: unknown) => void)[]> | undefined = bus?.listeners;
    if (!listeners) throw new Error("client event bus not reachable");
    for (const callbacks of listeners.values()) {
      for (const cb of callbacks) {
        if (String(cb).includes(`.${name}(`)) {
          this.handlers.set(name, cb);
          return cb;
        }
      }
    }
    throw new Error(`client has no ${name} handler (client version changed?)`);
  }
}

(window as any).__openfrontAgent = new PageAgent();
