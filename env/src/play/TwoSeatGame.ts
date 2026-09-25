import { Config } from "../../../OpenFrontIO/src/core/configuration/Config";
import { Executor } from "../../../OpenFrontIO/src/core/execution/ExecutionManager";
import {
  Difficulty,
  GameMapSize,
  GameMode,
  GameType,
  PlayerID,
  PlayerInfo,
  PlayerType,
} from "../../../OpenFrontIO/src/core/game/Game";
import {
  ErrorUpdate,
  GameUpdateType,
  GameUpdateViewData,
} from "../../../OpenFrontIO/src/core/game/GameUpdates";
import { createGame } from "../../../OpenFrontIO/src/core/game/GameImpl";
import { createNationsForGame } from "../../../OpenFrontIO/src/core/game/NationCreation";
import { loadFreshTerrainMap } from "../Terrain";
import { GameRunner } from "../../../OpenFrontIO/src/core/GameRunner";
import { PseudoRandom } from "../../../OpenFrontIO/src/core/PseudoRandom";
import { GameConfig, GameStartInfo } from "../../../OpenFrontIO/src/core/Schemas";
import { simpleHash } from "../../../OpenFrontIO/src/core/Util";
import { Reward } from "../Reward";
import { TurnFeeder } from "../TurnFeeder";
import { ActionCodecImpl, createMaskBuffers } from "../ActionCodec";
import { ObservationEncoder } from "../ObservationEncoder";
import type {
  Action,
  ActionCodec,
  EnvConfig,
  MaskBuffers,
  Observation,
  ObservationEncoder as ObservationEncoderFn,
} from "../types";

export const AI_CLIENT_ID = "client0";
export const HUMAN_CLIENT_ID = "client1";

export type VersusTerminal = "ai_win" | "human_win" | "draw" | "tick_cap" | "sim_error" | null;

export interface VersusStep {
  aiObs: Observation;
  humanObs: Observation;
  aiMasks: MaskBuffers;
  humanMasks: MaskBuffers;
  aiReward: number;
  humanReward: number;
  done: boolean;
  terminal: VersusTerminal;
  ticks: number;
}

/**
 * Local human-vs-AI game: two human seats (AI + human) plus engine bots in
 * one sim, driven by direct method calls — no bridge protocol, no workers.
 *
 * Deliberately separate from GameEnv/the training bridge: training seats
 * exactly one learner per env and its wire format stays untouched. Versus
 * play is turn-paced by the human (no real-time deadline), so `step()`
 * advances `ticksPerStep` ticks per paired move. Spawn is seeded heuristic
 * for both seats (same candidate rule as GameEnv, two distinct tiles).
 * No replay recording in versus mode.
 */
export class TwoSeatGame {
  private runner!: GameRunner;
  private feeder!: TurnFeeder;
  private fatalError: string | undefined;
  private aiID!: PlayerID;
  private humanID!: PlayerID;
  private ticksAtSpawnEnd = 0;
  private lastSeed = "";
  private aiSmallID = -1;
  private humanSmallID = -1;
  private readonly aiReward = new Reward();
  private readonly humanReward = new Reward();

  constructor(
    private readonly config: EnvConfig,
    private readonly encoder: ObservationEncoderFn = new ObservationEncoder(),
    private readonly codec: ActionCodec = new ActionCodecImpl(),
  ) {}

  get aiPlayerID(): PlayerID {
    return this.aiID;
  }

  get humanPlayerID(): PlayerID {
    return this.humanID;
  }

  get ticks(): number {
    return this.runner.game.ticks();
  }

  async reset(seed: string): Promise<{ aiObs: Observation; humanObs: Observation }> {
    console.debug = () => {};
    this.fatalError = undefined;
    this.lastSeed = seed;

    const gameConfig: GameConfig = {
      gameMap: this.config.map,
      gameMapSize: GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Public,
      difficulty: Difficulty.Medium,
      nations: this.config.nations ?? "default",
      donateGold: false,
      donateTroops: false,
      bots: this.config.bots,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
    };
    const gameStart: GameStartInfo = {
      gameID: `versus-${seed}`,
      lobbyCreatedAt: 0,
      config: gameConfig,
      players: [],
    };

    const coreConfig = new Config(gameConfig, null, false);
    const terrain = await loadFreshTerrainMap(
      gameConfig.gameMap,
      gameConfig.gameMapSize,
      this.config.mapsDir,
    );
    const random = new PseudoRandom(simpleHash(gameStart.gameID));
    this.aiID = random.nextID();
    this.humanID = random.nextID();
    const nations = createNationsForGame(
      gameStart,
      terrain.nations,
      terrain.additionalNations,
      0,
      random,
    );

    const game = createGame(
      [
        new PlayerInfo("AIAgent", PlayerType.Human, AI_CLIENT_ID, this.aiID),
        new PlayerInfo("Human", PlayerType.Human, HUMAN_CLIENT_ID, this.humanID),
      ],
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      coreConfig,
      terrain.teamGameSpawnAreas,
    );

    this.runner = new GameRunner(
      game,
      new Executor(game, gameStart.gameID, undefined),
      (gu: GameUpdateViewData | ErrorUpdate) => {
        if ("errMsg" in gu) this.fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
      },
    );
    this.runner.init();
    this.feeder = new TurnFeeder(this.runner);

    const [aiSpawn, humanSpawn] = this.pickSpawnTiles(seed);
    if (aiSpawn < 0 || humanSpawn < 0) {
      throw new Error("reset(): no free land tiles found for versus spawns");
    }
    this.feeder.step(1, [
      { clientID: AI_CLIENT_ID, playerID: this.aiID, type: "spawn", tile: aiSpawn },
      { clientID: HUMAN_CLIENT_ID, playerID: this.humanID, type: "spawn", tile: humanSpawn },
    ] as any);
    while (this.runner.game.inSpawnPhase()) {
      if (this.fatalError !== undefined) throw new Error(`sim error during reset(): ${this.fatalError}`);
      this.feeder.step(1, []);
    }

    this.ticksAtSpawnEnd = this.runner.game.ticks();
    this.aiReward.init(this.runner.game, this.aiID);
    this.humanReward.init(this.runner.game, this.humanID);
    this.aiSmallID = this.resolveSmallID(this.aiID);
    this.humanSmallID = this.resolveSmallID(this.humanID);
    return { aiObs: this.encode(this.aiID), humanObs: this.encode(this.humanID) };
  }

  step(aiAction: Action, humanAction: Action): VersusStep {
    if (this.fatalError !== undefined) {
      return this.makeStep(new Float32Array(this.encoder.size), new Float32Array(this.encoder.size), 0, 0, "sim_error");
    }
    let intents: any[];
    try {
      intents = [
        ...this.codec.decode(aiAction, this.runner.game, this.aiID),
        ...this.codec.decode(humanAction, this.runner.game, this.humanID),
      ];
    } catch (e) {
      // Out-of-range heads are caller bugs: surface them, don't sim_error.
      throw e;
    }
    try {
      const result = this.feeder.step(this.config.ticksPerStep, intents);
      if (!result.ok || this.fatalError !== undefined) {
        return this.makeStep(new Float32Array(this.encoder.size), new Float32Array(this.encoder.size), 0, 0, "sim_error");
      }
      const terminal = this.computeTerminal();
      const aiObs = this.encode(this.aiID);
      const humanObs = this.encode(this.humanID);
      const aiReward = this.aiReward.step(this.runner.game, this.aiID, this.rewardTerminal(terminal, true));
      const humanReward = this.humanReward.step(this.runner.game, this.humanID, this.rewardTerminal(terminal, false));
      return this.makeStep(aiObs, humanObs, aiReward, humanReward, terminal);
    } catch (e) {
      const detail = e instanceof Error ? (e.stack ?? e.message) : String(e);
      this.fatalError = `raw engine throw: ${detail}`;
      return this.makeStep(new Float32Array(this.encoder.size), new Float32Array(this.encoder.size), 0, 0, "sim_error");
    }
  }

  masksFor(playerID: PlayerID): MaskBuffers {
    const masks = createMaskBuffers();
    try {
      this.codec.masks(this.runner.game, playerID, masks);
    } catch {
      masks.intentType.fill(0);
      masks.target.fill(0);
      masks.ratioBucket.fill(0);
      masks.spatialCell.fill(0);
      masks.buildSite.fill(0);
    }
    return masks;
  }

  /**
   * Compact ownership snapshot for the dashboard canvas: 64x32 cells over
   * the letterboxed 128x64 raster geometry. Per cell: 0 water, 1 unowned
   * land, 2 AI, 3 human, 4 other. Plus per-seat tile/troop/alive stats.
   *
   * Each snapshot cell covers ~30x30 engine tiles, so one center sample
   * would miss small territories: a 5x5 subgrid probes land/water/other
   * ownership, then every AI- and human-owned tile is marked exactly via a
   * single tile-state scan — both seats always render, however small their
   * foothold. Presence wins over majority (AI > human > other > unowned >
   * water) in the subgrid pass.
   */
  snapshot(forHuman = true): {
    w: number; h: number; cells: number[];
    ai: { tiles: number; troops: number; alive: boolean };
    human: { tiles: number; troops: number; alive: boolean };
    ticks: number; done: boolean; terminal: VersusTerminal;
  } {
    const W = 64;
    const H = 32;
    const SUB = 5;
    const game = this.runner.game;
    const map = game.map();
    const stateBuf = map.tileStateBuffer();
    const mapW = map.width();
    const mapH = map.height();
    // Raster geometry shared by the subgrid probe and the exact marking
    // pass below: engine tile -> snapshot cell through the 128x64 grid.
    const scale = Math.max(mapW / 128, mapH / 64);
    const scaledW = mapW / scale;
    const scaledH = mapH / scale;
    const offsetX = Math.floor((128 - scaledW) / 2);
    const offsetY = Math.floor((64 - scaledH) / 2);
    const tileToCell = (tile: number): number => {
      const tx = tile % mapW;
      const ty = Math.floor(tile / mapW);
      const rx = Math.floor(tx / scale + offsetX);
      const ry = Math.floor(ty / scale + offsetY);
      if (rx < 0 || ry < 0 || rx >= 128 || ry >= 64) return -1;
      const cx = Math.min(W - 1, Math.floor((rx / 128) * W));
      const cy = Math.min(H - 1, Math.floor((ry / 64) * H));
      return cy * W + cx;
    };
    const cells: number[] = new Array(W * H);
    for (let cy = 0; cy < H; cy++) {
      for (let cx = 0; cx < W; cx++) {
        let code = 0;
        for (let sy = 0; sy < SUB; sy++) {
          for (let sx = 0; sx < SUB; sx++) {
            const rx = Math.floor((((cx + (sx + 0.5) / SUB) / W) * 128));
            const ry = Math.floor((((cy + (sy + 0.5) / SUB) / H) * 64));
            const tile = this.rasterCellToTile(rx, ry, mapW, mapH);
            if (tile < 0 || !map.isLand(tile)) continue;
            const owner = stateBuf[tile] & 0xfff;
            let sample = 1;
            if (owner !== 0) {
              if (owner === this.aiSmallID) sample = 2;
              else if (owner === this.humanSmallID) sample = 3;
              else sample = 4;
            }
            if (sample === 2) {
              code = 2;
              break;
            }
            if (sample === 3 && code !== 2) code = 3;
            else if (sample === 4 && code !== 2 && code !== 3) code = 4;
            else if (code === 0) code = 1;
          }
          if (code === 2) break;
        }
        cells[cy * W + cx] = code;
      }
    }
    // Exact marking: every AI/human-owned tile forces its snapshot cell.
    // A 50-tile spawn is ~2.5% of one probe sample away from invisible;
    // this scan makes both seats render deterministically.
    if (this.aiSmallID >= 0 || this.humanSmallID >= 0) {
      for (let tile = 0; tile < mapW * mapH; tile++) {
        if (!map.isLand(tile)) continue;
        const owner = stateBuf[tile] & 0xfff;
        if (owner !== this.aiSmallID && owner !== this.humanSmallID) continue;
        const cell = tileToCell(tile);
        if (cell >= 0) cells[cell] = owner === this.aiSmallID ? 2 : 3;
      }
    }
    void forHuman;
    const ai = game.player(this.aiID);
    const human = game.player(this.humanID);
    const terminal = this.computeTerminal();
    return {
      w: W, h: H, cells,
      ai: { tiles: ai.numTilesOwned(), troops: ai.troops(), alive: ai.isAlive() },
      human: { tiles: human.numTilesOwned(), troops: human.troops(), alive: human.isAlive() },
      ticks: game.ticks(), done: terminal !== null, terminal,
    };
  }

  private makeStep(aiObs: Observation, humanObs: Observation, aiReward: number, humanReward: number, terminal: VersusTerminal): VersusStep {
    const done = terminal !== null;
    return {
      aiObs, humanObs,
      aiMasks: done && terminal === "sim_error" ? createMaskBuffers() : this.masksFor(this.aiID),
      humanMasks: done && terminal === "sim_error" ? createMaskBuffers() : this.masksFor(this.humanID),
      aiReward, humanReward, done, terminal, ticks: this.runner.game.ticks(),
    };
  }

  private computeTerminal(): VersusTerminal {
    if (this.fatalError !== undefined) return "sim_error";
    const game = this.runner.game;
    let aiAlive = true;
    let humanAlive = true;
    try {
      aiAlive = game.player(this.aiID).isAlive();
      humanAlive = game.player(this.humanID).isAlive();
    } catch {
      return "sim_error";
    }
    if (!aiAlive && !humanAlive) return "draw";
    if (!aiAlive) return "human_win";
    if (!humanAlive) return "ai_win";
    if (game.ticks() - this.ticksAtSpawnEnd >= this.config.tickCap) return "tick_cap";
    return null;
  }

  private rewardTerminal(terminal: VersusTerminal, forAI: boolean): "win" | "death" | "tick_cap" | "sim_error" | null {
    switch (terminal) {
      case "ai_win": return forAI ? "win" : "death";
      case "human_win": return forAI ? "death" : "win";
      case "draw": return "death";
      case "tick_cap": return "tick_cap";
      case "sim_error": return "sim_error";
      default: return null;
    }
  }

  private encode(playerID: PlayerID): Observation {
    const obs = new Float32Array(this.encoder.size);
    this.encoder.encode(this.runner.game, playerID, obs);
    return obs;
  }

  private smallID(playerID: PlayerID): number {
    if (playerID === this.aiID && this.aiSmallID >= 0) return this.aiSmallID;
    if (playerID === this.humanID && this.humanSmallID >= 0) return this.humanSmallID;
    return this.resolveSmallID(playerID);
  }

  private resolveSmallID(playerID: PlayerID): number {
    // playerBySmallID is keyed by small numeric ID; find ours by scanning
    // the 12-bit space once (cheap: 4096 iterations per snapshot).
    const game = this.runner.game;
    for (let id = 1; id < 4096; id++) {
      try {
        const p = game.playerBySmallID(id) as { id?: () => PlayerID };
        if (p && typeof p.id === "function" && p.id() === playerID) return id;
      } catch {
        continue;
      }
    }
    return -1;
  }

  private rasterCellToTile(rx: number, ry: number, mapW: number, mapH: number): number {
    // Mirror the encoder letterbox: scale = max(mapW/128, mapH/64), centered.
    const scale = Math.max(mapW / 128, mapH / 64);
    const scaledW = mapW / scale;
    const scaledH = mapH / scale;
    const offsetX = Math.floor((128 - scaledW) / 2);
    const offsetY = Math.floor((64 - scaledH) / 2);
    const lx = rx - offsetX;
    const ly = ry - offsetY;
    if (lx < 0 || ly < 0 || lx >= scaledW || ly >= scaledH) return -1;
    const tx = Math.min(mapW - 1, Math.floor(lx * scale));
    const ty = Math.min(mapH - 1, Math.floor(ly * scale));
    return ty * mapW + tx;
  }

  private pickSpawnTiles(seed: string): [number, number] {
    const map = this.runner.game.map();
    const w = map.width();
    const h = map.height();
    const candidates: number[] = [];
    for (let i = 0; i < w * h; i++) {
      if (!map.isLand(i) || map.hasOwner(i)) continue;
      let neutralNeighbours = 0;
      for (const n of map.neighbors(i)) {
        if (map.isLand(n) && !map.hasOwner(n)) neutralNeighbours++;
      }
      if (neutralNeighbours === 4) candidates.push(i);
    }
    if (candidates.length < 2) return [-1, -1];
    // Seeded shuffle, then take the first tile plus the tile farthest from
    // it among the next 8 draws: distinct spawns that usually start apart.
    const random = new PseudoRandom(simpleHash(`${seed}:versus-spawn`));
    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(random.next() * (i + 1));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }
    const first = candidates[0];
    const fx = first % w;
    const fy = Math.floor(first / w);
    let best = candidates[1];
    let bestDist = -1;
    for (const tile of candidates.slice(1, 9)) {
      const dx = (tile % w) - fx;
      const dy = Math.floor(tile / w) - fy;
      const dist = dx * dx + dy * dy;
      if (dist > bestDist) {
        bestDist = dist;
        best = tile;
      }
    }
    return [first, best];
  }
}
