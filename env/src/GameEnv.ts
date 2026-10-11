import { Config } from "../../OpenFrontIO/src/core/configuration/Config";
import { Executor } from "../../OpenFrontIO/src/core/execution/ExecutionManager";
import {
  Difficulty,
  Game,
  GameMapSize,
  GameMode,
  GameType,
  PlayerID,
  PlayerInfo,
  PlayerType,
} from "../../OpenFrontIO/src/core/game/Game";
import {
  BrokeAllianceUpdate,
  ConquestUpdate,
  ErrorUpdate,
  GameUpdateType,
  GameUpdateViewData,
  HashUpdate,
} from "../../OpenFrontIO/src/core/game/GameUpdates";
import { createGame } from "../../OpenFrontIO/src/core/game/GameImpl";
import { createNationsForGame } from "../../OpenFrontIO/src/core/game/NationCreation";
import { GameRunner } from "../../OpenFrontIO/src/core/GameRunner";
import { PseudoRandom } from "../../OpenFrontIO/src/core/PseudoRandom";
import type { GameConfig, GameStartInfo, Turn } from "../../OpenFrontIO/src/core/Schemas";
import { simpleHash } from "../../OpenFrontIO/src/core/Util";
import { ActionCodecImpl } from "./ActionCodec";
import { CombatTracker, observeConquests } from "./Combat";
import { DeathReport } from "./DeathReport";
import { RewardWeights } from "./BehaviorConfig";
import { createMasks, Masks } from "./Layout";
import { createSeatObservation, ObservationEncoder, SeatObservation, MAX_SEATS } from "./ObservationEncoder";
import { buildGameRecord, replayGameID, seatClientID, shouldRecordReplay, writeReplayFile } from "./Replay";
import { boatEvents, BoatTrack, earnedGold, newDefensePosts, nukeEvents, NukeTrack, placementScore, Reward, SeatEvents } from "./Reward";
import { loadFreshTerrainMap } from "./Terrain";
import { TurnFeeder } from "./TurnFeeder";
import type { Action, EnvConfig, TerminalReason } from "./types";

/** What one seat sees after a reset or step. */
export interface SeatStep {
  obs: SeatObservation;
  masks: Masks;
  reward: number;
  /** This seat's episode ended on this step (terminal says why). */
  done: boolean;
  /** The seat acted on this step / will act on the next one. Dead seats go
   *  inactive until the game resets; their rows carry no data. */
  active: boolean;
  terminal: TerminalReason;
  /** Episode-end readouts for logging, valid on every row. */
  tiles: number;
  troops: number;
  kills: number;
  placement: number;
}

interface Seat {
  id: PlayerID;
  clientID: string;
  smallID: number;
  reward: Reward;
  active: boolean;
  kills: number;
  pendingKills: number;
  pendingBetrayals: number;
  combat: CombatTracker;
  /** earnedGold() at the previous step. */
  earned: number;
  /** Own defense post IDs at the previous step. */
  posts: Set<number>;
  /** Own boats at the previous step. */
  boats: Map<number, BoatTrack>;
  /** Own nukes in flight at the previous step. */
  nukes: Map<number, NukeTrack>;
  /** Set when OPENFRONT_DEATH_LOG is on. */
  death?: DeathReport;
}

const nukeTerms = (d: { hostile: number; friendly: number }) => ({ nukeDamage: d.hostile, nukeSelfDamage: d.friendly });

/** Minimum tile distance between seat spawns, halved until it fits. */
const SEAT_SPAWN_DISTANCE = 150;
/** Close-spawn games: every seat within this distance of seat 0. */
const CLOSE_SPAWN_MAX = 60;
const CLOSE_SPAWN_MIN = 15;

/** Decoded intents priced by the `diplomacy` reward weight. */
const DIPLOMATIC_INTENTS = new Set(["allianceRequest", "breakAlliance", "embargo", "targetPlayer"]);

/**
 * One game with `config.seats` agent-controlled human seats plus engine bots
 * and nations. seats = 1 is classic training vs bots; seats > 1 is real
 * self-play: several policies (the learner and pool snapshots, chosen by the
 * trainer) fight inside the same simulation, alongside the bots.
 *
 * Constructed the way the real game is: production Config and map, nations
 * from the manifest, and every seat a PlayerType.Human whose actions are
 * ordinary Intents inside Turns — the path the browser client uses. Seat
 * IDs are drawn from the game's PRNG before nations, exactly like
 * createGameRunner, so recorded games replay hash-for-hash.
 *
 * Episode end per seat: death (seat goes inactive, the game continues for
 * the others), win, or the post-spawn tick cap (placement reward). The game
 * ends when every seat is done or on a sim error; the caller resets it.
 */
export class GameEnv {
  readonly encoder: ObservationEncoder;
  private runner!: GameRunner;
  private feeder!: TurnFeeder;
  private seats: Seat[] = [];
  private fatalError: string | undefined;
  private winnerClientID: string | null | undefined;
  private ticksAtSpawnEnd = 0;
  private lastSeed = "";
  private over = true;
  private gamesCompleted = 0;
  private gameConfig: GameConfig | undefined;
  private startedAt = 0;
  private replayHashes = new Map<number, number>();
  private readonly obs: SeatObservation[];
  private readonly maskBufs: Masks[];

  constructor(
    private readonly config: EnvConfig,
    private readonly codec: ActionCodecImpl = new ActionCodecImpl(),
    private readonly rewardWeights: Partial<RewardWeights> = {},
    encoder?: ObservationEncoder,
  ) {
    if (!Number.isInteger(config.seats) || config.seats < 1 || config.seats > MAX_SEATS) {
      throw new Error(`seats must be 1..${MAX_SEATS}, got ${config.seats}`);
    }
    this.encoder = encoder ?? new ObservationEncoder(config.tickCap);
    this.obs = Array.from({ length: config.seats }, createSeatObservation);
    this.maskBufs = Array.from({ length: config.seats }, () => createMasks());
  }

  get numSeats(): number {
    return this.config.seats;
  }
  get seatIDs(): PlayerID[] {
    return this.seats.map((s) => s.id);
  }
  get game(): Game {
    return this.runner.game;
  }
  get ticks(): number {
    return this.runner.game.ticks();
  }
  /** True once every seat is done (or a sim error hit); reset() next. */
  get gameOver(): boolean {
    return this.over;
  }

  async reset(seed: string): Promise<SeatStep[]> {
    console.debug = () => {};
    this.fatalError = undefined;
    this.winnerClientID = undefined;
    this.lastSeed = seed;
    const S = this.config.seats;

    const gameConfig: GameConfig = {
      gameMap: this.config.map,
      gameMapSize: GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Public,
      difficulty: this.config.difficulty ?? Difficulty.Medium,
      nations: this.config.nations ?? "default",
      donateGold: false,
      donateTroops: false,
      bots: this.config.bots,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
    };
    this.gameConfig = gameConfig;
    this.startedAt = Date.now();
    const gameStart: GameStartInfo = { gameID: replayGameID(seed), lobbyCreatedAt: 0, config: gameConfig, players: [] };

    // Fresh terrain every reset: the engine's loadTerrainMap() caches the
    // LIVE map, so reuse would start game N on game N-1's ownership.
    const terrain = await loadFreshTerrainMap(gameConfig.gameMap, gameConfig.gameMapSize, this.config.mapsDir);
    const random = new PseudoRandom(simpleHash(gameStart.gameID));
    const humans: PlayerInfo[] = [];
    for (let i = 0; i < S; i++) {
      humans.push(new PlayerInfo(`RLAgent${i}`, PlayerType.Human, seatClientID(i), random.nextID()));
    }
    const nations = createNationsForGame(gameStart, terrain.nations, terrain.additionalNations, S, random);
    const game = createGame(humans, nations, terrain.gameMap, terrain.miniGameMap, new Config(gameConfig, null, false), terrain.teamGameSpawnAreas);
    observeConquests(game, (winner, loser) => {
      for (const seat of this.seats) {
        seat.combat.onConquer(winner, loser);
        if (seat.death !== undefined && loser !== null && loser.id() === seat.id) seat.death.onConquer(winner);
      }
    });
    this.runner = new GameRunner(game, new Executor(game, gameStart.gameID, undefined), (gu) => this.onGameUpdate(gu));
    this.runner.init();
    this.feeder = new TurnFeeder(this.runner);
    this.replayHashes = new Map();
    this.seats = humans.map((h) => {
      const smallID = game.player(h.id).smallID();
      return {
        id: h.id,
        clientID: h.clientID!,
        smallID,
        reward: new Reward(this.rewardWeights),
        active: true,
        kills: 0,
        pendingKills: 0,
        pendingBetrayals: 0,
        combat: new CombatTracker(smallID),
        earned: 0,
        posts: new Set<number>(),
        boats: new Map<number, BoatTrack>(),
        nukes: new Map<number, NukeTrack>(),
        death: DeathReport.file() !== undefined ? new DeathReport() : undefined,
      };
    });

    // Spawn tiles come from their own stream; they are recorded in the turns,
    // so replays never resample them.
    const spawns = this.pickSpawnTiles(new PseudoRandom(simpleHash(`${seed}:spawn`)), S);
    this.feeder.step(
      1,
      this.seats.map((s, i) => ({ type: "spawn", clientID: s.clientID, tile: spawns[i] })) as Turn["intents"],
    );
    while (game.inSpawnPhase()) {
      if (this.fatalError !== undefined) throw new Error(`sim error during reset(): ${this.fatalError}`);
      this.feeder.step(1, []);
    }
    if (this.fatalError !== undefined) throw new Error(`sim error during reset(): ${this.fatalError}`);
    this.ticksAtSpawnEnd = game.ticks();
    for (const seat of this.seats) {
      seat.reward.init(game, seat.id);
      seat.earned = earnedGold(game, game.player(seat.id));
    }
    this.over = false;
    return this.observe(this.seats.map(() => null), this.seats.map(() => 0));
  }

  step(actions: Action[]): SeatStep[] {
    if (this.over) throw new Error("step() after the game ended; call reset() first");
    if (actions.length !== this.seats.length) {
      throw new Error(`need ${this.seats.length} actions, got ${actions.length}`);
    }
    // decode() throws RangeError only for out-of-range heads (a caller bug):
    // kept outside the sim try so it fails loudly instead of as sim_error.
    const intents: Turn["intents"] = [];
    const diplomacy = this.seats.map(() => 0);
    this.seats.forEach((seat, i) => {
      if (!seat.active) return;
      // decode() stamps each intent with the seat's clientID.
      const decoded = this.codec.decode(actions[i], this.runner.game, seat.id) as Turn["intents"];
      diplomacy[i] = decoded.filter((intent) => DIPLOMATIC_INTENTS.has(intent.type)).length;
      intents.push(...decoded);
    });
    try {
      const result = this.feeder.step(this.config.ticksPerStep, intents);
      if (!result.ok || this.fatalError !== undefined) return this.simError(new Error(this.fatalError ?? "tick failed"));
      const terminals = this.computeTerminals();
      const game = this.runner.game;
      const rewards = this.seats.map((seat, i) => {
        if (!seat.active) return 0;
        const player = game.player(seat.id);
        seat.combat.update(player);
        const earned = earnedGold(game, player);
        const posts = newDefensePosts(game, player, seat.posts);
        const boats = boatEvents(game, player, seat.boats);
        const events: SeatEvents = {
          earned: earned - seat.earned,
          postsBuilt: posts.built,
          postsCrowded: posts.crowded,
          boatsLaunched: boats.launched,
          boatsLanded: boats.landed,
          ...nukeTerms(nukeEvents(game, player, seat.nukes)),
          kills: seat.pendingKills,
          betrayals: seat.pendingBetrayals,
          diplomacy: diplomacy[i],
          ...seat.combat.take(),
        };
        seat.earned = earned;
        seat.kills += events.kills;
        seat.pendingKills = 0;
        seat.pendingBetrayals = 0;
        return seat.reward.step(game, seat.id, terminals[i], events);
      });
      const out = this.observe(terminals, rewards);
      this.reportDeaths(terminals);
      this.seats.forEach((seat, i) => {
        if (terminals[i] !== null) seat.active = false;
      });
      if (this.seats.every((s) => !s.active)) this.finish();
      return out;
    } catch (e) {
      return this.simError(e);
    }
  }

  /**
   * End the game now (the trainer has no use for the remaining seats, e.g.
   * every learner seat is dead). Active seats report done with "aborted"
   * and zero reward.
   */
  abort(): SeatStep[] {
    if (this.over) return this.observe(this.seats.map(() => null), this.seats.map(() => 0));
    const terminals: TerminalReason[] = this.seats.map((s) => (s.active ? "aborted" : null));
    const out = this.observe(terminals, this.seats.map(() => 0), false);
    for (const seat of this.seats) seat.active = false;
    this.finish();
    return out;
  }

  /** Rows for the current state. Masks are computed only for seats that act next. */
  private observe(terminals: TerminalReason[], rewards: number[], encode = true): SeatStep[] {
    const game = this.runner.game;
    if (encode) this.encoder.encode(game, this.seatIDs, this.obs);
    return this.seats.map((seat, i) => {
      const masks = this.maskBufs[i];
      const done = terminals[i] !== null;
      if (seat.active && !done) {
        this.codec.masks(game, seat.id, masks);
      } else {
        masks.flat.fill(0);
        masks.intent[0] = 1;
      }
      const player = game.player(seat.id);
      return {
        obs: this.obs[i],
        masks,
        reward: rewards[i],
        done,
        active: seat.active,
        terminal: terminals[i],
        tiles: player.numTilesOwned(),
        troops: player.troops(),
        kills: seat.kills,
        placement: placementScore(game, player),
      };
    });
  }

  private computeTerminals(): TerminalReason[] {
    const game = this.runner.game;
    const capped = game.ticks() - this.ticksAtSpawnEnd >= this.config.tickCap;
    const gameEnded = this.winnerClientID !== undefined;
    return this.seats.map((seat) => {
      if (!seat.active) return null;
      if (gameEnded && this.winnerClientID === seat.clientID) return "win";
      if (!game.player(seat.id).isAlive()) return "death";
      // Someone else won, or time is up: scored by placement.
      if (gameEnded || capped) return "tick_cap";
      return null;
    });
  }

  private simError(e: unknown): SeatStep[] {
    const detail = e instanceof Error ? (e.stack ?? e.message) : String(e);
    this.fatalError ??= `raw engine throw: ${detail}`;
    console.warn(`GameEnv sim_error, seed ${this.lastSeed}: ${detail}`);
    const terminals: TerminalReason[] = this.seats.map((s) => (s.active ? "sim_error" : null));
    const rows = this.seats.map((seat, i): SeatStep => {
      const masks = this.maskBufs[i];
      masks.flat.fill(0);
      masks.intent[0] = 1;
      this.obs[i].raster.fill(0);
      this.obs[i].features.fill(0);
      return { obs: this.obs[i], masks, reward: 0, done: terminals[i] !== null, active: seat.active, terminal: terminals[i], tiles: 0, troops: 0, kills: seat.kills, placement: 0 };
    });
    for (const seat of this.seats) seat.active = false;
    this.finish();
    return rows;
  }

  private reportDeaths(terminals: TerminalReason[]): void {
    const file = DeathReport.file();
    if (file === undefined) return;
    const game = this.runner.game;
    const minute = Math.round(((game.ticks() - this.ticksAtSpawnEnd) / 600) * 10) / 10;
    this.seats.forEach((seat, i) => {
      if (seat.death === undefined || !seat.active) return;
      if (terminals[i] === "death") {
        seat.death.write(file, { seed: this.lastSeed, seat: i, minute, difficulty: String(this.config.difficulty ?? "Medium") });
      } else {
        seat.death.step(game, game.player(seat.id), minute);
      }
    });
  }

  private finish(): void {
    if (this.over) return;
    this.over = true;
    this.maybeWriteReplay();
  }

  private onGameUpdate(gu: GameUpdateViewData | ErrorUpdate): void {
    if ("errMsg" in gu) {
      this.fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
      return;
    }
    for (const hu of gu.updates[GameUpdateType.Hash] as HashUpdate[]) {
      this.replayHashes.set(hu.tick, hu.hash);
    }
    // Honest kill credit: the engine names the conqueror of every eliminated player.
    for (const cu of gu.updates[GameUpdateType.ConquestEvent] as ConquestUpdate[]) {
      const seat = this.seats.find((s) => s.id === cu.conquerorId);
      if (seat !== undefined) seat.pendingKills++;
    }
    for (const bu of gu.updates[GameUpdateType.BrokeAlliance] as BrokeAllianceUpdate[]) {
      const seat = this.seats.find((s) => s.smallID === bu.traitorID);
      if (seat !== undefined) seat.pendingBetrayals++;
    }
    const wins = gu.updates[GameUpdateType.Win];
    if (wins.length > 0) {
      const winner = wins[wins.length - 1].winner;
      // ["player", clientID] per GameImpl.makeWinner; any other shape is not a seat.
      this.winnerClientID = winner !== undefined && winner[0] === "player" ? String(winner[1]) : null;
    }
  }

  private maybeWriteReplay(): void {
    const replay = this.config.replay;
    const index = this.gamesCompleted++;
    if (replay === undefined || this.gameConfig === undefined || !shouldRecordReplay(index, replay.every, replay.slot, replay.stride)) return;
    try {
      const record = buildGameRecord({
        seed: this.lastSeed,
        gameConfig: this.gameConfig,
        turns: this.feeder.turns,
        seats: this.config.seats,
        winnerClientID: this.winnerClientID ?? null,
        startedAt: this.startedAt,
        endedAt: Date.now(),
        hashes: this.replayHashes,
      });
      // The best any agent seat did, not just seat 0: a game where seat 0
      // fell early but another seat took the map is not a loss.
      const game = this.runner.game;
      const won = this.seats.some((s) => s.clientID === this.winnerClientID);
      const alive = this.seats.some((s) => game.player(s.id).isAlive());
      const outcome = won ? "won" : alive ? "alive" : "dead";
      const file = writeReplayFile(replay, this.lastSeed, index, outcome, record);
      if (file !== undefined) console.log(`replay recorded: ${file}`);
    } catch (e) {
      console.warn(`replay build failed, seed ${this.lastSeed}: ${String(e)}`);
    }
  }

  /**
   * Seeded spawn tiles: neutral land with four neutral land neighbours.
   * Usually spread at least SEAT_SPAWN_DISTANCE apart (halved until all
   * seats fit); with probability `closeSpawn` every seat lands within
   * CLOSE_SPAWN_MAX of seat 0, so the policy also learns openings with a
   * hostile neighbour from tick one.
   */
  private pickSpawnTiles(random: PseudoRandom, count: number): number[] {
    const map = this.runner.game.map();
    const w = map.width();
    const candidates: number[] = [];
    for (let i = 0; i < w * map.height(); i++) {
      if (!map.isLand(i) || map.hasOwner(i)) continue;
      let neutral = 0;
      for (const n of map.neighbors(i)) if (map.isLand(n) && !map.hasOwner(n)) neutral++;
      if (neutral === 4) candidates.push(i);
    }
    if (candidates.length < count) throw new Error("reset(): not enough free land for spawns");
    const dist = (a: number, b: number) => Math.hypot((a % w) - (b % w), Math.floor(a / w) - Math.floor(b / w));
    const picked = [random.randElement(candidates)];
    const closeSpawn = this.config.closeSpawn ?? 0;
    if (count > 1 && closeSpawn > 0 && random.nextInt(0, 1000) < closeSpawn * 1000) {
      const near = candidates.filter((t) => dist(t, picked[0]) <= CLOSE_SPAWN_MAX);
      for (let tries = 0; tries < 500 && picked.length < count && near.length > 0; tries++) {
        const t = random.randElement(near);
        if (picked.every((p) => dist(p, t) >= CLOSE_SPAWN_MIN)) picked.push(t);
      }
      if (picked.length === count) return picked;
      picked.length = 1; // too little land around seat 0: fall back to spread
    }
    for (let minDist = SEAT_SPAWN_DISTANCE; picked.length < count; minDist = Math.floor(minDist / 2)) {
      for (let tries = 0; tries < 2000 && picked.length < count; tries++) {
        const t = random.randElement(candidates);
        const far = picked.every((p) => dist(p, t) >= minDist);
        if (far) picked.push(t);
      }
    }
    return picked;
  }
}
