import { Config } from "../../OpenFrontIO/src/core/configuration/Config";
import { Executor } from "../../OpenFrontIO/src/core/execution/ExecutionManager";
import {
  Difficulty,
  GameMapSize,
  GameMode,
  GameType,
  PlayerID,
  PlayerInfo,
  PlayerType,
} from "../../OpenFrontIO/src/core/game/Game";
import {
  ErrorUpdate,
  GameUpdateType,
  GameUpdateViewData,
} from "../../OpenFrontIO/src/core/game/GameUpdates";
import { createGame } from "../../OpenFrontIO/src/core/game/GameImpl";
import { createNationsForGame } from "../../OpenFrontIO/src/core/game/NationCreation";
import { loadFreshTerrainMap } from "./Terrain";
import { GameRunner } from "../../OpenFrontIO/src/core/GameRunner";
import { PseudoRandom } from "../../OpenFrontIO/src/core/PseudoRandom";
import { GameConfig, GameStartInfo } from "../../OpenFrontIO/src/core/Schemas";
import { simpleHash } from "../../OpenFrontIO/src/core/Util";
import { Reward, TerminalReason } from "./Reward";
import { TurnFeeder } from "./TurnFeeder";
import { createMaskBuffers } from "./ActionCodec";
import {
  Action,
  ActionCodec,
  EnvConfig,
  MaskBuffers,
  ObservationEncoder,
  Observation,
  StepResult,
} from "./types";

const AGENT_ID: PlayerID = "agent0";
const AGENT_CLIENT_ID = "client0";

/**
 * Owns one episode's lifecycle (spec §5.1). Constructed the way
 * FullGamePerf.ts constructs its game — real Config, production map via
 * NodeGameMapLoader — except the human seat is passed to createGame()
 * instead of left empty. The agent's actions become ordinary Intents
 * inside a Turn, the same path the real browser client uses (verified
 * §2.7).
 *
 * Phase-1 simplification, flagged rather than hidden: `reset()` picks the
 * agent's spawn tile by seeded sampling (not a policy action) among neutral
 * land tiles with four neutral neighbours. Real spawn placement is a
 * legitimate thing to learn eventually, but gating GameEnv's own lifecycle on
 * that design isn't worth it. Swapping this for an action-driven spawn is a
 * local change inside reset(), not a GameEnv interface change.
 */
export class GameEnv {
  private runner!: GameRunner;
  private feeder!: TurnFeeder;
  private fatalError: string | undefined;
  private winnerAgentID: PlayerID | undefined;
  private ticksAtSpawnEnd = 0;
  private lastSeed = "";
  private readonly reward: Reward;

  constructor(
    private readonly config: EnvConfig,
    private readonly encoder: ObservationEncoder,
    private readonly codec: ActionCodec,
    reward?: Reward,
  ) {
    this.reward = reward ?? new Reward();
  }

  async reset(seed: string): Promise<Observation> {
    console.debug = () => {};
    this.fatalError = undefined;
    this.winnerAgentID = undefined;
    this.lastSeed = seed;

    const gameConfig: GameConfig = {
      gameMap: this.config.map,
      gameMapSize: GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Public,
      difficulty: Difficulty.Medium,
      nations: "default",
      donateGold: false,
      donateTroops: false,
      bots: this.config.bots,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
    };
    const gameStart: GameStartInfo = {
      gameID: seed,
      lobbyCreatedAt: 0,
      config: gameConfig,
      players: [],
    };

    const coreConfig = new Config(gameConfig, null, false);
    // Fresh terrain every reset: the engine's loadTerrainMap() caches the
    // LIVE GameMap by map:size with no eviction, so reusing it would build
    // episode N on episode 1's dirt (see Terrain.ts).
    const terrain = await loadFreshTerrainMap(
      gameConfig.gameMap,
      gameConfig.gameMapSize,
      this.config.mapsDir,
    );
    const random = new PseudoRandom(simpleHash(seed));
    const nations = createNationsForGame(
      gameStart,
      terrain.nations,
      terrain.additionalNations,
      0,
      random,
    );

    const human = new PlayerInfo(
      "RLAgent",
      PlayerType.Human,
      AGENT_CLIENT_ID,
      AGENT_ID,
    );
    const game = createGame(
      [human],
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      coreConfig,
      terrain.teamGameSpawnAreas,
    );

    this.runner = new GameRunner(
      game,
      new Executor(game, seed, undefined),
      (gu: GameUpdateViewData | ErrorUpdate) => this.onGameUpdate(gu),
    );
    this.runner.init();
    this.feeder = new TurnFeeder(this.runner);

    const spawnTile = this.pickSpawnTile(random);
    if (spawnTile < 0) {
      throw new Error("reset(): no free land tile found for agent spawn");
    }
    this.feeder.step(1, [
      { clientID: AGENT_CLIENT_ID, playerID: AGENT_ID, type: "spawn", tile: spawnTile },
    ] as any);
    while (this.runner.game.inSpawnPhase()) {
      this.assertNoFatalError();
      this.feeder.step(1, []);
    }
    this.assertNoFatalError();

    this.ticksAtSpawnEnd = this.runner.game.ticks();
    this.reward.init(this.runner.game, AGENT_ID);

    return this.encodeObservation();
  }

  step(action: Action): StepResult {
    // Intentionally asymmetric with the other three terminal reasons. After
    // win/death/tick_cap the episode is over but the engine underneath is
    // perfectly healthy — GameEnv doesn't police re-stepping a done episode
    // (see GameEnv.test.ts's "does not auto-reset" case; that's the
    // caller's contract to honor, same as any Gym env). A sim_error means a
    // caught exception happened *inside* GameRunner.executeNextTick() —
    // engine-internal state may be left inconsistent, so continuing to
    // drive it is unsafe in a way the other three terminals aren't. Hence
    // the hard stop here applies only to this one case.
    if (this.fatalError !== undefined) {
      throw new Error("step() called after a terminal episode; call reset() first");
    }

    // decode() stays outside the try below on purpose: it never throws on
    // in-range heads (illegal combinations decode to []), so a throw here is
    // a caller bug (out-of-range head), not a simulation failure, and must
    // keep failing fast rather than masquerading as sim_error.
    const intents = this.codec.decode(action, this.runner.game, AGENT_ID);
    try {
      const result = this.feeder.step(this.config.ticksPerStep, intents as any);

      const terminal = this.computeTerminal(result.ok);
      const obs = this.encodeObservation();
      const reward = this.reward.step(this.runner.game, AGENT_ID, terminal);
      const masks = createMaskBuffers();
      this.codec.masks(this.runner.game, AGENT_ID, masks);

      return {
        obs,
        reward,
        done: terminal !== null,
        info: { terminalReason: terminal, ticks: this.runner.game.ticks(), masks },
      };
    } catch (e) {
      // GameRunner catches tick errors into ErrorUpdate (handled above via
      // fatalError), but NOT everything it runs: the per-tick view-data pass
      // (placeName over every player) throws raw past executeNextTick's
      // try/catch, as do engine queries on state a caught error already left
      // inconsistent. A raw throw must end the episode as sim_error, never
      // escape into the training batch or kill the worker (§7).
      return this.simError(e);
    }
  }

  close(): void {
    // Nothing to release yet — no worker threads, no open handles. Exists
    // so callers have a stable lifecycle method as the bridge (§5.6) grows.
  }

  /** Test/debug escape hatch onto the underlying engine state. Also what
   *  phase 3's shadow-sim hash check (§10.3) will read the state hash
   *  through, once that's built. */
  get game() {
    return this.runner.game;
  }

  /**
   * Current tick count. The bridge (§5.6) reports it on every reply,
   * including RESET replies whose StepResult never exists.
   */
  get ticks(): number {
    if (this.runner === undefined) {
      throw new Error("ticks read before reset()");
    }
    return this.runner.game.ticks();
  }

  /**
   * Next step masks for the current state. Same call the step path makes,
   * exposed so the bridge can answer RESET with honest masks instead of
   * all open bits. A damaged state yields zeroed masks, matching the
   * sim_error reply contract.
   */
  currentMasks(): MaskBuffers {
    const masks = createMaskBuffers();
    try {
      this.codec.masks(this.runner.game, AGENT_ID, masks);
    } catch {
      masks.intentType.fill(0);
      masks.target.fill(0);
      masks.ratioBucket.fill(0);
      masks.spatialCell.fill(0);
      masks.buildSite.fill(0);
    }
    return masks;
  }

  private computeTerminal(feederOk: boolean): TerminalReason {
    if (!feederOk || this.fatalError !== undefined) return "sim_error";
    if (this.winnerAgentID !== undefined) {
      return this.winnerAgentID === AGENT_ID ? "win" : "death";
    }
    const agent = this.runner.game.player(AGENT_ID);
    if (!agent.isAlive()) return "death";
    if (this.runner.game.ticks() - this.ticksAtSpawnEnd >= this.config.tickCap) {
      return "tick_cap";
    }
    return null;
  }

  /**
   * Convert a raw engine throw into a non-scoring sim_error terminal (§7).
   * Latches fatalError so further step() calls fail fast until reset(), and
   * returns zeroed obs/masks — all-masked is the only honest mask for a
   * state we no longer trust. Reward is still routed through Reward.step()
   * for the terminal term, falling back to 0 if even reading state throws.
   */
  private simError(e: unknown): StepResult {
    const detail = e instanceof Error ? (e.stack ?? e.message) : String(e);
    this.fatalError = `raw engine throw: ${detail}`;
    console.warn(`GameEnv sim_error, seed ${this.lastSeed}: ${detail}`);

    let reward = 0;
    try {
      reward = this.reward.step(this.runner.game, AGENT_ID, "sim_error");
    } catch {
      reward = 0;
    }
    return {
      obs: new Float32Array(this.encoder.size),
      reward,
      done: true,
      info: {
        terminalReason: "sim_error",
        ticks: this.runner.game.ticks(),
        masks: createMaskBuffers(),
      },
    };
  }

  private encodeObservation(): Observation {
    const obs = new Float32Array(this.encoder.size);
    this.encoder.encode(this.runner.game, AGENT_ID, obs);
    return obs;
  }

  private assertNoFatalError(): void {
    if (this.fatalError !== undefined) {
      throw new Error(`sim error during reset(): ${this.fatalError}`);
    }
  }

  private onGameUpdate(gu: GameUpdateViewData | ErrorUpdate): void {
    if ("errMsg" in gu) {
      this.fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
      return;
    }
    const winUpdates = gu.updates[GameUpdateType.Win];
    if (winUpdates.length > 0) {
      const winner = winUpdates[winUpdates.length - 1].winner;
      // ["player", clientID, ...] per GameImpl.makeWinner — the second
      // element is the client ID, not the player ID or a small ID. Team
      // and nation wins never name our single-player client, so they're
      // always an opponent win from the agent's point of view.
      const agentWon = winner !== undefined
        && winner[0] === "player"
        && winner[1] === AGENT_CLIENT_ID;
      this.winnerAgentID = agentWon ? AGENT_ID : "opponent";
    }
  }

  private pickSpawnTile(random: PseudoRandom): number {
    // Sampled from every qualifying tile via the episode's seeded PRNG, not
    // the first one found scanning from index 0 — the latter would spawn
    // every episode on a given map in the same corner regardless of seed,
    // baking a positional prior into the policy before training even starts.
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
    return candidates.length > 0 ? random.randElement(candidates) : -1;
  }
}
