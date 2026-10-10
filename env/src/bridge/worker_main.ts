import path from "path";
import { GameMapType } from "../../../OpenFrontIO/src/core/game/Game";
import { ActionCodecImpl } from "../ActionCodec";
import { BehaviorConfig, DEFAULT_TROOP_RESERVE, normalizeRewardWeights, RewardWeights } from "../BehaviorConfig";
import { GameEnv } from "../GameEnv";
import { BUILD_TYPES, NUM_INTENTS } from "../Layout";
import { ObservationEncoder } from "../ObservationEncoder";
import { replayOptionsFromEnv } from "../Replay";
import type { EnvConfig } from "../types";
import { WorkerServer } from "./WorkerServer";

export interface WorkerOptions {
  host: string;
  port: number;
  games: number;
  seats: number;
  map: GameMapType;
  mapsDir: string;
  bots: number;
  nations: "default" | "disabled";
  ticksPerStep: number;
  tickCap: number;
  disabledIntents: number[];
  disabledBuilds: string[];
  allowDonations: boolean;
  troopReserve: number;
  closeSpawn: number;
  reward: RewardWeights;
}

const USAGE = `openfront worker: hosts G games x S seats behind the bridge protocol.
Usage: tsx src/bridge/worker_main.ts --port N [options]
  --host NAME          bind host (127.0.0.1)
  --port N             TCP port (required)
  --games G            games hosted by this process (1)
  --seats S            agent seats per game, 1..8 (1)
  --map NAME           GameMapType key (World)
  --maps-dir DIR       engine resources/maps dir
  --bots N             engine bots (20)
  --nations MODE       default | disabled
  --ticks-per-step N   (10)
  --tick-cap N         post-spawn tick cap (6000)
  --disable-intent K   shut one intent type, repeatable (1..13)
  --disable-build T    exclude one build type, repeatable (e.g. Factory)
  --allow-donations    open gold/troop donations (off by default)
  --troop-reserve F    share of max troops attacks must leave home (0.1)
  --close-spawn P      chance a multi-seat game spawns its seats close (0)
  --reward JSON        reward weight overrides, e.g. {"land":1,"kill":0.5}`;

/** Parse the CLI. Throws on bad input: a silently wrong map or tick cap
 *  would corrupt training data far from where it was misconfigured. */
export function parseWorkerArgs(argv: string[]): WorkerOptions {
  const o: WorkerOptions = {
    host: "127.0.0.1",
    port: -1,
    games: 1,
    seats: 1,
    map: GameMapType.World,
    mapsDir: path.resolve(process.cwd(), "../OpenFrontIO/resources/maps"),
    bots: 20,
    nations: "default",
    ticksPerStep: 10,
    tickCap: 6000,
    disabledIntents: [],
    disabledBuilds: [],
    allowDonations: false,
    troopReserve: DEFAULT_TROOP_RESERVE,
    closeSpawn: 0,
    reward: normalizeRewardWeights(),
  };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    const value = (): string => {
      const v = argv[++i];
      if (v === undefined || v.startsWith("--")) throw new Error(`${flag} needs a value`);
      return v;
    };
    const fraction = (): number => {
      const raw = value();
      const v = Number(raw);
      if (!Number.isFinite(v) || v < 0 || v > 1) throw new Error(`${flag} must be a number 0..1, got ${raw}`);
      return v;
    };
    const int = (min: number, max: number): number => {
      const raw = value();
      const v = Number(raw);
      if (!Number.isInteger(v) || v < min || v > max) throw new Error(`${flag} must be an integer ${min}..${max}, got ${raw}`);
      return v;
    };
    switch (flag) {
      case "--host": o.host = value(); break;
      case "--port": o.port = int(1, 65535); break;
      case "--games": o.games = int(1, 256); break;
      case "--seats": o.seats = int(1, 8); break;
      case "--map": {
        const name = value();
        const map = (GameMapType as Record<string, GameMapType>)[name];
        if (map === undefined) throw new Error(`unknown map ${name}`);
        o.map = map;
        break;
      }
      case "--maps-dir": o.mapsDir = value(); break;
      case "--bots": o.bots = int(0, 400); break;
      case "--nations": {
        const v = value();
        if (v !== "default" && v !== "disabled") throw new Error(`nations must be default or disabled, got ${v}`);
        o.nations = v;
        break;
      }
      case "--ticks-per-step": o.ticksPerStep = int(1, 1000); break;
      case "--tick-cap": o.tickCap = int(1, 1_000_000); break;
      case "--disable-intent": o.disabledIntents.push(int(1, NUM_INTENTS - 1)); break;
      case "--disable-build": {
        const v = value();
        if (!(BUILD_TYPES as readonly string[]).includes(v)) throw new Error(`unknown build type ${v}`);
        o.disabledBuilds.push(v);
        break;
      }
      case "--allow-donations": o.allowDonations = true; break;
      case "--troop-reserve": o.troopReserve = fraction(); break;
      case "--close-spawn": o.closeSpawn = fraction(); break;
      case "--reward": {
        let parsed: unknown;
        try {
          parsed = JSON.parse(value());
        } catch {
          throw new Error("--reward must be a JSON object");
        }
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
          throw new Error("--reward must be a JSON object");
        }
        o.reward = normalizeRewardWeights(parsed as Record<string, unknown>);
        break;
      }
      case "--help":
      case "-h":
        console.log(USAGE);
        process.exit(0);
        break;
      default:
        throw new Error(`unknown flag ${flag}`);
    }
  }
  if (o.port < 0) throw new Error("--port is required");
  return o;
}

async function main(): Promise<void> {
  const o = parseWorkerArgs(process.argv.slice(2));
  const behavior: BehaviorConfig = {
    disabledIntents: o.disabledIntents,
    disabledBuilds: o.disabledBuilds,
    allowDonations: o.allowDonations,
    troopReserve: o.troopReserve,
  };
  const codec = new ActionCodecImpl(behavior);
  const encoder = new ObservationEncoder(o.tickCap);
  const config: EnvConfig = {
    map: o.map,
    mapsDir: o.mapsDir,
    bots: o.bots,
    nations: o.nations,
    ticksPerStep: o.ticksPerStep,
    tickCap: o.tickCap,
    seats: o.seats,
    closeSpawn: o.closeSpawn,
    // Replays are enabled through the environment; Python sets it before spawning.
    replay: replayOptionsFromEnv(),
  };
  const server = new WorkerServer({
    games: o.games,
    seats: o.seats,
    host: o.host,
    port: o.port,
    createGame: () => new GameEnv(config, codec, o.reward, encoder),
  });
  const shutdown = (): void => void server.close().finally(() => process.exit(0));
  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
  const addr = await server.start();
  console.error(`worker listening on ${addr.host}:${addr.port} (${o.games} games x ${o.seats} seats)`);
}

const invokedDirectly = process.argv[1]?.endsWith("worker_main.ts") || process.argv[1]?.endsWith("worker_main.js");
if (invokedDirectly) {
  main().catch((err) => {
    console.error(err instanceof Error ? (err.stack ?? err.message) : String(err));
    process.exit(1);
  });
}
