import path from "path";
import { GameMapType } from "../../../OpenFrontIO/src/core/game/Game";
import { ActionCodecImpl } from "../ActionCodec";
import {
  BehaviorConfig,
  DEFAULT_REWARD_WEIGHTS,
  KNOWN_BUILD_TYPES,
  RewardWeights,
} from "../BehaviorConfig";
import { GameEnv } from "../GameEnv";
import { ObservationEncoder } from "../ObservationEncoder";
import { replayOptionsFromEnv } from "../Replay";
import { Reward } from "../Reward";
import type { EnvConfig } from "../types";
import { WorkerServer } from "./WorkerServer";

export interface WorkerMainOptions {
  host: string;
  port: number | undefined;
  path: string | undefined;
  numEnvs: number;
  map: GameMapType;
  mapsDir: string;
  bots: number;
  ticksPerStep: number;
  tickCap: number;
  strictIllegalActions: boolean;
  /** Side-quest AI control UI: intent types to keep shut. */
  disabledIntents: number[];
  /** Side-quest AI control UI: buildable unit types to exclude. */
  disabledBuilds: string[];
  /** Side-quest AI control UI: shaping override, defaults are phase 1. */
  rewardWeights: RewardWeights;
}

export const DEFAULT_WORKER_OPTIONS: WorkerMainOptions = {
  host: "127.0.0.1",
  port: undefined,
  path: undefined,
  numEnvs: 1,
  map: GameMapType.World,
  mapsDir: path.resolve(process.cwd(), "../OpenFrontIO/resources/maps"),
  bots: 20,
  ticksPerStep: 10,
  tickCap: 6000,
  strictIllegalActions: true,
  disabledIntents: [],
  disabledBuilds: [],
  rewardWeights: { ...DEFAULT_REWARD_WEIGHTS },
};

/**
 * Parse the worker CLI. Exported for tests. Throws on bad input rather
 * than guessing, since a silently wrong map or tick cap would corrupt
 * training data far from where it was misconfigured.
 */
export function parseWorkerArgs(argv: string[]): WorkerMainOptions {
  // Fresh copies of the mutable defaults: the spread below would share
  // the array/object references with DEFAULT_WORKER_OPTIONS.
  const options: WorkerMainOptions = {
    ...DEFAULT_WORKER_OPTIONS,
    disabledIntents: [...DEFAULT_WORKER_OPTIONS.disabledIntents],
    disabledBuilds: [...DEFAULT_WORKER_OPTIONS.disabledBuilds],
    rewardWeights: { ...DEFAULT_WORKER_OPTIONS.rewardWeights },
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = (): string => {
      const value = argv[i + 1];
      if (value === undefined || value.startsWith("--")) {
        throw new Error(`${arg} needs a value`);
      }
      i++;
      return value;
    };
    const integer = (raw: string, name: string): number => {
      const value = Number(raw);
      if (!Number.isInteger(value)) throw new Error(`${name} must be an integer, got ${raw}`);
      return value;
    };
    switch (arg) {
      case "--host":
        options.host = next();
        break;
      case "--port":
        options.port = integer(next(), "port");
        break;
      case "--path":
        options.path = next();
        break;
      case "--num-envs":
        options.numEnvs = integer(next(), "num-envs");
        break;
      case "--map": {
        const name = next();
        const map = (GameMapType as Record<string, GameMapType>)[name];
        if (map === undefined) throw new Error(`unknown map ${name}`);
        options.map = map;
        break;
      }
      case "--maps-dir":
        options.mapsDir = next();
        break;
      case "--bots":
        options.bots = integer(next(), "bots");
        break;
      case "--ticks-per-step":
        options.ticksPerStep = integer(next(), "ticks-per-step");
        break;
      case "--tick-cap":
        options.tickCap = integer(next(), "tick-cap");
        break;
      case "--strict":
        options.strictIllegalActions = true;
        break;
      case "--lenient":
        options.strictIllegalActions = false;
        break;
      case "--disable-intent": {
        const value = integer(next(), "disable-intent");
        if (value < 1 || value > 13) {
          throw new Error(`disable-intent must be 1..13, got ${value}`);
        }
        options.disabledIntents = [...options.disabledIntents, value];
        break;
      }
      case "--disable-build": {
        const name = next();
        if (!(KNOWN_BUILD_TYPES as readonly string[]).includes(name)) {
          throw new Error(`unknown build type ${name}`);
        }
        options.disabledBuilds = [...options.disabledBuilds, name];
        break;
      }
      case "--reward-weights": {
        const parts = next().split(",");
        if (parts.length !== 5 && parts.length !== 6) {
          throw new Error("--reward-weights needs land,troop,gold,win,death[,kill]");
        }
        const values = parts.map((raw) => Number(raw));
        if (values.some((v) => !Number.isFinite(v))) {
          throw new Error(`--reward-weights must be finite numbers, got ${parts}`);
        }
        const [land, troop, gold, win, death] = values;
        // Old 5-value configs keep working: kill defaults to 0, which
        // makes the kill bonus vanish and restores phase-1 behavior.
        options.rewardWeights = { land, troop, gold, win, death, kill: values[5] ?? 0 };
        break;
      }
      case "--help":
      case "-h":
        printUsage();
        process.exit(0);
        break;
      default:
        throw new Error(`unknown flag ${arg}`);
    }
  }
  if (options.port === undefined && options.path === undefined) {
    throw new Error("one of --port or --path is required");
  }
  if (options.numEnvs < 1 || options.numEnvs > 1024) {
    throw new Error(`num-envs must be 1..1024, got ${options.numEnvs}`);
  }
  return options;
}

function printUsage(): void {
  console.log(`openfront worker: hosts M GameEnvs behind the bridge protocol.
Usage: tsx src/bridge/worker_main.ts --port 5001 [options]
  --host NAME        bind host, default 127.0.0.1
  --port N           TCP port to listen on (or --path for IPC)
  --path P           IPC path or Windows named pipe, wins over --port
  --num-envs M       envs hosted in this process, default 1
  --map NAME         GameMapType key such as World, default World
  --maps-dir DIR     engine resources/maps dir
  --bots N           default 20
  --ticks-per-step N default 10
  --tick-cap N       default 6000
  --strict           illegal actions fail loudly (default)
  --lenient          count illegal actions and retry as noop
  --disable-intent N keep one intent type shut, repeatable, 1..13 (side quest)
  --disable-build T  exclude one build type, repeatable, e.g. Factory
  --reward-weights L,T,G,W,D[,K]
                     shaping override as land,troop,gold,win,death
                     plus optional per-kill bonus, default 0`);
}

async function main(): Promise<void> {
  const options = parseWorkerArgs(process.argv.slice(2));
  const encoder = new ObservationEncoder();
  const behavior: BehaviorConfig = {
    disabledIntents: options.disabledIntents,
    disabledBuilds: options.disabledBuilds,
    rewardWeights: options.rewardWeights,
  };
  const codec = new ActionCodecImpl(behavior);
  const config: EnvConfig = {
    map: options.map,
    mapsDir: options.mapsDir,
    bots: options.bots,
    ticksPerStep: options.ticksPerStep,
    tickCap: options.tickCap,
    // Phase-8 replays arrive via environment, not the wire protocol, so the
    // bridge schema stays at version 1. Python sets these before spawning.
    replay: replayOptionsFromEnv(),
  };
  const server = new WorkerServer({
    numEnvs: options.numEnvs,
    createEnv: () => new GameEnv(config, encoder, codec, new Reward({ weights: behavior.rewardWeights })),
    obsSize: encoder.size,
    strictIllegalActions: options.strictIllegalActions,
    listen:
      options.path !== undefined
        ? { path: options.path }
        : { host: options.host, port: options.port },
  });
  const shutdown = (): void => {
    void server.close().finally(() => process.exit(0));
  };
  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
  await server.start();
  console.error(`worker listening: ${JSON.stringify(server.getAddress())}`);
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  (process.argv[1].endsWith("worker_main.ts") || process.argv[1].endsWith("worker_main.js"));
if (invokedDirectly) {
  main().catch((err) => {
    console.error(err instanceof Error ? (err.stack ?? err.message) : String(err));
    process.exit(1);
  });
}
