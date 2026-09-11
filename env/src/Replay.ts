import fs from "fs";
import path from "path";
import type {
  GameConfig,
  PlayerRecord,
  Turn,
  Winner,
} from "../../OpenFrontIO/src/core/Schemas";
import { createPartialGameRecord } from "../../OpenFrontIO/src/core/Util";
import { simpleHash } from "../../OpenFrontIO/src/core/Util";
import type { ReplayOptions } from "./types";

/**
 * Phase-8 data replays. A recorded episode is a GameRecord-compatible JSON
 * file the upstream dev client plays back visually (see the phase-8 spec):
 * the engine is deterministic and every non-human player acts through
 * internal seeded executions, so the seed plus the agent's turns reproduce
 * the whole game. No protocol or schema change: recording is a local file
 * write on the Node side, enabled through environment variables.
 */

/** Engine commit the training sim runs. Stamped on every record so a
 *  future version check can tell what built it. */
export const TRAINING_ENGINE_COMMIT =
  "f51f165b947a92cc683ade3a72c3800300e86e61";

/** Record IDs must match /^[A-Za-z0-9]{8}$/ (engine ID schema), which our
 *  training IDs ("client0", free-form seeds) do not. Fixed replay identity
 *  for the agent; the game ID derives from the seed below. */
export const REPLAY_CLIENT_ID = "RLCLIENT";

/** The singleplayer-style player list a record carries. Bots and nations
 *  respawn from the config deterministically, so only the agent is named —
 *  exactly like a singleplayer lobby with one human. */
export function replayPlayers(
  clientID: string = REPLAY_CLIENT_ID,
  username: string = "RLAgent",
): PlayerRecord[] {
  return [{ clientID, username, clanTag: null, persistentID: null }];
}

/** Deterministic 8-char alphanumeric game ID derived from the episode seed.
 *  Informational only; the seed itself rides in the filename. */
export function replayGameID(seed: string): string {
  return (simpleHash(seed) >>> 0).toString(36).padStart(8, "0").slice(-8);
}

/** Deep clone converting bigints to numbers. Turn intents can carry bigint
 *  gold amounts, which JSON.stringify rejects and the record schema does
 *  not accept. Values stay bit-exact: game amounts fit well under 2^53. */
export function toJsonSafe<T>(value: T): T {
  if (typeof value === "bigint") return Number(value) as T;
  if (Array.isArray(value)) return value.map(toJsonSafe) as T;
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = toJsonSafe(v);
    return out as T;
  }
  return value;
}

export interface BuildRecordArgs {
  seed: string;
  gameConfig: GameConfig;
  /** All turns fed this episode, spawn phase included. */
  turns: Turn[];
  agentClientID: string;
  agentWon: boolean;
  startedAt: number;
  endedAt: number;
  /** State hashes keyed by tick, as collected from HashUpdates. Stamped
   *  onto the matching turns so the replay harness can verify sync. */
  hashes?: Map<number, number>;
}

/**
 * Build the record object. Reuses the engine's own createPartialGameRecord
 * (same empty-turn filtering as real archives) and adds the git commit the
 * client checks. Every intent's clientID is remapped to the replay identity
 * so the record validates; the sim resolves turns by content, which is
 * otherwise byte-identical to what training fed.
 */
export function buildGameRecord(args: BuildRecordArgs) {
  const turns: Turn[] = toJsonSafe(
    args.turns.map((turn) => ({
      ...turn,
      intents: turn.intents.map((intent) => ({
        ...intent,
        clientID: REPLAY_CLIENT_ID,
      })),
    })),
  );
  const winner: Winner =
    args.agentWon ? ["player", REPLAY_CLIENT_ID] : undefined;
  if (args.hashes !== undefined) {
    for (const [tick, hash] of args.hashes) {
      const turn = turns[tick];
      // turnNumber equals position by feeder construction; the guard keeps
      // a misaligned hash off the wrong turn rather than failing the write.
      if (turn !== undefined && turn.turnNumber === tick) turn.hash = hash;
    }
  }
  const record = createPartialGameRecord(
    replayGameID(args.seed),
    toJsonSafe(args.gameConfig),
    replayPlayers(),
    turns,
    args.startedAt,
    args.endedAt,
    winner,
  );
  return { ...record, gitCommit: TRAINING_ENGINE_COMMIT };
}

/** True when this episode index is due for recording. Episode 0 always
 *  records, so `every: 1` means every episode. Pure for tests. */
export function shouldRecordReplay(episodeIndex: number, every: number): boolean {
  return every >= 1 && episodeIndex % every === 0;
}

function safeFileStem(seed: string): string {
  return seed.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 64) || "unseeded";
}

/**
 * Write one record file. Never throws: recording must not break training,
 * so callers treat a returned undefined as "skipped or failed, logged".
 */
export function writeReplayFile(
  replay: ReplayOptions,
  seed: string,
  episodeIndex: number,
  terminal: string,
  record: unknown,
): string | undefined {
  try {
    fs.mkdirSync(replay.dir, { recursive: true });
    const name = `ep${episodeIndex}-${safeFileStem(seed)}-${terminal}.json`;
    const file = path.join(replay.dir, name);
    fs.writeFileSync(file, JSON.stringify(record));
    return file;
  } catch (e) {
    console.warn(`replay write failed, seed ${seed}: ${String(e)}`);
    return undefined;
  }
}

/** Read recording options from the environment (worker_main only). */
export function replayOptionsFromEnv(
  env: Record<string, string | undefined> = process.env,
): ReplayOptions | undefined {
  const dir = env.OPENFRONT_REPLAY_DIR;
  if (!dir) return undefined;
  const every = Number(env.OPENFRONT_REPLAY_EVERY ?? "10000");
  if (!Number.isInteger(every) || every < 1) {
    throw new Error(
      `OPENFRONT_REPLAY_EVERY must be a positive integer, got ${env.OPENFRONT_REPLAY_EVERY}`,
    );
  }
  return { dir, every };
}
