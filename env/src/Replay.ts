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
 * Data replays. A recorded game is a GameRecord-compatible JSON
 * file the upstream dev client plays back visually: the engine is
 * deterministic and every non-human player acts through internal seeded
 * executions, so the seed plus the seats' turns reproduce the whole game.
 * Recording is a local file write on the Node side, enabled through
 * environment variables.
 */

/** Engine commit the training sim runs. Stamped on every record so a
 *  future version check can tell what built it. */
export const TRAINING_ENGINE_COMMIT =
  "f51f165b947a92cc683ade3a72c3800300e86e61";

/** Seat i's client ID, used both in training and in records. Record IDs
 *  must match /^[A-Za-z0-9]{8}$/ (engine ID schema), so training uses
 *  record-valid IDs directly and recorded turns need no remapping. */
export function seatClientID(seat: number): string {
  if (!Number.isInteger(seat) || seat < 0 || seat > 99) throw new Error(`bad seat ${seat}`);
  return `RLSEAT${String(seat).padStart(2, "0")}`;
}

/** The lobby player list a record carries: one human per seat, in seat
 *  order (the order IDs are drawn in). Bots and nations respawn from the
 *  config deterministically. */
export function replayPlayers(seats: number): PlayerRecord[] {
  return Array.from({ length: seats }, (_, i) => ({
    clientID: seatClientID(i),
    username: `RLAgent${i}`,
    clanTag: null,
    persistentID: null,
  }));
}

/** Deterministic 8-char alphanumeric game ID derived from the seed. The
 *  whole game (nations, executor, seat IDs) derives from it. */
export function replayGameID(seed: string): string {
  return (simpleHash(seed) >>> 0).toString(36).padStart(8, "0").slice(-8);
}

/** Deep clone converting bigints to numbers (JSON and the record schema
 *  reject bigints; game amounts fit well under 2^53). */
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
  /** All turns fed this game, spawn phase included. */
  turns: Turn[];
  seats: number;
  /** Winning seat's client ID, or null. */
  winnerClientID: string | null;
  startedAt: number;
  endedAt: number;
  /** State hashes keyed by tick, stamped onto turns for sync checks. */
  hashes?: Map<number, number>;
}

/** Build the record with the engine's own createPartialGameRecord. */
export function buildGameRecord(args: BuildRecordArgs) {
  const turns: Turn[] = toJsonSafe(args.turns.map((turn) => ({ ...turn, intents: [...turn.intents] })));
  const winner: Winner = args.winnerClientID !== null ? ["player", args.winnerClientID] : undefined;
  if (args.hashes !== undefined) {
    for (const [tick, hash] of args.hashes) {
      const turn = turns[tick];
      // turnNumber equals position by feeder construction.
      if (turn !== undefined && turn.turnNumber === tick) turn.hash = hash;
    }
  }
  const record = createPartialGameRecord(
    replayGameID(args.seed),
    toJsonSafe(args.gameConfig),
    replayPlayers(args.seats),
    turns,
    args.startedAt,
    args.endedAt,
    winner,
  );
  return { ...record, gitCommit: TRAINING_ENGINE_COMMIT };
}

/** True when this game index is due for recording. Game 0 always records,
 *  so `every: 1` means every game. */
export function shouldRecordReplay(index: number, every: number): boolean {
  return every >= 1 && index % every === 0;
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
  index: number,
  outcome: string,
  record: unknown,
): string | undefined {
  try {
    fs.mkdirSync(replay.dir, { recursive: true });
    const name = `game${index}-${safeFileStem(seed)}-${outcome}.json`;
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
