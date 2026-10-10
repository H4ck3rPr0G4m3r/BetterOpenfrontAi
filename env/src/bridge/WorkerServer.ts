import net from "net";
import type { SeatStep } from "../GameEnv";
import type { Action } from "../types";
import {
  decodeFramePayload,
  decodeReset,
  decodeStep,
  encodeError,
  encodeFrame,
  encodeHello,
  encodeRows,
  FLAG_DONE,
  FLAG_FRESH,
  FLAG_NEXT,
  FLAG_VALID,
  FrameReader,
  MAX_SEED_BYTES,
  MessageType,
  ProtocolError,
  ProtocolErrorCode,
  Row,
} from "./Protocol";

/** The slice of GameEnv the server drives; tests substitute fakes. */
export interface WorkerGame {
  readonly numSeats: number;
  readonly gameOver: boolean;
  readonly ticks: number;
  reset(seed: string): Promise<SeatStep[]>;
  step(actions: Action[]): SeatStep[];
  abort(): SeatStep[];
}

export interface WorkerServerOptions {
  games: number;
  seats: number;
  createGame: (index: number) => WorkerGame;
  host?: string;
  port?: number;
  logger?: Pick<Console, "info" | "warn" | "error">;
}

interface GameSlot {
  game: WorkerGame;
  seed: string;
  baseSeed: string;
  resets: number;
}

/**
 * Node half of the bridge. Hosts `games` GameEnvs of `seats` seats each in
 * one process and speaks the v2 protocol to one Python client at a time
 * (a second connection is refused). Finished games reset inside the same
 * STEP reply. A protocol violation sends ERROR and drops the connection:
 * silent desync would corrupt training data. Sim errors are not protocol
 * errors — GameEnv turns them into sim_error terminals.
 */
export class WorkerServer {
  private server: net.Server | null = null;
  private client: net.Socket | null = null;
  private expectedSeq = 1;
  private readonly slots: GameSlot[];
  private queue: Promise<void> = Promise.resolve();
  private readonly logger: Pick<Console, "info" | "warn" | "error">;
  readonly stats = { steps: 0, games: 0, simErrors: 0 };

  constructor(private readonly options: WorkerServerOptions) {
    if (!Number.isInteger(options.games) || options.games < 1) throw new Error(`games must be >= 1`);
    this.logger = options.logger ?? console;
    this.slots = Array.from({ length: options.games }, (_, i) => {
      const game = options.createGame(i);
      if (game.numSeats !== options.seats) throw new Error(`game ${i} has ${game.numSeats} seats, expected ${options.seats}`);
      return { game, seed: "", baseSeed: "", resets: 0 };
    });
  }

  get rows(): number {
    return this.options.games * this.options.seats;
  }

  async start(): Promise<{ host: string; port: number }> {
    this.server = net.createServer((socket) => this.onConnection(socket));
    await new Promise<void>((resolve, reject) => {
      this.server!.once("error", reject);
      this.server!.listen(this.options.port ?? 0, this.options.host ?? "127.0.0.1", () => resolve());
    });
    const addr = this.server.address() as net.AddressInfo;
    return { host: addr.address, port: addr.port };
  }

  async close(): Promise<void> {
    this.client?.destroy();
    this.client = null;
    const server = this.server;
    this.server = null;
    if (server !== null) await new Promise<void>((resolve) => server.close(() => resolve()));
  }

  private onConnection(socket: net.Socket): void {
    if (this.client !== null) {
      socket.destroy();
      return;
    }
    socket.setNoDelay(true);
    this.client = socket;
    this.expectedSeq = 1;
    const reader = new FrameReader();
    socket.on("data", (chunk: Buffer) => {
      const bytes = new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
      this.queue = this.queue
        .then(async () => {
          if (this.client !== socket) return;
          let payloads: Uint8Array[];
          try {
            payloads = reader.push(bytes);
          } catch (err) {
            this.fail(socket, this.expectedSeq, err);
            return;
          }
          for (const payload of payloads) {
            await this.onFrame(socket, payload);
            if (this.client !== socket) return;
          }
        })
        .catch((err) => this.logger.error(`bridge dispatch failed: ${describe(err)}`));
    });
    const drop = () => {
      if (this.client === socket) this.client = null;
    };
    socket.on("close", drop);
    socket.on("error", drop);
    socket.write(encodeFrame(0, MessageType.Hello, encodeHello(this.options.games, this.options.seats)));
  }

  private async onFrame(socket: net.Socket, payload: Uint8Array): Promise<void> {
    let seq = this.expectedSeq;
    try {
      const frame = decodeFramePayload(payload);
      seq = frame.seq;
      if (frame.seq !== this.expectedSeq) {
        throw new ProtocolError(ProtocolErrorCode.BadSequence, `expected seq ${this.expectedSeq}, got ${frame.seq}`);
      }
      this.expectedSeq = (this.expectedSeq + 1) >>> 0 || 1;
      switch (frame.msgType) {
        case MessageType.Reset:
          socket.write(encodeFrame(seq, MessageType.Rows, encodeRows(await this.onReset(frame.body))));
          break;
        case MessageType.Step:
          socket.write(encodeFrame(seq, MessageType.Rows, encodeRows(await this.onStep(frame.body))));
          break;
        case MessageType.Close:
          socket.end();
          if (this.client === socket) this.client = null;
          break;
        default:
          throw new ProtocolError(ProtocolErrorCode.BadMessage, `unknown message type ${frame.msgType}`);
      }
    } catch (err) {
      this.fail(socket, seq, err);
    }
  }

  private async onReset(body: Uint8Array): Promise<Row[]> {
    const seeds = decodeReset(body, this.options.games);
    const rows: Row[] = [];
    for (let g = 0; g < this.slots.length; g++) {
      const slot = this.slots[g];
      slot.seed = slot.baseSeed = seeds[g];
      slot.resets = 0;
      const fresh = await slot.game.reset(seeds[g]);
      rows.push(...fresh.map((s) => toRow(s, slot.game.ticks, FLAG_NEXT | FLAG_FRESH)));
    }
    return rows;
  }

  private async onStep(body: Uint8Array): Promise<Row[]> {
    const req = decodeStep(body, this.rows, this.options.games);
    const rows: Row[] = [];
    const S = this.options.seats;
    for (let g = 0; g < this.slots.length; g++) {
      const slot = this.slots[g];
      if (slot.baseSeed === "") throw new ProtocolError(ProtocolErrorCode.BadMessage, "STEP before RESET");
      let steps: SeatStep[];
      if (slot.game.gameOver) {
        throw new ProtocolError(ProtocolErrorCode.Internal, `game ${g} is over but was not reset`);
      } else if (req.abort.has(g)) {
        steps = slot.game.abort();
      } else {
        try {
          steps = slot.game.step(req.actions.slice(g * S, (g + 1) * S));
        } catch (err) {
          if (err instanceof RangeError) {
            throw new ProtocolError(ProtocolErrorCode.IllegalAction, `game ${g}: ${err.message}`);
          }
          throw err;
        }
      }
      this.stats.steps++;
      const stepRows = steps.map((s) =>
        toRow(s, slot.game.ticks, (s.active ? FLAG_VALID : 0) | (s.done ? FLAG_DONE : 0) | (s.active && !s.done ? FLAG_NEXT : 0)),
      );
      if (slot.game.gameOver) {
        this.stats.games++;
        if (steps.some((s) => s.terminal === "sim_error")) this.stats.simErrors++;
        slot.seed = nextSeed(slot, req.seeds[g]);
        const fresh = await slot.game.reset(slot.seed);
        fresh.forEach((s, i) => {
          const r = stepRows[i];
          stepRows[i] = {
            ...toRow(s, slot.game.ticks, (r.flags & (FLAG_VALID | FLAG_DONE)) | FLAG_NEXT | FLAG_FRESH),
            reward: r.reward,
            terminal: r.terminal,
            tiles: r.tiles,
            troops: r.troops,
            kills: r.kills,
            placement: r.placement,
          };
        });
      }
      rows.push(...stepRows);
    }
    return rows;
  }

  private fail(socket: net.Socket, seq: number, err: unknown): void {
    const code = err instanceof ProtocolError ? err.errorCode : ProtocolErrorCode.Internal;
    this.logger.error(`bridge error code=${code}: ${describe(err)}`);
    try {
      socket.write(encodeFrame(seq >>> 0, MessageType.Error, encodeError(code, describe(err))));
    } catch {
      // socket already unusable; destroying is the signal
    }
    socket.destroy();
    if (this.client === socket) this.client = null;
  }
}

function toRow(s: SeatStep, ticks: number, flags: number): Row {
  return {
    raster: s.obs.raster,
    features: s.obs.features,
    masks: s.masks.flat,
    reward: s.reward,
    tiles: s.tiles,
    troops: s.troops,
    kills: s.kills,
    placement: s.placement,
    ticks,
    terminal: s.terminal,
    flags,
  };
}

/** Next game seed: the client's, or "<base>#<n>" derived deterministically. */
function nextSeed(slot: GameSlot, requested: string): string {
  slot.resets++;
  if (requested !== "") return requested;
  const suffix = `#${slot.resets}`;
  let base = slot.baseSeed;
  while (new TextEncoder().encode(base + suffix).length > MAX_SEED_BYTES) base = [...base].slice(0, -1).join("");
  return base + suffix;
}

function describe(err: unknown): string {
  return err instanceof Error ? (err.stack ?? err.message) : String(err);
}
