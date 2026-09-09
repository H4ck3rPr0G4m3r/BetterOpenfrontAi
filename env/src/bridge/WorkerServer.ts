import net from "net";
import {
  FrameReader,
  MASK_BUILD_SITES,
  MASK_INTENT_TYPES,
  MASK_RATIO_BUCKETS,
  MASK_SPATIAL_CELLS,
  MASK_TARGET_SLOTS,
  MAX_SEED_BYTES,
  MessageType,
  ProtocolError,
  ProtocolErrorCode,
  SCHEMA_VERSION,
  decodeFramePayload,
  decodeResetBody,
  decodeStepBody,
  encodeErrorBody,
  encodeFrame,
  encodeHelloBody,
  encodeResetReplyBody,
  encodeStepReplyBody,
  flattenMasks,
  terminalToCode,
} from "./Protocol";
import type { Action, MaskBuffers, StepResult } from "../types";

/**
 * One episode slot the worker drives. GameEnv satisfies this structurally,
 * and the shape stays small on purpose so tests can use fakes without the
 * engine. `ticks` and `currentMasks` exist because RESET replies need both
 * while no StepResult exists yet.
 */
export interface WorkerEnv {
  reset(seed: string): Promise<Float32Array>;
  step(action: Action): StepResult;
  readonly ticks: number;
  currentMasks(): MaskBuffers;
  close(): void;
}

export interface WorkerListen {
  host?: string;
  port?: number;
  /** IPC path or Windows named pipe. Takes precedence over host/port. */
  path?: string;
}

export interface WorkerServerOptions {
  numEnvs: number;
  createEnv: (index: number) => WorkerEnv;
  /** Float count per observation, advertised in HELLO and validated. */
  obsSize: number;
  /**
   * True means an out of range action head ends the connection with an
   * error. False counts it and retries the env as noop. Test and dev runs
   * want the loud failure, production runs want the counter. Spec section 7.
   */
  strictIllegalActions?: boolean;
  listen?: WorkerListen;
  logger?: Pick<Console, "debug" | "info" | "warn" | "error">;
}

export interface WorkerStats {
  episodes: number;
  steps: number;
  simErrors: number;
  illegalDrops: number;
}

interface EnvSlot {
  env: WorkerEnv;
  seed: string;
  baseSeed: string;
  episodes: number;
  steps: number;
  history: Action[];
}

const NOOP_ACTION: Action = {
  intentType: 0,
  target: 0,
  ratioBucket: 0,
  spatialCell: 0,
  buildSite: 0,
};

/** History entries kept per env for the sim_error log line. Episodes are a few hundred steps. */
const MAX_HISTORY = 2048;

/**
 * Node half of the bridge, spec section 5.6. Hosts M envs in one process,
 * speaks one socket to Python, and owns auto reset so a done reply already
 * carries the next episode. One client at a time. A second connection is
 * destroyed immediately.
 *
 * Error handling follows section 7. A caught simulation failure ends its
 * episode as sim_error with the seed and action history logged, never
 * crashing the worker. A protocol violation is fatal: an ERROR frame goes
 * out and the connection is destroyed, since silent desync would corrupt
 * training data.
 */
export class WorkerServer {
  private readonly numEnvs: number;
  private readonly createEnv: (index: number) => WorkerEnv;
  private readonly obsSize: number;
  private readonly strictIllegalActions: boolean;
  private readonly listen: WorkerListen;
  private readonly logger: Pick<Console, "debug" | "info" | "warn" | "error">;

  private server: net.Server | null = null;
  private client: net.Socket | null = null;
  private expectedSeq = 1;
  private slots: EnvSlot[] = [];
  private processing: Promise<void> = Promise.resolve();
  private stats: WorkerStats = { episodes: 0, steps: 0, simErrors: 0, illegalDrops: 0 };

  constructor(options: WorkerServerOptions) {
    if (!Number.isInteger(options.numEnvs) || options.numEnvs < 1 || options.numEnvs > 1024) {
      throw new Error(`numEnvs must be 1..1024, got ${options.numEnvs}`);
    }
    if (!Number.isInteger(options.obsSize) || options.obsSize < 1) {
      throw new Error(`obsSize must be a positive integer, got ${options.obsSize}`);
    }
    this.numEnvs = options.numEnvs;
    this.createEnv = options.createEnv;
    this.obsSize = options.obsSize;
    this.strictIllegalActions = options.strictIllegalActions ?? true;
    this.listen = options.listen ?? {};
    this.logger = options.logger ?? console;
    for (let i = 0; i < this.numEnvs; i++) {
      this.slots.push({
        env: this.createEnv(i),
        seed: "",
        baseSeed: "",
        episodes: 0,
        steps: 0,
        history: [],
      });
    }
  }

  async start(): Promise<void> {
    if (this.server !== null) return;
    this.server = net.createServer((socket) => this.onConnection(socket));
    await new Promise<void>((resolve, reject) => {
      const onError = (err: Error): void => {
        this.server?.off("listening", onListening);
        reject(err);
      };
      const onListening = (): void => {
        this.server?.off("error", onError);
        resolve();
      };
      this.server?.once("error", onError);
      this.server?.once("listening", onListening);
      if (this.listen.path !== undefined) {
        this.server?.listen(this.listen.path);
      } else {
        this.server?.listen(this.listen.port ?? 0, this.listen.host ?? "127.0.0.1");
      }
    });
  }

  getAddress(): { host: string; port: number } | { path: string } {
    if (this.server === null) {
      throw new Error("getAddress before start()");
    }
    const addr = this.server.address();
    if (addr === null || typeof addr === "string") {
      if (this.listen.path !== undefined) return { path: this.listen.path };
      throw new Error("server is not listening");
    }
    return { host: addr.address, port: addr.port };
  }

  getStats(): WorkerStats {
    return { ...this.stats };
  }

  async close(): Promise<void> {
    this.client?.destroy();
    this.client = null;
    if (this.server !== null) {
      const server = this.server;
      this.server = null;
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
    for (const slot of this.slots) {
      try {
        slot.env.close();
      } catch (err) {
        this.logger.warn(`env close threw: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  private onConnection(socket: net.Socket): void {
    if (this.client !== null) {
      socket.destroy();
      return;
    }
    this.client = socket;
    this.expectedSeq = 1;
    // The session is captured per connection. Chunks are queued behind
    // async resets and steps, so a chunk must never be processed against
    // whatever client and reader happen to be current when its turn comes:
    // a disconnected socket's leftovers would otherwise desync the next
    // client's session.
    const session = { socket, reader: new FrameReader() };
    socket.on("data", (chunk: Buffer) => {
      if (this.client !== socket) return;
      const bytes = new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
      this.processing = this.processing
        .then(() => this.onData(session, bytes))
        .catch((err) => {
          this.logger.error(
            `bridge dispatch failed: ${err instanceof Error ? err.message : String(err)}`,
          );
        });
    });
    socket.on("close", () => {
      if (this.client === socket) this.client = null;
    });
    socket.on("error", () => {
      if (this.client === socket) this.client = null;
    });

    const hello = encodeHelloBody({
      version: SCHEMA_VERSION,
      numEnvs: this.numEnvs,
      obsSize: this.obsSize,
      intentTypes: MASK_INTENT_TYPES,
      targetSlots: MASK_TARGET_SLOTS,
      ratioBuckets: MASK_RATIO_BUCKETS,
      spatialCells: MASK_SPATIAL_CELLS,
      buildSites: MASK_BUILD_SITES,
    });
    socket.write(encodeFrame(0, MessageType.Hello, hello));
  }

  private async onData(
    session: { socket: net.Socket; reader: FrameReader },
    chunk: Uint8Array,
  ): Promise<void> {
    const { socket, reader } = session;
    if (this.client !== socket) return;
    let payloads: Uint8Array[];
    try {
      payloads = reader.push(chunk);
    } catch (err) {
      this.fail(socket, this.expectedSeq, ProtocolErrorCode.BadLength, describeError(err));
      return;
    }
    for (const payload of payloads) {
      await this.onFrame(socket, payload);
      if (this.client !== socket) return;
    }
  }

  private async onFrame(socket: net.Socket, payload: Uint8Array): Promise<void> {
    let frame;
    try {
      frame = decodeFramePayload(payload);
    } catch (err) {
      this.fail(socket, this.expectedSeq, ProtocolErrorCode.BadLength, describeError(err));
      return;
    }
    if (frame.seq !== this.expectedSeq) {
      this.fail(
        socket,
        frame.seq,
        ProtocolErrorCode.BadSequence,
        `expected seq ${this.expectedSeq}, got ${frame.seq}`,
      );
      return;
    }
    this.expectedSeq = (this.expectedSeq + 1) >>> 0;
    if (this.expectedSeq === 0) this.expectedSeq = 1;

    try {
      switch (frame.msgType) {
        case MessageType.Reset:
          await this.onReset(socket, frame.seq, frame.body);
          break;
        case MessageType.Step:
          await this.onStep(socket, frame.seq, frame.body);
          break;
        case MessageType.Close:
          if (frame.body.length !== 0) {
            throw new ProtocolError(ProtocolErrorCode.BadLength, "close takes no body");
          }
          socket.end();
          if (this.client === socket) this.client = null;
          break;
        default:
          throw new ProtocolError(
            ProtocolErrorCode.BadMessage,
            `unknown message type ${frame.msgType}`,
          );
      }
    } catch (err) {
      if (err instanceof ProtocolError) {
        this.fail(socket, frame.seq, err.errorCode, err.message);
        return;
      }
      this.fail(socket, frame.seq, ProtocolErrorCode.Internal, describeError(err));
    }
  }

  private async onReset(socket: net.Socket, seq: number, body: Uint8Array): Promise<void> {
    const { version, seeds } = decodeResetBody(body);
    if (version !== SCHEMA_VERSION) {
      throw new ProtocolError(
        ProtocolErrorCode.VersionMismatch,
        `client schema ${version}, server ${SCHEMA_VERSION}`,
      );
    }
    if (seeds.length !== this.numEnvs) {
      throw new ProtocolError(
        ProtocolErrorCode.BadMessage,
        `reset covers ${seeds.length} envs, server hosts ${this.numEnvs}`,
      );
    }
    const obsList: Float32Array[] = [];
    const ticks: number[] = [];
    const masks: Uint8Array[] = [];
    for (let i = 0; i < this.numEnvs; i++) {
      const slot = this.slots[i];
      const obs = await slot.env.reset(seeds[i]);
      if (obs.length !== this.obsSize) {
        throw new ProtocolError(
          ProtocolErrorCode.Internal,
          `env ${i} obs holds ${obs.length} floats, expected ${this.obsSize}`,
        );
      }
      slot.seed = seeds[i];
      slot.baseSeed = seeds[i];
      slot.episodes = 0;
      slot.steps = 0;
      slot.history = [];
      obsList.push(obs);
      ticks.push(slot.env.ticks);
      masks.push(flattenMasks(slot.env.currentMasks()));
    }
    this.logger.debug(`bridge reset seq=${seq} seeds=${seeds.join(",")}`);
    socket.write(
      encodeFrame(seq, MessageType.ResetReply, encodeResetReplyBody(obsList, ticks, masks, this.obsSize)),
    );
  }

  private async onStep(socket: net.Socket, seq: number, body: Uint8Array): Promise<void> {
    const { version, actions, nextSeeds } = decodeStepBody(body, this.numEnvs);
    if (version !== SCHEMA_VERSION) {
      throw new ProtocolError(
        ProtocolErrorCode.VersionMismatch,
        `client schema ${version}, server ${SCHEMA_VERSION}`,
      );
    }
    if (actions.length !== this.numEnvs || nextSeeds.length !== this.numEnvs) {
      throw new ProtocolError(ProtocolErrorCode.BadMessage, "step covers wrong env count");
    }
    for (const slot of this.slots) {
      if (slot.baseSeed === "") {
        throw new ProtocolError(ProtocolErrorCode.BadMessage, "step arrived before reset");
      }
    }

    const obsList: Float32Array[] = [];
    const rewards: number[] = [];
    const done: boolean[] = [];
    const terminalCodes: number[] = [];
    const ticks: number[] = [];
    const masks: Uint8Array[] = [];

    for (let i = 0; i < this.numEnvs; i++) {
      const slot = this.slots[i];
      let result: StepResult;
      try {
        result = slot.env.step(actions[i]);
      } catch (err) {
        if (err instanceof RangeError) {
          this.stats.illegalDrops += 1;
          this.logger.warn(
            `env ${i} illegal action dropped (seed ${slot.seed}): ${describeError(err)}`,
          );
          if (this.strictIllegalActions) {
            throw new ProtocolError(
              ProtocolErrorCode.IllegalAction,
              `env ${i}: ${describeError(err)}`,
            );
          }
          result = slot.env.step(NOOP_ACTION);
        } else {
          throw err;
        }
      }
      slot.steps += 1;
      this.stats.steps += 1;
      pushHistory(slot, actions[i]);

      if (result.done) {
        this.stats.episodes += 1;
        slot.episodes += 1;
        const terminal = result.info.terminalReason;
        if (terminal === "sim_error") {
          this.stats.simErrors += 1;
          this.logger.warn(
            `env ${i} sim_error seed=${slot.seed} ticks=${result.info.ticks} ` +
              `history=${JSON.stringify(slot.history)}`,
          );
        }
        const resetSeed = pickResetSeed(slot, nextSeeds[i]);
        const freshObs = await slot.env.reset(resetSeed);
        if (freshObs.length !== this.obsSize) {
          throw new ProtocolError(
            ProtocolErrorCode.Internal,
            `env ${i} obs holds ${freshObs.length} floats, expected ${this.obsSize}`,
          );
        }
        slot.seed = resetSeed;
        slot.history = [];
        obsList.push(freshObs);
        rewards.push(result.reward);
        done.push(true);
        terminalCodes.push(terminalToCode(terminal));
        ticks.push(slot.env.ticks);
        masks.push(flattenMasks(slot.env.currentMasks()));
      } else {
        obsList.push(result.obs);
        rewards.push(result.reward);
        done.push(false);
        terminalCodes.push(terminalToCode(result.info.terminalReason));
        ticks.push(result.info.ticks);
        masks.push(flattenMasks(result.info.masks));
      }
    }
    socket.write(
      encodeFrame(
        seq,
        MessageType.StepReply,
        encodeStepReplyBody(obsList, rewards, done, terminalCodes, ticks, masks, this.obsSize),
      ),
    );
  }

  private fail(socket: net.Socket, seq: number, code: ProtocolErrorCode, message: string): void {
    try {
      socket.write(encodeFrame(seq >>> 0, MessageType.Error, encodeErrorBody(code, message)));
    } catch {
      // The socket is already unusable. Destroying below is the signal.
    }
    this.logger.error(`bridge error code=${code}: ${message}`);
    socket.destroy();
    if (this.client === socket) this.client = null;
  }
}

function pushHistory(slot: EnvSlot, action: Action): void {
  slot.history.push({ ...action });
  if (slot.history.length > MAX_HISTORY) {
    slot.history.splice(0, slot.history.length - MAX_HISTORY);
  }
}

function pickResetSeed(slot: EnvSlot, nextSeed: string): string {
  if (nextSeed !== "") return nextSeed;
  const suffix = `#auto${slot.episodes}`;
  const encoder = new TextEncoder();
  let base = slot.baseSeed;
  // Byte length, not string length: non-ASCII seeds are wider in UTF-8
  // than .length suggests. Trimming whole code points keeps the seed valid.
  while (encoder.encode(base).length > MAX_SEED_BYTES - suffix.length && base !== "") {
    base = [...base].slice(0, -1).join("");
  }
  return `${base}${suffix}`;
}

function describeError(err: unknown): string {
  return err instanceof Error ? (err.stack ?? err.message) : String(err);
}
