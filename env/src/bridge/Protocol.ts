import type { Action } from "../types";
import type { TerminalReason } from "../Reward";

/**
 * Binary bridge protocol, spec section 5.6.
 *
 * This file is the single shared schema description. The Python mirror
 * `train/openfront_gym/protocol.py` implements the same layout byte for
 * byte. HELLO carries a schema version so the two ends fail loudly instead
 * of silently drifting, and every RESET and STEP body repeats it.
 *
 * Framing, both directions over a local socket:
 *   [u32LE payloadLen][payload]
 *   payload = [u32LE seq][u8 msgType][body...]
 *
 * The client (Python) numbers requests from 1 upward. HELLO is sent by the
 * server on connect with seq 0 and needs no ack. Every other reply echoes
 * the request seq. A seq that does not match the expected counter is fatal
 * on both sides, per section 7, since silent desync would corrupt training
 * data in a way that is very hard to diagnose later.
 *
 * All integers are little endian. Floats are IEEE 754 float32 LE.
 *
 * Message types:
 *   0x01 RESET       client -> server. Seeds for all M envs.
 *   0x02 STEP        client -> server. Actions for all M envs plus the
 *                    seeds to use if an env terminates on this step.
 *   0x03 CLOSE       client -> server. No body. The server drops the client
 *                    and keeps listening. Envs are kept for the next RESET.
 *   0x10 HELLO       server -> client on connect. Versions and shapes.
 *   0x11 RESET_REPLY server -> client. Initial obs, ticks, masks.
 *   0x12 STEP_REPLY  server -> client. Post step obs, rewards, dones,
 *                    terminals, ticks, next step masks.
 *   0xFF ERROR       server -> client, then the connection is destroyed.
 *
 * Body layouts:
 *   HELLO:      [u32 version][u32 numEnvs][u32 obsSize]
 *               [u32 intentTypes][u32 targetSlots][u32 ratioBuckets]
 *               [u32 spatialCells][u32 buildSites]
 *   RESET:      [u32 version][u32 numEnvs][seed...]
 *               seed = [u16 len][utf8 bytes], non empty
 *   STEP:       [u32 version][actions][nextSeed...]
 *               actions = M times [u16 intentType][u16 target]
 *               [u16 ratioBucket][u16 spatialCell][u16 buildSite]
 *               nextSeed = [u16 len][utf8 bytes], empty means derive
 *   RESET_REPLY:[obs f32, M*obsSize][ticks u32, M][masks u8, M*maskTotal]
 *   STEP_REPLY: [obs f32, M*obsSize][rewards f32, M][done u8, M]
 *               [terminal u8, M][ticks u32, M][masks u8, M*maskTotal]
 *   ERROR:      [u8 code][u16 msgLen][utf8 bytes]
 *
 * Mask order inside each per env block: intentType, target, ratioBucket,
 * spatialCell, buildSite, concatenated in that order.
 *
 * Terminal codes: 0 continuing, 1 win, 2 death, 3 tick_cap, 4 sim_error.
 *
 * On a done step the obs, ticks and masks describe the fresh auto reset
 * episode, while reward, done and terminal describe the finished one.
 * The terminal tick count is not preserved. Phase 2 trainers that need
 * value bootstrapping on tick_cap should know the bootstrap reads the
 * fresh state, not the terminal one.
 */

export const SCHEMA_VERSION = 1;

export const enum MessageType {
  Reset = 0x01,
  Step = 0x02,
  Close = 0x03,
  Hello = 0x10,
  ResetReply = 0x11,
  StepReply = 0x12,
  Error = 0xff,
}

export const enum TerminalCode {
  Continuing = 0,
  Win = 1,
  Death = 2,
  TickCap = 3,
  SimError = 4,
}

export const enum ProtocolErrorCode {
  BadSequence = 1,
  BadLength = 2,
  BadMessage = 3,
  VersionMismatch = 4,
  IllegalAction = 5,
  Internal = 6,
}

/** Mask head lengths, mirroring ActionCodec. Checked in BridgeSizes.test. */
export const MASK_INTENT_TYPES = 14;
export const MASK_TARGET_SLOTS = 17;
export const MASK_RATIO_BUCKETS = 8;
export const MASK_SPATIAL_CELLS = 8192;
export const MASK_BUILD_SITES = 32;
export const MASK_TOTAL =
  MASK_INTENT_TYPES +
  MASK_TARGET_SLOTS +
  MASK_RATIO_BUCKETS +
  MASK_SPATIAL_CELLS +
  MASK_BUILD_SITES;

/** Float count per observation, mirroring ObservationEncoder.size. */
export const OBSERVATION_SIZE = 65684;

export const ACTION_U16_PER_ENV = 5;
export const ACTION_BYTES_PER_ENV = ACTION_U16_PER_ENV * 2;

/** Largest accepted frame payload, a guard against corrupt lengths. */
export const MAX_FRAME_BYTES = 64 * 1024 * 1024;

/** Largest accepted seed in bytes. */
export const MAX_SEED_BYTES = 256;

export class ProtocolError extends Error {
  readonly errorCode: ProtocolErrorCode;
  constructor(errorCode: ProtocolErrorCode, message: string) {
    super(`protocol: ${message}`);
    this.name = "ProtocolError";
    this.errorCode = errorCode;
  }
}

export function terminalToCode(reason: TerminalReason): number {
  switch (reason) {
    case null:
      return TerminalCode.Continuing;
    case "win":
      return TerminalCode.Win;
    case "death":
      return TerminalCode.Death;
    case "tick_cap":
      return TerminalCode.TickCap;
    case "sim_error":
      return TerminalCode.SimError;
    default:
      throw new ProtocolError(
        ProtocolErrorCode.BadMessage,
        `unknown terminal reason ${(reason as string) ?? "null"}`,
      );
  }
}

export function codeToTerminal(code: number): TerminalReason {
  switch (code) {
    case TerminalCode.Continuing:
      return null;
    case TerminalCode.Win:
      return "win";
    case TerminalCode.Death:
      return "death";
    case TerminalCode.TickCap:
      return "tick_cap";
    case TerminalCode.SimError:
      return "sim_error";
    default:
      throw new ProtocolError(
        ProtocolErrorCode.BadMessage,
        `unknown terminal code ${code}`,
      );
  }
}

export interface HelloInfo {
  version: number;
  numEnvs: number;
  obsSize: number;
  intentTypes: number;
  targetSlots: number;
  ratioBuckets: number;
  spatialCells: number;
  buildSites: number;
}

export interface DecodedFrame {
  seq: number;
  msgType: number;
  body: Uint8Array;
}

const textEncoder = new TextEncoder();
const utf8Strict = new TextDecoder("utf-8", { fatal: true });

function decodeUtf8(bytes: Uint8Array, what: string): string {
  try {
    return utf8Strict.decode(bytes);
  } catch {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, `${what} is not valid UTF-8`);
  }
}

function checkU32(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, `${name} out of u32 range`);
  }
}

function checkU16(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff) {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, `${name} out of u16 range`);
  }
}

function checkU8(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0 || value > 0xff) {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, `${name} out of u8 range`);
  }
}

/** Wrap seq, type and body in the length prefix. */
export function encodeFrame(seq: number, msgType: number, body: Uint8Array): Uint8Array {
  checkU32(seq, "seq");
  checkU8(msgType, "msgType");
  const payloadLen = 4 + 1 + body.length;
  const out = new Uint8Array(4 + payloadLen);
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  view.setUint32(0, payloadLen, true);
  view.setUint32(4, seq >>> 0, true);
  out[8] = msgType & 0xff;
  out.set(body, 9);
  return out;
}

/** Split one framed payload back into seq, type and body. */
export function decodeFramePayload(payload: Uint8Array): DecodedFrame {
  if (payload.length < 5) {
    throw new ProtocolError(
      ProtocolErrorCode.BadLength,
      `payload too short: ${payload.length}`,
    );
  }
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  return {
    seq: view.getUint32(0, true),
    msgType: payload[4],
    body: payload.slice(5),
  };
}

/**
 * Incremental stream splitter. Push TCP chunks in, get complete payloads
 * out. Keeps leftover bytes buffered. Throws ProtocolError on an absurd
 * length prefix rather than allocating for it.
 */
export class FrameReader {
  private buffered = new Uint8Array(0);

  push(chunk: Uint8Array): Uint8Array[] {
    const joined = new Uint8Array(this.buffered.length + chunk.length);
    joined.set(this.buffered, 0);
    joined.set(chunk, this.buffered.length);
    this.buffered = joined;

    const frames: Uint8Array[] = [];
    for (;;) {
      if (this.buffered.length < 4) return frames;
      const view = new DataView(
        this.buffered.buffer,
        this.buffered.byteOffset,
        this.buffered.byteLength,
      );
      const payloadLen = view.getUint32(0, true);
      if (payloadLen < 5) {
        throw new ProtocolError(
          ProtocolErrorCode.BadLength,
          `frame length too short: ${payloadLen}`,
        );
      }
      if (payloadLen > MAX_FRAME_BYTES) {
        throw new ProtocolError(
          ProtocolErrorCode.BadLength,
          `frame length too large: ${payloadLen}`,
        );
      }
      if (this.buffered.length < 4 + payloadLen) return frames;
      frames.push(this.buffered.slice(4, 4 + payloadLen));
      this.buffered = this.buffered.slice(4 + payloadLen);
    }
  }
}

function encodeSeedList(seeds: string[], allowEmpty: boolean): Uint8Array {
  const encoded = seeds.map((s) => textEncoder.encode(s));
  let total = 0;
  for (const bytes of encoded) {
    checkU16(bytes.length, "seed length");
    if (bytes.length > MAX_SEED_BYTES) {
      throw new ProtocolError(
        ProtocolErrorCode.BadMessage,
        `seed longer than ${MAX_SEED_BYTES} bytes`,
      );
    }
    if (!allowEmpty && bytes.length === 0) {
      throw new ProtocolError(ProtocolErrorCode.BadMessage, "seed must not be empty");
    }
    total += 2 + bytes.length;
  }
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  let offset = 0;
  for (const bytes of encoded) {
    view.setUint16(offset, bytes.length, true);
    offset += 2;
    out.set(bytes, offset);
    offset += bytes.length;
  }
  return out;
}

function decodeSeedList(
  view: DataView,
  bytes: Uint8Array,
  offset: number,
  count: number,
  allowEmpty: boolean,
): { seeds: string[]; offset: number } {
  const seeds: string[] = [];
  for (let i = 0; i < count; i++) {
    if (offset + 2 > bytes.length) {
      throw new ProtocolError(ProtocolErrorCode.BadLength, "seed list truncated");
    }
    const len = view.getUint16(offset, true);
    offset += 2;
    if (len > MAX_SEED_BYTES) {
      throw new ProtocolError(
        ProtocolErrorCode.BadMessage,
        `seed longer than ${MAX_SEED_BYTES} bytes`,
      );
    }
    if (offset + len > bytes.length) {
      throw new ProtocolError(ProtocolErrorCode.BadLength, "seed bytes truncated");
    }
    const seed = decodeUtf8(bytes.slice(offset, offset + len), "seed");
    if (!allowEmpty && seed.length === 0) {
      throw new ProtocolError(ProtocolErrorCode.BadMessage, "seed must not be empty");
    }
    seeds.push(seed);
    offset += len;
  }
  return { seeds, offset };
}

export function encodeHelloBody(info: HelloInfo): Uint8Array {
  const out = new Uint8Array(8 * 4);
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  view.setUint32(0, info.version, true);
  view.setUint32(4, info.numEnvs, true);
  view.setUint32(8, info.obsSize, true);
  view.setUint32(12, info.intentTypes, true);
  view.setUint32(16, info.targetSlots, true);
  view.setUint32(20, info.ratioBuckets, true);
  view.setUint32(24, info.spatialCells, true);
  view.setUint32(28, info.buildSites, true);
  return out;
}

export function decodeHelloBody(body: Uint8Array): HelloInfo {
  if (body.length !== 32) {
    throw new ProtocolError(
      ProtocolErrorCode.BadLength,
      `hello body must be 32 bytes, got ${body.length}`,
    );
  }
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  return {
    version: view.getUint32(0, true),
    numEnvs: view.getUint32(4, true),
    obsSize: view.getUint32(8, true),
    intentTypes: view.getUint32(12, true),
    targetSlots: view.getUint32(16, true),
    ratioBuckets: view.getUint32(20, true),
    spatialCells: view.getUint32(24, true),
    buildSites: view.getUint32(28, true),
  };
}

export function encodeResetBody(version: number, seeds: string[]): Uint8Array {
  checkU32(version, "version");
  if (seeds.length < 1 || seeds.length > 1024) {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, `bad numEnvs ${seeds.length}`);
  }
  const seedBytes = encodeSeedList(seeds, false);
  const out = new Uint8Array(8 + seedBytes.length);
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  view.setUint32(0, version >>> 0, true);
  view.setUint32(4, seeds.length, true);
  out.set(seedBytes, 8);
  return out;
}

export function decodeResetBody(body: Uint8Array): { version: number; seeds: string[] } {
  if (body.length < 8) {
    throw new ProtocolError(ProtocolErrorCode.BadLength, "reset body too short");
  }
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  const version = view.getUint32(0, true);
  const numEnvs = view.getUint32(4, true);
  if (numEnvs === 0 || numEnvs > 1024) {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, `bad numEnvs ${numEnvs}`);
  }
  const { seeds, offset } = decodeSeedList(view, body, 8, numEnvs, false);
  if (offset !== body.length) {
    throw new ProtocolError(
      ProtocolErrorCode.BadLength,
      `reset body has ${body.length - offset} trailing bytes`,
    );
  }
  return { version, seeds };
}

export function encodeStepBody(
  version: number,
  actions: Action[],
  nextSeeds: string[],
): Uint8Array {
  checkU32(version, "version");
  if (actions.length !== nextSeeds.length) {
    throw new ProtocolError(
      ProtocolErrorCode.BadMessage,
      "actions and nextSeeds must cover the same envs",
    );
  }
  checkU32(actions.length, "numEnvs");
  const seedBytes = encodeSeedList(nextSeeds, true);
  const out = new Uint8Array(4 + actions.length * ACTION_BYTES_PER_ENV + seedBytes.length);
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  view.setUint32(0, version >>> 0, true);
  let offset = 4;
  for (const a of actions) {
    checkU16(a.intentType, "intentType");
    checkU16(a.target, "target");
    checkU16(a.ratioBucket, "ratioBucket");
    checkU16(a.spatialCell, "spatialCell");
    checkU16(a.buildSite, "buildSite");
    view.setUint16(offset, a.intentType, true);
    view.setUint16(offset + 2, a.target, true);
    view.setUint16(offset + 4, a.ratioBucket, true);
    view.setUint16(offset + 6, a.spatialCell, true);
    view.setUint16(offset + 8, a.buildSite, true);
    offset += ACTION_BYTES_PER_ENV;
  }
  out.set(seedBytes, offset);
  return out;
}

export function decodeStepBody(
  body: Uint8Array,
  numEnvs: number,
): { version: number; actions: Action[]; nextSeeds: string[] } {
  const head = 4 + numEnvs * ACTION_BYTES_PER_ENV;
  if (body.length < head) {
    throw new ProtocolError(ProtocolErrorCode.BadLength, "step body too short");
  }
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  const version = view.getUint32(0, true);
  const actions: Action[] = [];
  let offset = 4;
  for (let i = 0; i < numEnvs; i++) {
    actions.push({
      intentType: view.getUint16(offset, true),
      target: view.getUint16(offset + 2, true),
      ratioBucket: view.getUint16(offset + 4, true),
      spatialCell: view.getUint16(offset + 6, true),
      buildSite: view.getUint16(offset + 8, true),
    });
    offset += ACTION_BYTES_PER_ENV;
  }
  const { seeds, offset: end } = decodeSeedList(view, body, offset, numEnvs, true);
  if (end !== body.length) {
    throw new ProtocolError(
      ProtocolErrorCode.BadLength,
      `step body has ${body.length - end} trailing bytes`,
    );
  }
  return { version, actions, nextSeeds: seeds };
}

function packMasks(maskList: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(maskList.length * MASK_TOTAL);
  let offset = 0;
  for (const m of maskList) {
    if (m.length !== MASK_TOTAL) {
      throw new ProtocolError(
        ProtocolErrorCode.BadMessage,
        `mask block must be ${MASK_TOTAL} bytes, got ${m.length}`,
      );
    }
    out.set(m, offset);
    offset += MASK_TOTAL;
  }
  return out;
}

function unpackMasks(packed: Uint8Array, numEnvs: number): Uint8Array[] {
  if (packed.length !== numEnvs * MASK_TOTAL) {
    throw new ProtocolError(ProtocolErrorCode.BadLength, "mask section wrong size");
  }
  const out: Uint8Array[] = [];
  for (let i = 0; i < numEnvs; i++) {
    out.push(packed.slice(i * MASK_TOTAL, (i + 1) * MASK_TOTAL));
  }
  return out;
}

/** Flatten per head mask buffers into one MASK_TOTAL block in schema order. */
export function flattenMasks(mask: {
  intentType: Uint8Array;
  target: Uint8Array;
  ratioBucket: Uint8Array;
  spatialCell: Uint8Array;
  buildSite: Uint8Array;
}): Uint8Array {
  const out = new Uint8Array(MASK_TOTAL);
  let offset = 0;
  const parts = [mask.intentType, mask.target, mask.ratioBucket, mask.spatialCell, mask.buildSite];
  const want = [MASK_INTENT_TYPES, MASK_TARGET_SLOTS, MASK_RATIO_BUCKETS, MASK_SPATIAL_CELLS, MASK_BUILD_SITES];
  for (let i = 0; i < parts.length; i++) {
    if (parts[i].length !== want[i]) {
      throw new ProtocolError(
        ProtocolErrorCode.BadMessage,
        `mask head ${i} must be ${want[i]} bytes, got ${parts[i].length}`,
      );
    }
    out.set(parts[i], offset);
    offset += want[i];
  }
  return out;
}

/** Split one MASK_TOTAL block back into per head views. */
export function splitMasks(block: Uint8Array): {
  intentType: Uint8Array;
  target: Uint8Array;
  ratioBucket: Uint8Array;
  spatialCell: Uint8Array;
  buildSite: Uint8Array;
} {
  if (block.length !== MASK_TOTAL) {
    throw new ProtocolError(
      ProtocolErrorCode.BadLength,
      `mask block must be ${MASK_TOTAL} bytes, got ${block.length}`,
    );
  }
  let offset = 0;
  const take = (n: number): Uint8Array => {
    const part = block.slice(offset, offset + n);
    offset += n;
    return part;
  };
  return {
    intentType: take(MASK_INTENT_TYPES),
    target: take(MASK_TARGET_SLOTS),
    ratioBucket: take(MASK_RATIO_BUCKETS),
    spatialCell: take(MASK_SPATIAL_CELLS),
    buildSite: take(MASK_BUILD_SITES),
  };
}

export function resetReplySize(numEnvs: number, obsSize: number): number {
  return numEnvs * obsSize * 4 + numEnvs * 4 + numEnvs * MASK_TOTAL;
}

export function stepReplySize(numEnvs: number, obsSize: number): number {
  return (
    numEnvs * obsSize * 4 + numEnvs * 4 + numEnvs + numEnvs + numEnvs * 4 + numEnvs * MASK_TOTAL
  );
}

export function encodeResetReplyBody(
  obsList: Float32Array[],
  ticks: number[],
  maskList: Uint8Array[],
  obsSize: number,
): Uint8Array {
  const numEnvs = obsList.length;
  if (ticks.length !== numEnvs || maskList.length !== numEnvs) {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, "reset reply arrays disagree");
  }
  const out = new Uint8Array(resetReplySize(numEnvs, obsSize));
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  let offset = 0;
  for (const obs of obsList) {
    if (obs.length !== obsSize) {
      throw new ProtocolError(
        ProtocolErrorCode.BadMessage,
        `obs must hold ${obsSize} floats, got ${obs.length}`,
      );
    }
    for (let i = 0; i < obsSize; i++) {
      view.setFloat32(offset, obs[i], true);
      offset += 4;
    }
  }
  for (const t of ticks) {
    checkU32(t, "ticks");
    view.setUint32(offset, t, true);
    offset += 4;
  }
  out.set(packMasks(maskList), offset);
  return out;
}

export function decodeResetReplyBody(
  body: Uint8Array,
  numEnvs: number,
  obsSize: number,
): { obs: Float32Array[]; ticks: number[]; masks: Uint8Array[] } {
  if (body.length !== resetReplySize(numEnvs, obsSize)) {
    throw new ProtocolError(
      ProtocolErrorCode.BadLength,
      `reset reply must be ${resetReplySize(numEnvs, obsSize)} bytes, got ${body.length}`,
    );
  }
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  let offset = 0;
  const obs: Float32Array[] = [];
  for (let e = 0; e < numEnvs; e++) {
    const arr = new Float32Array(obsSize);
    for (let i = 0; i < obsSize; i++) {
      arr[i] = view.getFloat32(offset, true);
      offset += 4;
    }
    obs.push(arr);
  }
  const ticks: number[] = [];
  for (let e = 0; e < numEnvs; e++) {
    ticks.push(view.getUint32(offset, true));
    offset += 4;
  }
  const masks = unpackMasks(body.slice(offset), numEnvs);
  return { obs, ticks, masks };
}

export function encodeStepReplyBody(
  obsList: Float32Array[],
  rewards: number[],
  done: boolean[] | Uint8Array,
  terminalCodes: number[],
  ticks: number[],
  maskList: Uint8Array[],
  obsSize: number,
): Uint8Array {
  const numEnvs = obsList.length;
  if (
    rewards.length !== numEnvs ||
    done.length !== numEnvs ||
    terminalCodes.length !== numEnvs ||
    ticks.length !== numEnvs ||
    maskList.length !== numEnvs
  ) {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, "step reply arrays disagree");
  }
  const out = new Uint8Array(stepReplySize(numEnvs, obsSize));
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  let offset = 0;
  for (const obs of obsList) {
    if (obs.length !== obsSize) {
      throw new ProtocolError(
        ProtocolErrorCode.BadMessage,
        `obs must hold ${obsSize} floats, got ${obs.length}`,
      );
    }
    for (let i = 0; i < obsSize; i++) {
      view.setFloat32(offset, obs[i], true);
      offset += 4;
    }
  }
  for (const r of rewards) {
    if (!Number.isFinite(r)) {
      throw new ProtocolError(ProtocolErrorCode.BadMessage, "reward must be finite");
    }
    view.setFloat32(offset, r, true);
    offset += 4;
  }
  for (let e = 0; e < numEnvs; e++) out[offset++] = done[e] ? 1 : 0;
  for (const code of terminalCodes) {
    if (!Number.isInteger(code) || code < 0 || code > 255) {
      throw new ProtocolError(ProtocolErrorCode.BadMessage, `bad terminal code ${code}`);
    }
    out[offset++] = code;
  }
  for (const t of ticks) {
    checkU32(t, "ticks");
    view.setUint32(offset, t, true);
    offset += 4;
  }
  out.set(packMasks(maskList), offset);
  return out;
}

export function decodeStepReplyBody(
  body: Uint8Array,
  numEnvs: number,
  obsSize: number,
): {
  obs: Float32Array[];
  rewards: number[];
  done: boolean[];
  terminalCodes: number[];
  ticks: number[];
  masks: Uint8Array[];
} {
  if (body.length !== stepReplySize(numEnvs, obsSize)) {
    throw new ProtocolError(
      ProtocolErrorCode.BadLength,
      `step reply must be ${stepReplySize(numEnvs, obsSize)} bytes, got ${body.length}`,
    );
  }
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  let offset = 0;
  const obs: Float32Array[] = [];
  for (let e = 0; e < numEnvs; e++) {
    const arr = new Float32Array(obsSize);
    for (let i = 0; i < obsSize; i++) {
      arr[i] = view.getFloat32(offset, true);
      offset += 4;
    }
    obs.push(arr);
  }
  const rewards: number[] = [];
  for (let e = 0; e < numEnvs; e++) {
    rewards.push(view.getFloat32(offset, true));
    offset += 4;
  }
  const done: boolean[] = [];
  for (let e = 0; e < numEnvs; e++) done.push(body[offset++] === 1);
  const terminalCodes: number[] = [];
  for (let e = 0; e < numEnvs; e++) terminalCodes.push(body[offset++]);
  const ticks: number[] = [];
  for (let e = 0; e < numEnvs; e++) {
    ticks.push(view.getUint32(offset, true));
    offset += 4;
  }
  const masks = unpackMasks(body.slice(offset), numEnvs);
  return { obs, rewards, done, terminalCodes, ticks, masks };
}

export function encodeErrorBody(code: number, message: string): Uint8Array {
  if (!Number.isInteger(code) || code < 0 || code > 255) {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, `bad error code ${code}`);
  }
  const msgBytes = textEncoder.encode(message);
  checkU16(msgBytes.length, "error message length");
  const out = new Uint8Array(1 + 2 + msgBytes.length);
  out[0] = code;
  new DataView(out.buffer, out.byteOffset, out.byteLength).setUint16(1, msgBytes.length, true);
  out.set(msgBytes, 3);
  return out;
}

export function decodeErrorBody(body: Uint8Array): { code: number; message: string } {
  if (body.length < 3) {
    throw new ProtocolError(ProtocolErrorCode.BadLength, "error body too short");
  }
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  const len = view.getUint16(1, true);
  if (3 + len !== body.length) {
    throw new ProtocolError(ProtocolErrorCode.BadLength, "error body length mismatch");
  }
  return { code: body[0], message: decodeUtf8(body.slice(3), "error message") };
}
