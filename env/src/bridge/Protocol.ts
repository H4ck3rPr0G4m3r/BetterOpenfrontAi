import {
  FEATURE_SIZE,
  layoutDescription,
  MASK_SIZE,
  RASTER_SIZE,
  SCHEMA_VERSION,
} from "../Layout";
import type { Action, TerminalReason } from "../types";

/**
 * Binary bridge protocol, schema v2. Mirror: train/openfront/protocol.py.
 *
 * Framing, both directions over a local TCP socket:
 *   [u32 payloadLen][u32 seq][u8 msgType][body]     (little endian)
 * The client numbers requests from 1; replies echo the request seq; HELLO
 * is seq 0. A sequence mismatch is fatal on both sides.
 *
 * Messages:
 *   0x10 HELLO  server->client  JSON {layout, games, seats, rows}
 *   0x01 RESET  client->server  JSON {seeds: string[one per game]}
 *   0x02 STEP   client->server  [u32 rows][rows x 5 x u16 action]
 *                               [u32 n][JSON {seeds?: string[per game, "" = derive], abort?: number[] game indices}]
 *   0x03 CLOSE  client->server  empty
 *   0x11 ROWS   server->client  reply to RESET and STEP, layout below
 *   0xFF ERROR  server->client  [u8 code][utf8 message], then disconnect
 *
 * Rows are game-major: row = game * seats + seat. ROWS body:
 *   [u32 rows]
 *   raster   u8  rows * RASTER_SIZE
 *   features f32 rows * FEATURE_SIZE
 *   masks    u8  rows * MASK_SIZE
 *   reward   f32 rows
 *   stats    f32 rows * 5   (tiles, troops, kills, placement, ticks)
 *   terminal u8  rows       (0 none, 1 win, 2 death, 3 tick_cap, 4 sim_error, 5 aborted)
 *   flags    u8  rows       bit0 VALID: the row acted this step (its reward/terminal are real)
 *                           bit1 DONE:  the row's episode ended this step
 *                           bit2 NEXT:  the row acts on the next step
 *                           bit3 FRESH: obs/masks belong to a freshly reset game
 * When a game ends, the server resets it in the same reply: the rows keep
 * the finished episode's reward/terminal/DONE but carry the new game's
 * obs and masks with FRESH and NEXT set.
 */

export const enum MessageType {
  Reset = 0x01,
  Step = 0x02,
  Close = 0x03,
  Hello = 0x10,
  Rows = 0x11,
  Error = 0xff,
}

export const enum ProtocolErrorCode {
  BadSequence = 1,
  BadLength = 2,
  BadMessage = 3,
  VersionMismatch = 4,
  IllegalAction = 5,
  Internal = 6,
}

export const FLAG_VALID = 1;
export const FLAG_DONE = 2;
export const FLAG_NEXT = 4;
export const FLAG_FRESH = 8;
export const STATS_PER_ROW = 5;
export const MAX_FRAME_BYTES = 256 * 1024 * 1024;
export const MAX_SEED_BYTES = 256;

export class ProtocolError extends Error {
  constructor(readonly errorCode: ProtocolErrorCode, message: string) {
    super(`protocol: ${message}`);
    this.name = "ProtocolError";
  }
}

const TERMINAL_CODES: Record<string, number> = {
  win: 1,
  death: 2,
  tick_cap: 3,
  sim_error: 4,
  aborted: 5,
};
export function terminalToCode(reason: TerminalReason): number {
  return reason === null ? 0 : TERMINAL_CODES[reason];
}

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });

export function encodeFrame(seq: number, msgType: number, body: Uint8Array): Uint8Array {
  const out = new Uint8Array(9 + body.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, 5 + body.length, true);
  view.setUint32(4, seq >>> 0, true);
  out[8] = msgType;
  out.set(body, 9);
  return out;
}

export function decodeFramePayload(payload: Uint8Array): { seq: number; msgType: number; body: Uint8Array } {
  if (payload.length < 5) throw new ProtocolError(ProtocolErrorCode.BadLength, `payload too short: ${payload.length}`);
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  return { seq: view.getUint32(0, true), msgType: payload[4], body: payload.subarray(5) };
}

/** Incremental stream splitter: push socket chunks, get whole payloads. */
export class FrameReader {
  private chunks: Uint8Array[] = [];
  private length = 0;

  push(chunk: Uint8Array): Uint8Array[] {
    this.chunks.push(chunk);
    this.length += chunk.length;
    const frames: Uint8Array[] = [];
    for (;;) {
      if (this.length < 4) return frames;
      const head = this.peek(4);
      const payloadLen = new DataView(head.buffer, head.byteOffset, 4).getUint32(0, true);
      if (payloadLen < 5 || payloadLen > MAX_FRAME_BYTES) {
        throw new ProtocolError(ProtocolErrorCode.BadLength, `bad frame length ${payloadLen}`);
      }
      if (this.length < 4 + payloadLen) return frames;
      frames.push(this.take(4 + payloadLen).slice(4));
    }
  }

  private peek(n: number): Uint8Array {
    if (this.chunks[0].length >= n) return this.chunks[0];
    return this.take(n, false);
  }

  private take(n: number, consume = true): Uint8Array {
    const out = new Uint8Array(n);
    let filled = 0;
    let i = 0;
    while (filled < n) {
      const c = this.chunks[i];
      const used = Math.min(c.length, n - filled);
      out.set(c.subarray(0, used), filled);
      filled += used;
      i++;
    }
    if (consume) {
      let remaining = n;
      while (remaining > 0) {
        const c = this.chunks[0];
        if (c.length <= remaining) {
          remaining -= c.length;
          this.chunks.shift();
        } else {
          this.chunks[0] = c.subarray(remaining);
          remaining = 0;
        }
      }
      this.length -= n;
    }
    return out;
  }
}

export function encodeJson(value: unknown): Uint8Array {
  return encoder.encode(JSON.stringify(value));
}

export function decodeJson(body: Uint8Array, what: string): any {
  try {
    return JSON.parse(decoder.decode(body));
  } catch {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, `${what} is not valid JSON`);
  }
}

export function encodeHello(games: number, seats: number): Uint8Array {
  return encodeJson({ layout: layoutDescription(), games, seats, rows: games * seats });
}

export function decodeReset(body: Uint8Array, games: number): string[] {
  const msg = decodeJson(body, "RESET");
  const seeds = msg?.seeds;
  if (!Array.isArray(seeds) || seeds.length !== games) {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, `RESET needs ${games} seeds`);
  }
  return seeds.map((s, i) => checkSeed(s, `seed ${i}`, false));
}

function checkSeed(raw: unknown, what: string, allowEmpty: boolean): string {
  if (typeof raw !== "string") throw new ProtocolError(ProtocolErrorCode.BadMessage, `${what} must be a string`);
  if (!allowEmpty && raw.length === 0) throw new ProtocolError(ProtocolErrorCode.BadMessage, `${what} is empty`);
  if (encoder.encode(raw).length > MAX_SEED_BYTES) throw new ProtocolError(ProtocolErrorCode.BadMessage, `${what} too long`);
  return raw;
}

export interface StepRequest {
  actions: Action[];
  /** Seed per game for its next reset; "" means derive from the last one. */
  seeds: string[];
  abort: Set<number>;
}

export function encodeStep(actions: Action[], seeds: string[] = [], abort: number[] = []): Uint8Array {
  const json = encodeJson({ seeds, abort });
  const out = new Uint8Array(4 + actions.length * 10 + 4 + json.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, actions.length, true);
  actions.forEach((a, i) => {
    const o = 4 + i * 10;
    view.setUint16(o, a.intentType, true);
    view.setUint16(o + 2, a.target, true);
    view.setUint16(o + 4, a.ratioBucket, true);
    view.setUint16(o + 6, a.spatialCell, true);
    view.setUint16(o + 8, a.buildSite, true);
  });
  const jsonAt = 4 + actions.length * 10;
  view.setUint32(jsonAt, json.length, true);
  out.set(json, jsonAt + 4);
  return out;
}

export function decodeStep(body: Uint8Array, rows: number, games: number): StepRequest {
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  if (body.length < 4) throw new ProtocolError(ProtocolErrorCode.BadLength, "STEP too short");
  const n = view.getUint32(0, true);
  if (n !== rows) throw new ProtocolError(ProtocolErrorCode.BadMessage, `STEP covers ${n} rows, server has ${rows}`);
  const jsonAt = 4 + rows * 10;
  if (body.length < jsonAt + 4) throw new ProtocolError(ProtocolErrorCode.BadLength, "STEP truncated");
  const actions: Action[] = [];
  for (let i = 0; i < rows; i++) {
    const o = 4 + i * 10;
    actions.push({
      intentType: view.getUint16(o, true),
      target: view.getUint16(o + 2, true),
      ratioBucket: view.getUint16(o + 4, true),
      spatialCell: view.getUint16(o + 6, true),
      buildSite: view.getUint16(o + 8, true),
    });
  }
  const jsonLen = view.getUint32(jsonAt, true);
  if (body.length !== jsonAt + 4 + jsonLen) throw new ProtocolError(ProtocolErrorCode.BadLength, "STEP length mismatch");
  const extra = decodeJson(body.subarray(jsonAt + 4), "STEP trailer");
  const seedsRaw: unknown[] = Array.isArray(extra?.seeds) ? extra.seeds : [];
  if (seedsRaw.length !== 0 && seedsRaw.length !== games) {
    throw new ProtocolError(ProtocolErrorCode.BadMessage, `STEP seeds must cover ${games} games`);
  }
  const seeds = seedsRaw.length === 0 ? new Array(games).fill("") : seedsRaw.map((s, i) => checkSeed(s, `seed ${i}`, true));
  const abort = new Set<number>();
  for (const g of Array.isArray(extra?.abort) ? extra.abort : []) {
    if (!Number.isInteger(g) || g < 0 || g >= games) throw new ProtocolError(ProtocolErrorCode.BadMessage, `bad abort game ${g}`);
    abort.add(g);
  }
  return { actions, seeds, abort };
}

export interface Row {
  raster: Uint8Array;
  features: Float32Array;
  masks: Uint8Array;
  reward: number;
  tiles: number;
  troops: number;
  kills: number;
  placement: number;
  ticks: number;
  terminal: TerminalReason;
  flags: number;
}

export function rowsSize(rows: number): number {
  return 4 + rows * (RASTER_SIZE + FEATURE_SIZE * 4 + MASK_SIZE + 4 + STATS_PER_ROW * 4 + 2);
}

export function encodeRows(rows: Row[]): Uint8Array {
  const n = rows.length;
  const out = new Uint8Array(rowsSize(n));
  const view = new DataView(out.buffer);
  view.setUint32(0, n, true);
  let o = 4;
  for (const r of rows) {
    out.set(r.raster, o);
    o += RASTER_SIZE;
  }
  for (const r of rows) {
    out.set(new Uint8Array(r.features.buffer, r.features.byteOffset, FEATURE_SIZE * 4), o);
    o += FEATURE_SIZE * 4;
  }
  for (const r of rows) {
    out.set(r.masks, o);
    o += MASK_SIZE;
  }
  for (const r of rows) {
    view.setFloat32(o, r.reward, true);
    o += 4;
  }
  for (const r of rows) {
    for (const v of [r.tiles, r.troops, r.kills, r.placement, r.ticks]) {
      view.setFloat32(o, v, true);
      o += 4;
    }
  }
  for (const r of rows) out[o++] = terminalToCode(r.terminal);
  for (const r of rows) out[o++] = r.flags;
  return out;
}

export function encodeError(code: number, message: string): Uint8Array {
  const text = encoder.encode(message.slice(0, 4000));
  const out = new Uint8Array(1 + text.length);
  out[0] = code;
  out.set(text, 1);
  return out;
}

export { SCHEMA_VERSION };
