import { describe, expect, it } from "vitest";
import {
  FrameReader,
  MASK_BUILD_SITES,
  MASK_INTENT_TYPES,
  MASK_RATIO_BUCKETS,
  MASK_SPATIAL_CELLS,
  MASK_TARGET_SLOTS,
  MASK_TOTAL,
  MessageType,
  OBSERVATION_SIZE,
  ProtocolError,
  SCHEMA_VERSION,
  TerminalCode,
  codeToTerminal,
  decodeErrorBody,
  decodeFramePayload,
  decodeHelloBody,
  decodeResetBody,
  decodeResetReplyBody,
  decodeStepBody,
  decodeStepReplyBody,
  encodeErrorBody,
  encodeFrame,
  encodeHelloBody,
  encodeResetBody,
  encodeResetReplyBody,
  encodeStepBody,
  encodeStepReplyBody,
  flattenMasks,
  splitMasks,
  terminalToCode,
} from "../src/bridge/Protocol";
import type { Action } from "../src/types";

const OBS = 5;

function testAction(overrides: Partial<Action> = {}): Action {
  return {
    intentType: 1,
    target: 0,
    ratioBucket: 4,
    spatialCell: 100,
    buildSite: 3,
    ...overrides,
  };
}

function testMasks(fill = 1): Uint8Array {
  return new Uint8Array(MASK_TOTAL).fill(fill);
}

describe("Protocol framing", () => {
  it("round trips seq, type and body through the length prefix", () => {
    const body = new Uint8Array([9, 8, 7, 6]);
    const framed = encodeFrame(41, MessageType.Step, body);

    expect(framed.length).toBe(4 + 4 + 1 + 4);
    const view = new DataView(framed.buffer, framed.byteOffset, framed.byteLength);
    expect(view.getUint32(0, true)).toBe(4 + 1 + 4);

    const frame = decodeFramePayload(framed.slice(4));
    expect(frame.seq).toBe(41);
    expect(frame.msgType).toBe(MessageType.Step);
    expect([...frame.body]).toEqual([9, 8, 7, 6]);
  });

  it("rejects out-of-range message types at encode time", () => {
    expect(() => encodeFrame(1, 256, new Uint8Array(0))).toThrow(ProtocolError);
    expect(() => encodeFrame(1, -1, new Uint8Array(0))).toThrow(ProtocolError);
  });

  it("rejects non-UTF8 seeds and error text instead of replacing bytes", () => {
    const badSeed = new Uint8Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0xff]);
    expect(() => decodeResetBody(badSeed)).toThrow(ProtocolError);
    const badError = new Uint8Array([4, 1, 0, 0xff]);
    expect(() => decodeErrorBody(badError)).toThrow(ProtocolError);
  });
  it("matches the cross language anchor vector", () => {
    // Mirrored byte for byte in train/tests/test_protocol.py. If this
    // changes, the Python side must change with it.
    const framed = encodeFrame(7, MessageType.Close, new Uint8Array(0));
    expect([...framed]).toEqual([5, 0, 0, 0, 7, 0, 0, 0, 3]);
  });

  it("splits coalesced frames and reassembles split chunks", () => {
    const a = encodeFrame(1, MessageType.Reset, new Uint8Array([1]));
    const b = encodeFrame(2, MessageType.Step, new Uint8Array([2, 3]));
    const joined = new Uint8Array(a.length + b.length);
    joined.set(a, 0);
    joined.set(b, a.length);

    const coalesced = new FrameReader().push(joined);
    expect(coalesced.length).toBe(2);

    const reader = new FrameReader();
    const first = reader.push(joined.slice(0, 3));
    expect(first.length).toBe(0);
    const rest = reader.push(joined.slice(3));
    expect(rest.length).toBe(2);
  });

  it("rejects absurd frame lengths instead of allocating", () => {
    const reader = new FrameReader();
    const prefix = new Uint8Array(4);
    new DataView(prefix.buffer).setUint32(0, 0xffff_ffff, true);
    expect(() => reader.push(prefix)).toThrow(ProtocolError);
  });
});

describe("Protocol bodies", () => {
  it("round trips hello", () => {
    const info = {
      version: SCHEMA_VERSION,
      numEnvs: 4,
      obsSize: OBSERVATION_SIZE,
      intentTypes: MASK_INTENT_TYPES,
      targetSlots: MASK_TARGET_SLOTS,
      ratioBuckets: MASK_RATIO_BUCKETS,
      spatialCells: MASK_SPATIAL_CELLS,
      buildSites: MASK_BUILD_SITES,
    };
    expect(decodeHelloBody(encodeHelloBody(info))).toEqual(info);
  });

  it("round trips reset seeds including unicode", () => {
    const seeds = ["ep-0", "hódmezővásárhely-1"];
    const decoded = decodeResetBody(encodeResetBody(SCHEMA_VERSION, seeds));
    expect(decoded.version).toBe(SCHEMA_VERSION);
    expect(decoded.seeds).toEqual(seeds);
  });

  it("rejects empty reset seeds and trailing bytes", () => {
    expect(() => encodeResetBody(SCHEMA_VERSION, [])).toThrow(ProtocolError);
    expect(() => encodeResetBody(SCHEMA_VERSION, [""])).toThrow(ProtocolError);
    const good = encodeResetBody(SCHEMA_VERSION, ["a", "b"]);
    const bad = new Uint8Array(good.length + 1);
    bad.set(good, 0);
    expect(() => decodeResetBody(bad)).toThrow(ProtocolError);
  });

  it("round trips step actions and next seeds", () => {
    const actions = [testAction(), testAction({ intentType: 0, spatialCell: 8191 })];
    const nextSeeds = ["", "custom-next"];
    const decoded = decodeStepBody(
      encodeStepBody(SCHEMA_VERSION, actions, nextSeeds),
      2,
    );
    expect(decoded.version).toBe(SCHEMA_VERSION);
    expect(decoded.actions).toEqual(actions);
    expect(decoded.nextSeeds).toEqual(nextSeeds);
  });

  it("round trips a reset reply", () => {
    const obs = [new Float32Array([1, 2, 3, 4, 5]), new Float32Array([6, 7, 8, 9, 10])];
    const body = encodeResetReplyBody(obs, [302, 303], [testMasks(1), testMasks(0)], OBS);
    const decoded = decodeResetReplyBody(body, 2, OBS);
    expect([...decoded.obs[0]]).toEqual([1, 2, 3, 4, 5]);
    expect([...decoded.obs[1]]).toEqual([6, 7, 8, 9, 10]);
    expect(decoded.ticks).toEqual([302, 303]);
    expect([...decoded.masks[0]].every((v) => v === 1)).toBe(true);
    expect([...decoded.masks[1]].every((v) => v === 0)).toBe(true);
  });

  it("round trips a step reply", () => {
    const obs = [new Float32Array([0.5, -1.25, 3, 4, 5])];
    const body = encodeStepReplyBody(
      obs,
      [0.25],
      [true],
      [TerminalCode.Win],
      [312],
      [testMasks(1)],
      OBS,
    );
    const decoded = decodeStepReplyBody(body, 1, OBS);
    expect(decoded.rewards[0]).toBeCloseTo(0.25, 6);
    expect(decoded.done).toEqual([true]);
    expect(decoded.terminalCodes).toEqual([TerminalCode.Win]);
    expect(decoded.ticks).toEqual([312]);
    expect([...decoded.obs[0]][0]).toBeCloseTo(0.5, 6);
  });

  it("maps every terminal reason both ways", () => {
    expect(terminalToCode(null)).toBe(TerminalCode.Continuing);
    expect(terminalToCode("win")).toBe(TerminalCode.Win);
    expect(terminalToCode("death")).toBe(TerminalCode.Death);
    expect(terminalToCode("tick_cap")).toBe(TerminalCode.TickCap);
    expect(terminalToCode("sim_error")).toBe(TerminalCode.SimError);
    expect(codeToTerminal(TerminalCode.Win)).toBe("win");
    expect(codeToTerminal(TerminalCode.SimError)).toBe("sim_error");
    expect(codeToTerminal(TerminalCode.Continuing)).toBeNull();
    expect(() => codeToTerminal(99)).toThrow(ProtocolError);
  });

  it("flattens and splits masks in schema order", () => {
    const mask = {
      intentType: new Uint8Array(MASK_INTENT_TYPES).fill(1),
      target: new Uint8Array(MASK_TARGET_SLOTS).fill(0),
      ratioBucket: new Uint8Array(MASK_RATIO_BUCKETS).fill(1),
      spatialCell: new Uint8Array(MASK_SPATIAL_CELLS).fill(0),
      buildSite: new Uint8Array(MASK_BUILD_SITES).fill(1),
    };
    mask.intentType[0] = 7;
    const flat = flattenMasks(mask);
    expect(flat.length).toBe(MASK_TOTAL);
    expect(flat[0]).toBe(7);
    const split = splitMasks(flat);
    expect([...split.intentType]).toEqual([...mask.intentType]);
    expect([...split.spatialCell]).toEqual([...mask.spatialCell]);
    expect(() => splitMasks(new Uint8Array(3))).toThrow(ProtocolError);
  });

  it("round trips error bodies", () => {
    const decoded = decodeErrorBody(encodeErrorBody(4, "version drift"));
    expect(decoded).toEqual({ code: 4, message: "version drift" });
  });
});
