import net from "net";
import { afterEach, describe, expect, it } from "vitest";
import type { SeatStep } from "../src/GameEnv";
import { createMasks, FEATURE_SIZE, MASK_SIZE, RASTER_SIZE } from "../src/Layout";
import {
  decodeStep,
  encodeFrame,
  encodeJson,
  encodeStep,
  FLAG_DONE,
  FLAG_FRESH,
  FLAG_NEXT,
  FLAG_VALID,
  FrameReader,
  MessageType,
  rowsSize,
  STATS_PER_ROW,
} from "../src/bridge/Protocol";
import { parseWorkerArgs } from "../src/bridge/worker_main";
import { WorkerGame, WorkerServer } from "../src/bridge/WorkerServer";
import type { Action, TerminalReason } from "../src/types";

/** Scripted game: every seat dies on the step given in `deathAt`. */
class FakeGame implements WorkerGame {
  gameOver = true;
  ticks = 0;
  resets: string[] = [];
  private step_ = 0;
  constructor(readonly numSeats: number, private readonly deathAt = 3) {}
  private row(terminal: TerminalReason, active: boolean, reward = 0): SeatStep {
    return {
      obs: { raster: new Uint8Array(RASTER_SIZE).fill(this.resets.length), features: new Float32Array(FEATURE_SIZE) },
      masks: createMasks(),
      reward,
      done: terminal !== null,
      active,
      terminal,
      tiles: 1,
      troops: 2,
      kills: 0,
      placement: 0,
    };
  }
  async reset(seed: string): Promise<SeatStep[]> {
    this.resets.push(seed);
    this.gameOver = false;
    this.step_ = 0;
    return Array.from({ length: this.numSeats }, () => this.row(null, true));
  }
  step(actions: Action[]): SeatStep[] {
    if (actions.some((a) => a.intentType > 13)) throw new RangeError("bad head");
    this.step_++;
    this.ticks += 10;
    const dying = this.step_ === this.deathAt;
    if (dying) this.gameOver = true;
    return Array.from({ length: this.numSeats }, () => this.row(dying ? "death" : null, true, 1));
  }
  abort(): SeatStep[] {
    this.gameOver = true;
    return Array.from({ length: this.numSeats }, () => this.row("aborted", true));
  }
}

/** Minimal client: frame I/O over a socket, replies parsed for flags. */
async function connect(port: number) {
  const sock = net.connect(port, "127.0.0.1");
  const reader = new FrameReader();
  const queue: Uint8Array[] = [];
  const waiters: ((p: Uint8Array) => void)[] = [];
  sock.on("data", (chunk: Buffer) => {
    for (const p of reader.push(new Uint8Array(chunk))) {
      const w = waiters.shift();
      if (w) w(p);
      else queue.push(p);
    }
  });
  const next = () => new Promise<Uint8Array>((resolve) => (queue.length ? resolve(queue.shift()!) : waiters.push(resolve)));
  let seq = 0;
  const request = async (type: number, body: Uint8Array) => {
    seq++;
    sock.write(encodeFrame(seq, type, body));
    const payload = await next();
    return { seq: new DataView(payload.buffer).getUint32(0, true), type: payload[4], body: payload.subarray(5) };
  };
  const hello = await next();
  return { sock, hello: JSON.parse(new TextDecoder().decode(hello.subarray(5))), request };
}

function flags(body: Uint8Array, rows: number): number[] {
  return Array.from(body.subarray(body.length - rows));
}

describe("bridge protocol", () => {
  it("round-trips STEP and splits frames across chunk boundaries", () => {
    const actions: Action[] = [
      { intentType: 1, target: 2, ratioBucket: 3, spatialCell: 8191, buildSite: 31 },
      { intentType: 0, target: 0, ratioBucket: 0, spatialCell: 0, buildSite: 0 },
    ];
    const body = encodeStep(actions, ["a", ""], [1]);
    const req = decodeStep(body, 2, 2);
    expect(req.actions).toEqual(actions);
    expect(req.seeds).toEqual(["a", ""]);
    expect([...req.abort]).toEqual([1]);
    const frame = encodeFrame(9, MessageType.Step, body);
    const reader = new FrameReader();
    const out = [...reader.push(frame.subarray(0, 3)), ...reader.push(frame.subarray(3, 20)), ...reader.push(frame.subarray(20))];
    expect(out).toHaveLength(1);
    expect(out[0].length).toBe(frame.length - 4);
  });

  it("sizes ROWS replies from the layout", () => {
    expect(rowsSize(2)).toBe(4 + 2 * (RASTER_SIZE + FEATURE_SIZE * 4 + MASK_SIZE + 4 + STATS_PER_ROW * 4 + 2));
  });
});

describe("WorkerServer", () => {
  let server: WorkerServer | null = null;
  afterEach(async () => {
    await server?.close();
    server = null;
  });

  it("resets, steps, and auto-resets finished games in the same reply", async () => {
    const games: FakeGame[] = [];
    server = new WorkerServer({ games: 2, seats: 2, createGame: () => { const g = new FakeGame(2); games.push(g); return g; } });
    const { port } = await server.start();
    const client = await connect(port);
    expect(client.hello.rows).toBe(4);
    expect(client.hello.layout.maskSize).toBe(MASK_SIZE);

    const reset = await client.request(MessageType.Reset, encodeJson({ seeds: ["s0", "s1"] }));
    expect(reset.type).toBe(MessageType.Rows);
    expect(reset.body.length).toBe(rowsSize(4));
    expect(flags(reset.body, 4)).toEqual(new Array(4).fill(FLAG_NEXT | FLAG_FRESH));

    const noop = new Array(4).fill({ intentType: 0, target: 0, ratioBucket: 0, spatialCell: 0, buildSite: 0 });
    let step = await client.request(MessageType.Step, encodeStep(noop));
    expect(flags(step.body, 4)).toEqual(new Array(4).fill(FLAG_VALID | FLAG_NEXT));
    await client.request(MessageType.Step, encodeStep(noop));
    step = await client.request(MessageType.Step, encodeStep(noop, ["next0", ""]));
    expect(flags(step.body, 4)).toEqual(new Array(4).fill(FLAG_VALID | FLAG_DONE | FLAG_NEXT | FLAG_FRESH));
    expect(games[0].resets).toEqual(["s0", "next0"]);
    expect(games[1].resets).toEqual(["s1", "s1#1"]);
    client.sock.destroy();
  });

  it("aborts a game on request", async () => {
    server = new WorkerServer({ games: 2, seats: 1, createGame: () => new FakeGame(1, 100) });
    const { port } = await server.start();
    const client = await connect(port);
    await client.request(MessageType.Reset, encodeJson({ seeds: ["a", "b"] }));
    const noop = new Array(2).fill({ intentType: 0, target: 0, ratioBucket: 0, spatialCell: 0, buildSite: 0 });
    const step = await client.request(MessageType.Step, encodeStep(noop, [], [1]));
    const f = flags(step.body, 2);
    expect(f[0]).toBe(FLAG_VALID | FLAG_NEXT);
    expect(f[1]).toBe(FLAG_VALID | FLAG_DONE | FLAG_NEXT | FLAG_FRESH);
    const terminals = Array.from(step.body.subarray(step.body.length - 4, step.body.length - 2));
    expect(terminals).toEqual([0, 5]);
    client.sock.destroy();
  });

  it("answers protocol violations with ERROR", async () => {
    server = new WorkerServer({ games: 1, seats: 1, createGame: () => new FakeGame(1), logger: { info() {}, warn() {}, error() {} } });
    const { port } = await server.start();
    const client = await connect(port);
    const early = await client.request(MessageType.Step, encodeStep([{ intentType: 0, target: 0, ratioBucket: 0, spatialCell: 0, buildSite: 0 }]));
    expect(early.type).toBe(MessageType.Error);
    client.sock.destroy();
  });
});

describe("worker CLI", () => {
  it("parses games, seats, and named reward weights", () => {
    const o = parseWorkerArgs(["--port", "5000", "--games", "2", "--seats", "4", "--reward", '{"kill":1,"land":2}', "--disable-intent", "11"]);
    expect(o.games).toBe(2);
    expect(o.seats).toBe(4);
    expect(o.reward.kill).toBe(1);
    expect(o.reward.land).toBe(2);
    expect(o.disabledIntents).toEqual([11]);
  });

  it("rejects bad input instead of guessing", () => {
    expect(() => parseWorkerArgs([])).toThrow(/port/);
    expect(() => parseWorkerArgs(["--port", "1", "--seats", "9"])).toThrow();
    expect(() => parseWorkerArgs(["--port", "1", "--reward", '{"bogus":1}'])).toThrow(/unknown reward/);
    expect(() => parseWorkerArgs(["--port", "1", "--map", "Atlantis"])).toThrow(/unknown map/);
  });
});
