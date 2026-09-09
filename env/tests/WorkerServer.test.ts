import net from "net";
import { describe, expect, it } from "vitest";
import {
  FrameReader,
  MessageType,
  ProtocolErrorCode,
  SCHEMA_VERSION,
  decodeErrorBody,
  decodeFramePayload,
  decodeHelloBody,
  decodeResetReplyBody,
  decodeStepReplyBody,
  encodeFrame,
  encodeResetBody,
  encodeStepBody,
  type DecodedFrame,
} from "../src/bridge/Protocol";
import { WorkerServer, type WorkerEnv } from "../src/bridge/WorkerServer";
import type { Action, MaskBuffers, StepResult } from "../src/types";
import type { TerminalReason } from "../src/Reward";

const OBS = 8;

function testMasks(): MaskBuffers {
  const fill = (n: number): Uint8Array => new Uint8Array(n).fill(1);
  return {
    intentType: fill(14),
    target: fill(17),
    ratioBucket: fill(8),
    spatialCell: fill(8192),
    buildSite: fill(32),
  };
}

type ScriptedTerminal = TerminalReason;

function noopAction(): Action {
  return { intentType: 0, target: 0, ratioBucket: 0, spatialCell: 0, buildSite: 0 };
}

class FakeEnv implements WorkerEnv {
  resets: string[] = [];
  seenActions: Action[] = [];
  resetCount = 0;
  ticksValue = 302;
  /** Terminal per step call, consumed in order. Missing entries mean continue. */
  script: ScriptedTerminal[] = [];
  rangeLimit = 13;
  /** When set, reset waits for this gate, to stage reconnect races. */
  gate: (() => Promise<void>) | null = null;

  constructor(readonly obsSize: number = OBS) {}

  async reset(seed: string): Promise<Float32Array> {
    if (this.gate !== null) await this.gate();
    this.resets.push(seed);
    this.resetCount += 1;
    this.ticksValue = 302;
    return new Float32Array(this.obsSize).fill(this.resetCount);
  }

  get ticks(): number {
    return this.ticksValue;
  }

  currentMasks(): MaskBuffers {
    return testMasks();
  }

  step(action: Action): StepResult {
    if (action.intentType > this.rangeLimit) {
      throw new RangeError(`intentType ${action.intentType} out of range`);
    }
    this.seenActions.push({ ...action });
    const terminal = this.script[this.seenActions.length - 1] ?? null;
    this.ticksValue += 10;
    const done = terminal !== null;
    return {
      obs: new Float32Array(this.obsSize).fill(50 + this.seenActions.length),
      reward: done ? 1 : 0.1,
      done,
      info: { terminalReason: terminal, ticks: this.ticksValue, masks: testMasks() },
    };
  }

  close(): void {}
}

const silentLogger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
};

class TestClient {
  private reader = new FrameReader();
  private pending: Array<(frame: DecodedFrame) => void> = [];
  closed = false;
  hello!: ReturnType<typeof decodeHelloBody>;

  private constructor(readonly socket: net.Socket) {
    socket.on("data", (chunk: Buffer) => {
      const payloads = this.reader.push(
        new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength),
      );
      for (const payload of payloads) {
        const frame = decodeFramePayload(payload);
        const next = this.pending.shift();
        if (next !== undefined) next(frame);
      }
    });
    socket.on("close", () => {
      this.closed = true;
    });
  }

  static async connect(port: number, host: string): Promise<TestClient> {
    const socket = net.connect(port, host);
    const client = new TestClient(socket);
    const hello = await client.nextFrame();
    if (hello.msgType !== MessageType.Hello) {
      throw new Error(`expected hello, got ${hello.msgType}`);
    }
    client.hello = decodeHelloBody(hello.body);
    return client;
  }

  nextFrame(): Promise<DecodedFrame> {
    return new Promise((resolve) => {
      this.pending.push(resolve);
    });
  }

  async reset(seq: number, seeds: string[]): Promise<DecodedFrame> {
    const reply = this.nextFrame();
    this.socket.write(encodeFrame(seq, MessageType.Reset, encodeResetBody(SCHEMA_VERSION, seeds)));
    return reply;
  }

  async step(seq: number, actions: Action[], nextSeeds: string[]): Promise<DecodedFrame> {
    const reply = this.nextFrame();
    this.socket.write(
      encodeFrame(seq, MessageType.Step, encodeStepBody(SCHEMA_VERSION, actions, nextSeeds)),
    );
    return reply;
  }

  sendRaw(bytes: Uint8Array): void {
    this.socket.write(bytes);
  }

  waitClose(): Promise<void> {
    if (this.closed) return Promise.resolve();
    return new Promise((resolve) => this.socket.on("close", () => resolve()));
  }
}

async function startServer(
  numEnvs: number,
  createEnv: (index: number) => WorkerEnv,
  options: Partial<ConstructorParameters<typeof WorkerServer>[0]> = {},
): Promise<WorkerServer> {
  const server = new WorkerServer({
    numEnvs,
    createEnv,
    obsSize: OBS,
    logger: silentLogger,
    ...options,
  });
  await server.start();
  return server;
}

function serverPort(server: WorkerServer): { host: string; port: number } {
  const addr = server.getAddress();
  if (!("port" in addr)) throw new Error("expected a TCP address");
  return addr;
}

describe("WorkerServer", () => {
  it("says hello with the hosted shapes, then resets and steps", async () => {
    const env = new FakeEnv();
    const server = await startServer(1, () => env);
    try {
      const { host, port } = serverPort(server);
      const client = await TestClient.connect(port, host);
      expect(client.hello.version).toBe(SCHEMA_VERSION);
      expect(client.hello.numEnvs).toBe(1);
      expect(client.hello.obsSize).toBe(OBS);

      const resetReply = await client.reset(1, ["seed-a"]);
      expect(resetReply.seq).toBe(1);
      expect(resetReply.msgType).toBe(MessageType.ResetReply);
      const reset = decodeResetReplyBody(resetReply.body, 1, OBS);
      expect([...reset.obs[0]].every((v) => v === 1)).toBe(true);
      expect(reset.ticks).toEqual([302]);
      expect(reset.masks[0].length).toBe(14 + 17 + 8 + 8192 + 32);

      const stepReply = await client.step(2, [noopAction()], [""]);
      expect(stepReply.msgType).toBe(MessageType.StepReply);
      const step = decodeStepReplyBody(stepReply.body, 1, OBS);
      expect(step.done).toEqual([false]);
      expect(step.rewards[0]).toBeCloseTo(0.1, 6);
      expect(step.ticks).toEqual([312]);
      client.socket.destroy();
    } finally {
      await server.close();
    }
  });

  it("auto resets on done and reports the finished episode", async () => {
    const env = new FakeEnv();
    env.script = ["win"];
    const server = await startServer(1, () => env);
    try {
      const { host, port } = serverPort(server);
      const client = await TestClient.connect(port, host);
      await client.reset(1, ["base"]);
      const reply = await client.step(2, [noopAction()], [""]);
      const step = decodeStepReplyBody(reply.body, 1, OBS);
      expect(step.done).toEqual([true]);
      expect(step.terminalCodes).toEqual([1]);
      expect(step.rewards[0]).toBeCloseTo(1, 6);
      // Obs belongs to the fresh episode, not the terminal one.
      expect([...step.obs[0]].every((v) => v === 2)).toBe(true);
      expect(env.resets).toEqual(["base", "base#auto1"]);

      const next = await client.step(3, [noopAction()], [""]);
      const following = decodeStepReplyBody(next.body, 1, OBS);
      expect(following.done).toEqual([false]);
      expect(server.getStats().episodes).toBe(1);
      client.socket.destroy();
    } finally {
      await server.close();
    }
  });

  it("uses the client next seed when one is given", async () => {
    const env = new FakeEnv();
    env.script = [null, "death"];
    const server = await startServer(1, () => env);
    try {
      const { host, port } = serverPort(server);
      const client = await TestClient.connect(port, host);
      await client.reset(1, ["base"]);
      await client.step(2, [noopAction()], [""]);
      await client.step(3, [noopAction()], ["picked"]);
      expect(env.resets).toEqual(["base", "picked"]);
      client.socket.destroy();
    } finally {
      await server.close();
    }
  });

  it("counts sim_error and logs the seed with history", async () => {
    const env = new FakeEnv();
    env.script = ["sim_error"];
    const warnings: string[] = [];
    const server = await startServer(1, () => env, {
      logger: { ...silentLogger, warn: (msg: string) => warnings.push(String(msg)) },
    });
    try {
      const { host, port } = serverPort(server);
      const client = await TestClient.connect(port, host);
      await client.reset(1, ["unlucky"]);
      const reply = await client.step(2, [noopAction()], [""]);
      const step = decodeStepReplyBody(reply.body, 1, OBS);
      expect(step.done).toEqual([true]);
      expect(step.terminalCodes).toEqual([4]);
      expect(server.getStats().simErrors).toBe(1);
      expect(warnings.some((w) => w.includes("unlucky"))).toBe(true);
      client.socket.destroy();
    } finally {
      await server.close();
    }
  });

  it("fails loudly on an illegal action in strict mode", async () => {
    const env = new FakeEnv();
    const server = await startServer(1, () => env, { strictIllegalActions: true });
    try {
      const { host, port } = serverPort(server);
      const client = await TestClient.connect(port, host);
      await client.reset(1, ["s"]);
      const reply = await client.step(
        2,
        [{ intentType: 99, target: 0, ratioBucket: 0, spatialCell: 0, buildSite: 0 }],
        [""],
      );
      expect(reply.msgType).toBe(MessageType.Error);
      expect(decodeErrorBody(reply.body).code).toBe(ProtocolErrorCode.IllegalAction);
      await client.waitClose();
      expect(server.getStats().illegalDrops).toBe(1);
    } finally {
      await server.close();
    }
  });

  it("counts and retries an illegal action when not strict", async () => {
    const env = new FakeEnv();
    const server = await startServer(1, () => env, { strictIllegalActions: false });
    try {
      const { host, port } = serverPort(server);
      const client = await TestClient.connect(port, host);
      await client.reset(1, ["s"]);
      const reply = await client.step(
        2,
        [{ intentType: 99, target: 0, ratioBucket: 0, spatialCell: 0, buildSite: 0 }],
        [""],
      );
      expect(reply.msgType).toBe(MessageType.StepReply);
      const step = decodeStepReplyBody(reply.body, 1, OBS);
      expect(step.done).toEqual([false]);
      expect(server.getStats().illegalDrops).toBe(1);
      expect(env.seenActions[env.seenActions.length - 1]).toEqual(noopAction());
      client.socket.destroy();
    } finally {
      await server.close();
    }
  });

  it("rejects a sequence gap", async () => {
    const env = new FakeEnv();
    const server = await startServer(1, () => env);
    try {
      const { host, port } = serverPort(server);
      const client = await TestClient.connect(port, host);
      const pending = client.nextFrame();
      client.sendRaw(encodeFrame(999, MessageType.Reset, encodeResetBody(SCHEMA_VERSION, ["s"])));
      const reply = await pending;
      expect(reply.msgType).toBe(MessageType.Error);
      expect(decodeErrorBody(reply.body).code).toBe(ProtocolErrorCode.BadSequence);
      await client.waitClose();
    } finally {
      await server.close();
    }
  });

  it("rejects a step before any reset", async () => {
    const env = new FakeEnv();
    const server = await startServer(1, () => env);
    try {
      const { host, port } = serverPort(server);
      const client = await TestClient.connect(port, host);
      const reply = await client.step(1, [noopAction()], [""]);
      expect(reply.msgType).toBe(MessageType.Error);
      expect(decodeErrorBody(reply.body).code).toBe(ProtocolErrorCode.BadMessage);
      await client.waitClose();
    } finally {
      await server.close();
    }
  });

  it("batches two envs with independent terminals", async () => {
    const envA = new FakeEnv();
    envA.script = ["win"];
    const envB = new FakeEnv();
    const server = await startServer(2, (i) => (i === 0 ? envA : envB));
    try {
      const { host, port } = serverPort(server);
      const client = await TestClient.connect(port, host);
      expect(client.hello.numEnvs).toBe(2);
      await client.reset(1, ["a", "b"]);
      const reply = await client.step(2, [noopAction(), noopAction()], ["", ""]);
      const step = decodeStepReplyBody(reply.body, 2, OBS);
      expect(step.done).toEqual([true, false]);
      expect(step.terminalCodes[0]).toBe(1);
      expect(step.terminalCodes[1]).toBe(0);
      expect(envA.resets).toEqual(["a", "a#auto1"]);
      expect(envB.resets).toEqual(["b"]);
      client.socket.destroy();
    } finally {
      await server.close();
    }
  });

  it("drops a dead client's queued chunks instead of desyncing the next client", async () => {
    const env = new FakeEnv();
    let releaseGate!: () => void;
    const gateOpen = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    env.gate = () => gateOpen;
    const server = await startServer(1, () => env);
    try {
      const { host, port } = serverPort(server);
      const first = await TestClient.connect(port, host);
      // Two resets: the first blocks inside the env, the second queues
      // behind it on the shared processing chain.
      first.sendRaw(encodeFrame(1, MessageType.Reset, encodeResetBody(SCHEMA_VERSION, ["a"])));
      first.sendRaw(encodeFrame(2, MessageType.Reset, encodeResetBody(SCHEMA_VERSION, ["b"])));
      await new Promise((resolve) => setTimeout(resolve, 50));
      first.socket.destroy();
      await first.waitClose();
      // Let the server observe the disconnect before the next client
      // arrives, so the test stages queued chunks, not connection refusal.
      await new Promise((resolve) => setTimeout(resolve, 100));

      const second = await TestClient.connect(port, host);
      releaseGate();
      // The stale second reset must not poison this session: the new
      // client starts at seq 1 and gets a clean reply.
      const reply = await second.reset(1, ["c"]);
      expect(reply.msgType).toBe(MessageType.ResetReply);
      const reset = decodeResetReplyBody(reply.body, 1, OBS);
      expect(env.resets[env.resets.length - 1]).toBe("c");
      expect([...reset.obs[0]].every((v) => v === env.resetCount)).toBe(true);
      second.socket.destroy();
    } finally {
      await server.close();
    }
  });

  it("keeps derived auto-reset seeds within the byte limit for wide seeds", async () => {
    const env = new FakeEnv();
    env.script = ["win"];
    const server = await startServer(1, () => env);
    try {
      const { host, port } = serverPort(server);
      const client = await TestClient.connect(port, host);
      // 64 four-byte chars: 256 bytes, valid for RESET, but 262 with the
      // auto suffix, so the derived seed must be trimmed by byte length.
      const wideSeed = "😀".repeat(64);
      expect(Buffer.byteLength(wideSeed, "utf8")).toBe(256);
      await client.reset(1, [wideSeed]);
      const reply = await client.step(2, [noopAction()], [""]);
      expect(decodeStepReplyBody(reply.body, 1, OBS).done).toEqual([true]);
      expect(Buffer.byteLength(env.resets[1], "utf8")).toBeLessThanOrEqual(256);
      client.socket.destroy();
    } finally {
      await server.close();
    }
  });
});
