import { spawn, type ChildProcess } from "child_process";
import net from "net";
import path from "path";
import { fileURLToPath } from "url";
import { afterAll, describe, expect, it } from "vitest";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ENV_ROOT = path.join(__dirname, "..");
const TSX_CLI = path.join(ENV_ROOT, "node_modules", "tsx", "dist", "cli.mjs");

async function freePort(): Promise<number> {
  const server = net.createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  await new Promise<void>((resolve) => server.close(() => resolve()));
  return port;
}

class PlayClient {
  private socket!: net.Socket;
  private buffer = "";
  private waiters: Array<(msg: any) => void> = [];

  async connect(port: number): Promise<void> {
    await new Promise<void>((resolve, reject) => {
      this.socket = net.createConnection(port, "127.0.0.1");
      this.socket.setEncoding("utf8");
      this.socket.once("connect", resolve);
      this.socket.once("error", reject);
      this.socket.on("data", (chunk: string) => {
        this.buffer += chunk;
        let newline: number;
        while ((newline = this.buffer.indexOf("\n")) >= 0) {
          const line = this.buffer.slice(0, newline).trim();
          this.buffer = this.buffer.slice(newline + 1);
          if (!line) continue;
          const waiter = this.waiters.shift();
          waiter?.(JSON.parse(line));
        }
      });
    });
    // Connected: later errors tear down the socket instead of crashing.
    this.socket.on("error", () => this.socket.destroy());
  }

  send(command: unknown): Promise<any> {
    return new Promise((resolve) => {
      this.waiters.push(resolve);
      this.socket.write(`${JSON.stringify(command)}\n`);
    });
  }

  close(): void {
    this.socket.destroy();
  }
}

describe("versus play server", () => {
  let proc: ChildProcess | null = null;

  afterAll(() => {
    proc?.kill();
    proc = null;
  });

  it("serves reset/step/state over JSON lines", async () => {
    const port = await freePort();
    let stderr = "";
    proc = spawn(
      "node",
      [TSX_CLI, "src/play/play_server.ts", "--port", String(port)],
      { cwd: ENV_ROOT, env: { ...process.env, TSX_TSCONFIG_PATH: "tsconfig.probes.json", NODE_OPTIONS: "--preserve-symlinks" } },
    );
    proc.stderr?.setEncoding("utf8");
    proc.stderr?.on("data", (chunk: string) => {
      stderr = `${stderr}${chunk}`.slice(-2000);
    });
    const client = new PlayClient();
    const deadline = Date.now() + 60_000;
    let lastError: unknown = null;
    for (;;) {
      try {
        await client.connect(port);
        break;
      } catch (err) {
        lastError = err;
        if (Date.now() > deadline) {
          throw new Error(`play server never accepted (last: ${String(lastError)}; stderr: ${stderr})`);
        }
        await new Promise((resolve) => setTimeout(resolve, 250));
      }
    }
    try {
      const hello = await client.send({ cmd: "hello" });
      expect(hello.ok).toBe(true);
      expect(hello.service).toBe("versus-play");
      expect(hello.obsSize).toBeGreaterThan(0);

      const reset = await client.send({ cmd: "reset", seed: "versus-server-test", bots: 2, nations: "disabled" });
      expect(reset.ok).toBe(true);
      expect(reset.snapshot.cells.length).toBe(64 * 32);
      expect(typeof reset.aiObsB64).toBe("string");
      expect(reset.humanMasks.intent_type.length).toBe(14);

      const noop = { intentType: 0, target: 0, ratioBucket: 0, spatialCell: 0, buildSite: 0 };
      const step = await client.send({ cmd: "step", ai: noop, human: noop });
      expect(step.ok).toBe(true);
      expect(step.done).toBe(false);
      expect(step.ticks).toBeGreaterThan(reset.ticks);
      expect(typeof step.aiObsB64).toBe("string");
      expect(step.aiMasks.intent_type.length).toBe(14);
      expect(step.humanMasks.intent_type.length).toBe(14);
      expect(step.snapshot.cells.length).toBe(64 * 32);

      const state = await client.send({ cmd: "state" });
      expect(state.ok).toBe(true);
      expect(state.ticks).toBe(step.ticks);

      const bad = await client.send({ cmd: "step", ai: { ...noop, intentType: 99 }, human: noop });
      expect(bad.ok).toBe(false);
    } finally {
      client.close();
    }
  }, 120_000);
});
