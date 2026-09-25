import net from "net";
import path from "path";
import { GameMapType } from "../../../OpenFrontIO/src/core/game/Game";
import { TwoSeatGame } from "./TwoSeatGame";
import { ObservationEncoder } from "../ObservationEncoder";
import { ActionCodecImpl } from "../ActionCodec";
import type { Action, EnvConfig } from "../types";

const VERSION = 1;

interface PlayCommand {
  cmd: string;
  seed?: string;
  bots?: number;
  nations?: string;
  map?: string;
  ticksPerStep?: number;
  tickCap?: number;
  ai?: Action;
  human?: Action;
}

function encodeObsBase64(obs: Float32Array): string {
  return Buffer.from(obs.buffer, obs.byteOffset, obs.byteLength).toString("base64");
}

function maskToArrays(m: { intentType: Uint8Array; target: Uint8Array; ratioBucket: Uint8Array; spatialCell: Uint8Array; buildSite: Uint8Array }) {
  // Snake-case head names match openfront_gym/spaces.py HEAD_NAMES and
  // train/play_session.py HEAD_NAMES; one order everywhere.
  return {
    intent_type: Array.from(m.intentType),
    target: Array.from(m.target),
    ratio_bucket: Array.from(m.ratioBucket),
    spatial_cell: Array.from(m.spatialCell),
    build_site: Array.from(m.buildSite),
  };
}

function validateAction(raw: unknown): Action {
  const bounds: Array<[string, number]> = [
    ["intentType", 14],
    ["target", 17],
    ["ratioBucket", 8],
    ["spatialCell", 8192],
    ["buildSite", 32],
  ];
  if (typeof raw !== "object" || raw === null) throw new Error("action must be an object");
  const action: Record<string, number> = {};
  for (const [key, size] of bounds) {
    const value = (raw as Record<string, unknown>)[key];
    if (!Number.isInteger(value) || (value as number) < 0 || (value as number) >= size) {
      throw new Error(`action.${key} must be an integer 0..${size - 1}`);
    }
    action[key] = value as number;
  }
  return action as unknown as Action;
}

const NOOP: Action = { intentType: 0, target: 0, ratioBucket: 0, spatialCell: 0, buildSite: 0 };

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let port = 0;
  let host = "127.0.0.1";
  let mapsDir = path.resolve(process.cwd(), "../OpenFrontIO/resources/maps");
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--port") port = Number(args[++i]);
    else if (args[i] === "--host") host = args[++i];
    else if (args[i] === "--maps-dir") mapsDir = args[++i];
    else throw new Error(`unknown flag ${args[i]}`);
  }
  if (!port) throw new Error("--port is required");

  const encoder = new ObservationEncoder();
  const codec = new ActionCodecImpl();
  let game: TwoSeatGame | null = null;
  let done = true;

  const server = net.createServer((socket) => {
    socket.setEncoding("utf8");
    let buffer = "";
    const send = (msg: unknown) => socket.write(`${JSON.stringify(msg)}\n`);
    socket.on("data", (chunk: string) => {
      buffer += chunk;
      let newline: number;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        void handleLine(line, send).catch((err) =>
          send({ ok: false, error: err instanceof Error ? err.message : String(err) }),
        );
      }
    });
  });

  async function handleLine(line: string, send: (msg: unknown) => void): Promise<void> {
    let command: PlayCommand;
    try {
      command = JSON.parse(line) as PlayCommand;
    } catch {
      send({ ok: false, error: "invalid JSON" });
      return;
    }
    switch (command.cmd) {
      case "hello":
        send({ ok: true, service: "versus-play", version: VERSION, obsSize: encoder.size });
        return;
      case "reset": {
        const mapName = command.map ?? "World";
        const map = (GameMapType as Record<string, GameMapType>)[mapName];
        if (map === undefined) throw new Error(`unknown map ${mapName}`);
        if (command.nations !== undefined && command.nations !== "default" && command.nations !== "disabled") {
          throw new Error(`nations must be default or disabled, got ${command.nations}`);
        }
        const config: EnvConfig = {
          map,
          mapsDir,
          bots: command.bots ?? 5,
          ticksPerStep: command.ticksPerStep ?? 10,
          tickCap: command.tickCap ?? 6000,
          nations: (command.nations as "default" | "disabled" | undefined) ?? "default",
        };
        game = new TwoSeatGame(config, encoder, codec);
        const initial = await game.reset(command.seed ?? `versus-${Date.now()}`);
        done = false;
        send({
          ok: true,
          ticks: game.ticks,
          aiObsB64: encodeObsBase64(initial.aiObs),
          aiMasks: maskToArrays(game.masksFor(game.aiPlayerID)),
          humanMasks: maskToArrays(game.masksFor(game.humanPlayerID)),
          snapshot: game.snapshot(),
        });
        return;
      }
      case "step": {
        if (game === null || done) throw new Error("no active game, send reset first");
        const result = game.step(
          command.ai !== undefined ? validateAction(command.ai) : { ...NOOP },
          command.human !== undefined ? validateAction(command.human) : { ...NOOP },
        );
        done = result.done;
        send({
          ok: true,
          done: result.done,
          terminal: result.terminal,
          ticks: result.ticks,
          aiReward: result.aiReward,
          humanReward: result.humanReward,
          aiObsB64: encodeObsBase64(result.aiObs),
          aiMasks: maskToArrays(result.aiMasks),
          humanMasks: maskToArrays(result.humanMasks),
          snapshot: game.snapshot(),
        });
        return;
      }
      case "state": {
        if (game === null) throw new Error("no active game, send reset first");
        send({ ok: true, done, ticks: game.ticks, snapshot: game.snapshot() });
        return;
      }
      default:
        send({ ok: false, error: `unknown cmd ${command.cmd}` });
    }
  }

  await new Promise<void>((resolve) => server.listen(port, host, resolve));
  console.error(`versus play server listening on ${host}:${port}`);
}

const invokedDirectly =
  process.argv[1] !== undefined && process.argv[1].endsWith("play_server.ts");
if (invokedDirectly) {
  main().catch((err) => {
    console.error(err instanceof Error ? (err.stack ?? err.message) : String(err));
    process.exit(1);
  });
}
