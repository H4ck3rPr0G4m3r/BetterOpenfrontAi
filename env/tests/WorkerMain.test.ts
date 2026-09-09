import { describe, expect, it } from "vitest";
import { GameMapType } from "../../OpenFrontIO/src/core/game/Game";
import { DEFAULT_WORKER_OPTIONS, parseWorkerArgs } from "../src/bridge/worker_main";

describe("parseWorkerArgs", () => {
  it("fills defaults around a required port", () => {
    const options = parseWorkerArgs(["--port", "5001"]);
    expect(options).toEqual({ ...DEFAULT_WORKER_OPTIONS, port: 5001 });
  });

  it("maps map names to the engine enum", () => {
    expect(parseWorkerArgs(["--port", "1", "--map", "World"]).map).toBe(GameMapType.World);
    expect(parseWorkerArgs(["--port", "1", "--map", "Baltics"]).map).toBe(GameMapType.Baltics);
    expect(() => parseWorkerArgs(["--port", "1", "--map", "Atlantis"])).toThrow();
  });

  it("requires one of port or path", () => {
    expect(() => parseWorkerArgs([])).toThrow();
    expect(parseWorkerArgs(["--path", "openfront-0"]).path).toBe("openfront-0");
  });

  it("bounds num-envs and flips strictness", () => {
    expect(parseWorkerArgs(["--port", "1", "--num-envs", "4"]).numEnvs).toBe(4);
    expect(() => parseWorkerArgs(["--port", "1", "--num-envs", "0"])).toThrow();
    expect(parseWorkerArgs(["--port", "1", "--lenient"]).strictIllegalActions).toBe(false);
    expect(parseWorkerArgs(["--port", "1", "--lenient", "--strict"]).strictIllegalActions).toBe(
      true,
    );
  });

  it("rejects unknown flags and missing values", () => {
    expect(() => parseWorkerArgs(["--port", "1", "--bogus"])).toThrow();
    expect(() => parseWorkerArgs(["--port"])).toThrow();
  });
});
