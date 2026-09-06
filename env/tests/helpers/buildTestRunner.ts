/**
 * Test-only game construction: a real GameRunner + Executor (the actual
 * intent -> tick pipeline TurnFeeder and GameEnv drive), built on the small
 * `plains` test map and TestConfig for speed.
 *
 * This is deliberately NOT what GameEnv itself uses in phase 1 — GameEnv is
 * built like FullGamePerf.ts, against production maps and real Config
 * (see spec §5.1), because a policy trained against downscaled test data
 * doesn't transfer. This helper exists only to unit-test the turn-feeding
 * and episode-lifecycle *mechanics*, which don't depend on map fidelity.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Config } from "../../../OpenFrontIO/src/core/configuration/Config";
import { Executor } from "../../../OpenFrontIO/src/core/execution/ExecutionManager";
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  PlayerInfo,
  PlayerType,
} from "../../../OpenFrontIO/src/core/game/Game";
import { createGame } from "../../../OpenFrontIO/src/core/game/GameImpl";
import {
  genTerrainFromBin,
  MapManifest,
} from "../../../OpenFrontIO/src/core/game/TerrainMapLoader";
import { UserSettings } from "../../../OpenFrontIO/src/core/game/UserSettings";
import {
  ErrorUpdate,
  GameUpdateViewData,
} from "../../../OpenFrontIO/src/core/game/GameUpdates";
import { GameRunner } from "../../../OpenFrontIO/src/core/GameRunner";
import { GameConfig } from "../../../OpenFrontIO/src/core/Schemas";
import { TestConfig } from "../../../OpenFrontIO/tests/util/TestConfig";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TESTDATA_ROOT = path.join(
  __dirname,
  "../../../OpenFrontIO/tests/testdata/maps",
);
export const AGENT_ID = "agent0";

export interface TestGame {
  runner: GameRunner;
  fatalError: () => string | undefined;
}

export async function buildTestRunner(
  seed: string,
  mapName: string = "plains",
  configOverrides: Partial<GameConfig> = {},
): Promise<TestGame> {
  console.debug = () => {};

  const dir = path.join(TESTDATA_ROOT, mapName);
  const manifest = JSON.parse(
    fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
  ) as MapManifest;
  const gameMap = await genTerrainFromBin(
    manifest.map,
    fs.readFileSync(path.join(dir, "map.bin")),
  );
  const miniGameMap = await genTerrainFromBin(
    manifest.map4x,
    fs.readFileSync(path.join(dir, "map4x.bin")),
  );

  const gameConfig: GameConfig = {
    gameMap: GameMapType.Asia, // arbitrary — real terrain comes from testdata, not this enum
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Singleplayer,
    difficulty: Difficulty.Medium,
    nations: "disabled",
    donateGold: false,
    donateTroops: false,
    bots: 0,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
    ...configOverrides,
  };
  const config = new TestConfig(gameConfig, new UserSettings(), false);

  const human = new PlayerInfo("RLAgent", PlayerType.Human, "client0", AGENT_ID);
  const game = createGame([human], [], gameMap, miniGameMap, config);

  let fatalError: string | undefined;
  const runner = new GameRunner(
    game,
    new Executor(game, seed, undefined),
    (gu: GameUpdateViewData | ErrorUpdate) => {
      if ("errMsg" in gu) fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
    },
  );
  runner.init();

  return { runner, fatalError: () => fatalError };
}
