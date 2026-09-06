/**
 * Scans candidate seeds for World/N-bots games that survive the spawn
 * phase plus a 300-tick buffer cleanly, to find stable seeds for tests
 * around the "_borderTiles" engine crash documented in
 * ObservationEncoder.test.ts's buildWorldGame() doc comment.
 *
 * A pass here is NOT a guarantee: one seed that passed this exact check
 * still crashed after 10 more ticks in a real test (encoder-safe-10, see
 * that test file's history). Treat "OK" as "survived this much simulation",
 * not "safe forever" — a test that runs meaningfully more ticks than this
 * scanner's buffer needs its own check, not just a seed from this list.
 *
 * Kept, not deleted: the ObservationEncoder test suite's seeds
 * (encoder-safe-N) were chosen with this tool, and without it a future
 * maintainer who hits a new crashing seed has no way to reproduce how the
 * current ones were selected.
 *
 * Usage: npx tsx env/probes/SeedScan.ts <prefix> [bots]
 * Prints OK/CRASH per candidate and a list of the first 10 that pass.
 */
import path from "path";
import { fileURLToPath } from "url";
import { Config } from "../../OpenFrontIO/src/core/configuration/Config";
import { Executor } from "../../OpenFrontIO/src/core/execution/ExecutionManager";
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  PlayerInfo,
  PlayerType,
} from "../../OpenFrontIO/src/core/game/Game";
import { createGame } from "../../OpenFrontIO/src/core/game/GameImpl";
import { createNationsForGame } from "../../OpenFrontIO/src/core/game/NationCreation";
import { loadTerrainMap } from "../../OpenFrontIO/src/core/game/TerrainMapLoader";
import { GameRunner } from "../../OpenFrontIO/src/core/GameRunner";
import { PseudoRandom } from "../../OpenFrontIO/src/core/PseudoRandom";
import { GameConfig, GameStartInfo, Turn } from "../../OpenFrontIO/src/core/Schemas";
import { simpleHash } from "../../OpenFrontIO/src/core/Util";
import { NodeGameMapLoader } from "../../OpenFrontIO/tests/perf/fullgame/NodeGameMapLoader";

const PROJECT_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../OpenFrontIO",
);
const AGENT_ID = "agent0";

async function tryOneSeed(seed: string, bots: number): Promise<boolean> {
  console.debug = () => {};
  const gameConfig: GameConfig = {
    gameMap: GameMapType.World,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Public,
    difficulty: Difficulty.Medium,
    nations: "default",
    donateGold: false,
    donateTroops: false,
    bots,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
  };
  const gameStart: GameStartInfo = {
    gameID: seed,
    lobbyCreatedAt: 0,
    config: gameConfig,
    players: [],
  };
  const config = new Config(gameConfig, null, false);
  const mapLoader = new NodeGameMapLoader(path.join(PROJECT_ROOT, "resources/maps"));
  const terrain = await loadTerrainMap(gameConfig.gameMap, gameConfig.gameMapSize, mapLoader);
  const random = new PseudoRandom(simpleHash(seed));
  const nations = createNationsForGame(
    gameStart,
    terrain.nations,
    terrain.additionalNations,
    0,
    random,
  );
  const human = new PlayerInfo("RLAgent", PlayerType.Human, "client0", AGENT_ID);
  const game = createGame(
    [human],
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    config,
    terrain.teamGameSpawnAreas,
  );
  let ok = true;
  const runner = new GameRunner(game, new Executor(game, seed, undefined), (gu) => {
    if ("errMsg" in gu) ok = false;
  });
  runner.init();

  const map = game.map();
  let spawnTile = -1;
  for (let i = 0; i < map.width() * map.height(); i++) {
    if (!map.isLand(i) || map.hasOwner(i)) continue;
    let n4 = 0;
    for (const n of map.neighbors(i)) if (map.isLand(n) && !map.hasOwner(n)) n4++;
    if (n4 === 4) {
      spawnTile = i;
      break;
    }
  }
  let turnNumber = 0;
  const runTick = (intents: Turn["intents"] = []): void => {
    runner.addTurn({ turnNumber: turnNumber++, intents });
    runner.executeNextTick();
  };
  runTick([
    { clientID: "client0", playerID: AGENT_ID, type: "spawn", tile: spawnTile },
  ] as unknown as Turn["intents"]);
  let guard = 0;
  while (game.inSpawnPhase() && ok && guard++ < 1000) runTick();
  // A bit past spawn too, since some crashes might land just after.
  for (let i = 0; i < 300 && ok; i++) runTick();
  return ok;
}

async function main(): Promise<void> {
  const prefix = process.argv[2] ?? "seed";
  const bots = Number(process.argv[3] ?? 20);
  const good: string[] = [];
  for (let i = 0; i < 60; i++) {
    const seed = `${prefix}-${i}`;
    const ok = await tryOneSeed(seed, bots);
    console.log(`${ok ? "OK  " : "CRASH"} ${seed}`);
    if (ok) good.push(seed);
    if (good.length >= 10) break;
  }
  console.log("\nGood seeds:", good.join(", "));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
