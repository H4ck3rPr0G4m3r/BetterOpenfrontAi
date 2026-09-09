/**
 * Test-only world-game construction, shared by the encoder and codec suites:
 * a real production-map World game (what GameEnv constructs per spec §5.1)
 * with a spawned human agent past the spawn phase.
 *
 * Moved out of ObservationEncoder.test.ts unchanged so both suites build
 * exactly the same fixture. The engine-crash warning below moves with it.
 *
 * While writing the encoder tests we hit a reproducible "Cannot read
 * properties of undefined (reading '_borderTiles')" engine crash:
 * GameImpl.ts's updateBorderStatus calls owner(t) after hasOwner(t) passed,
 * and owner() (via playerBySmallID's `_playersBySmallID[id - 1]`) returns
 * undefined — a smallID present in the tile-state plane with no corresponding
 * player. It is **not eliminated by lowering bots** — it reproduces at
 * bots=20 too, just less reliably (env/probes/SeedScan.ts measured ~7/17
 * seeds failing within spawn + a 300-tick buffer at bots=20 on World, a
 * small sample and a floor rather than the true rate — see spec §9 criterion
 * 1 and §11's open question 5). The seeds below (encoder-safe-N) were
 * selected with that scanner because they don't trigger it within its
 * buffer; a "safe" seed can still crash further into the game, so re-check a
 * specific seed against however many extra ticks your test actually runs,
 * don't just trust scanner output at face value.
 *
 * This is a genuine pre-existing OpenFrontIO bug, not something our code
 * introduces — worth reporting upstream. GameEnv.computeTerminal() already
 * treats a caught simulation exception as a non-scoring "sim_error"
 * terminal (§7) precisely for cases like this; §9's success criteria were
 * updated to count that rate rather than assume it's zero.
 */
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
import { createNationsForGame } from "../../../OpenFrontIO/src/core/game/NationCreation";
import { loadFreshTerrainMap } from "../../src/Terrain";
import { GameRunner } from "../../../OpenFrontIO/src/core/GameRunner";
import { PseudoRandom } from "../../../OpenFrontIO/src/core/PseudoRandom";
import { GameConfig, GameStartInfo, Turn } from "../../../OpenFrontIO/src/core/Schemas";
import { simpleHash } from "../../../OpenFrontIO/src/core/Util";
import path from "path";
import { fileURLToPath } from "url";
import { TurnFeeder } from "../../src/TurnFeeder";

export const AGENT_ID = "agent0";
const AGENT_CLIENT_ID = "client0";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MAPS_DIR = path.join(__dirname, "../../../OpenFrontIO/resources/maps");

export async function buildWorldGame(
  seed: string,
  bots = 20,
  spawnMode: "first" | "random" = "first",
) {
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
  // Same fresh-terrain discipline as GameEnv.reset: never the engine's
  // cached live map (see Terrain.ts), or same-seed rebuilds diverge.
  const terrain = await loadFreshTerrainMap(gameConfig.gameMap, gameConfig.gameMapSize, MAPS_DIR);
  const random = new PseudoRandom(simpleHash(seed));
  const nations = createNationsForGame(
    gameStart,
    terrain.nations,
    terrain.additionalNations,
    0,
    random,
  );
  const human = new PlayerInfo("RLAgent", PlayerType.Human, AGENT_CLIENT_ID, AGENT_ID);
  const game = createGame(
    [human],
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    config,
    terrain.teamGameSpawnAreas,
  );
  const runner = new GameRunner(game, new Executor(game, seed, undefined), (gu) => {
    if ("errMsg" in gu) throw new Error(`sim error: ${gu.errMsg}\n${gu.stack ?? ""}`);
  });
  runner.init();
  const feeder = new TurnFeeder(runner);

  const map = game.map();
  // Same "surrounded by neutral land" quality filter GameEnv.pickSpawnTile()
  // and HumanSeatProbe.ts use — not just "the first unowned tile" scanning
  // from index 0, which can land the human adjacent to already-placed
  // nation territory and interact badly with border-tile bookkeeping.
  // "first" keeps the encoder suite's long-standing fixtures byte-identical;
  // "random" mirrors GameEnv.pickSpawnTile exactly (same PRNG object, already
  // advanced by nation creation above, so the same seed spawns the same tile
  // GameEnv.reset(seed) would choose) and spreads spawns into open areas.
  const candidates: number[] = [];
  for (let i = 0; i < map.width() * map.height(); i++) {
    if (!map.isLand(i) || map.hasOwner(i)) continue;
    let neutralNeighbours = 0;
    for (const n of map.neighbors(i)) {
      if (map.isLand(n) && !map.hasOwner(n)) neutralNeighbours++;
    }
    if (neutralNeighbours === 4) candidates.push(i);
  }
  if (candidates.length === 0) {
    throw new Error("buildWorldGame: no free land tile found for agent spawn");
  }
  const spawnTile = spawnMode === "first" ? candidates[0] : random.randElement(candidates);
  feeder.step(1, [
    { clientID: AGENT_CLIENT_ID, playerID: AGENT_ID, type: "spawn", tile: spawnTile },
  ] as unknown as Turn["intents"]);
  while (game.inSpawnPhase()) feeder.step(1, []);

  return { game, runner, feeder };
}

/**
 * buildWorldGame (random spawn, mirroring GameEnv.reset) plus a short idle
 * development buffer. The agent does NOT attack here: conquests run the
 * engine's border bookkeeping, which is where the _borderTiles crash lives,
 * so fixtures stay clear of it and tests that need attacks launch their own.
 * Leaves an alive agent with territory, accumulated troops, and developed
 * neighbors — the state masks and decodes are meant to run against.
 */
export async function buildPostSpawnGame(seed: string, bots = 20) {
  const built = await buildWorldGame(seed, bots, "random");
  const { game, feeder } = built;
  for (let i = 0; i < 3; i++) {
    feeder.step(10, []);
  }
  const agent = game.player(AGENT_ID);
  if (!agent.isAlive() || agent.numTilesOwned() === 0 || agent.troops() <= 0) {
    throw new Error(
      `fixture failed for seed ${seed}: alive=${agent.isAlive()} tiles=${agent.numTilesOwned()} troops=${agent.troops()}`,
    );
  }
  return built;
}
