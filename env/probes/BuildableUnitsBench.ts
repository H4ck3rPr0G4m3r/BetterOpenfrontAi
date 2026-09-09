/**
 * THROWAWAY MICROBENCHMARK — answers spec open question §11.1: what does
 * `player.buildableUnits(tile)` cost per call, and can the action codec
 * afford calling it once per raster cell (128×64 = 8192) every step?
 *
 * This is the gating measurement for §5.4 (dense spatial build masking vs.
 * a candidate-list fallback) and for the phase-3 real-time deadline (§10.4).
 * Nothing downstream should be built on a guess about this number.
 *
 * Lives outside OpenFrontIO/ on purpose — that tree may become a pinned
 * submodule, and throwaway probes shouldn't accumulate inside it.
 *
 * Usage: TSX_TSCONFIG_PATH=tsconfig.probes.json npx tsx probes/BuildableUnitsBench.ts [--map world] [--bots 50]
 * (from env/; the env var is load-bearing — see SeedScan.ts's header and
 * tsconfig.probes.json for why nation games miscompile without it).
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
  PlayerBuildable,
  PlayerInfo,
  PlayerType,
} from "../../OpenFrontIO/src/core/game/Game";
import { createGame } from "../../OpenFrontIO/src/core/game/GameImpl";
import { createNationsForGame } from "../../OpenFrontIO/src/core/game/NationCreation";
import { loadTerrainMap } from "../../OpenFrontIO/src/core/game/TerrainMapLoader";
import { GameRunner } from "../../OpenFrontIO/src/core/GameRunner";
import { PseudoRandom } from "../../OpenFrontIO/src/core/PseudoRandom";
import { GameConfig, GameStartInfo, Turn } from "../../OpenFrontIO/src/core/Schemas";
import { UnitType } from "../../OpenFrontIO/src/core/game/Game";
import { simpleHash } from "../../OpenFrontIO/src/core/Util";
import { NodeGameMapLoader } from "../../OpenFrontIO/tests/perf/fullgame/NodeGameMapLoader";

const PROJECT_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../OpenFrontIO",
);
const AGENT_ID = "agent0";
const SEED = "buildable-bench";
const RASTER_CELLS = 128 * 64; // §5.3's fixed observation/action grid

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] !== undefined
    ? process.argv[i + 1]
    : fallback;
}

const MAP_ARG = arg("map", "world");
const BOTS = parseInt(arg("bots", "50"), 10);

const MAP_BY_NAME: Record<string, GameMapType> = {
  world: GameMapType.World,
  baltics: GameMapType.Baltics,
  europe: GameMapType.Europe,
};

function percentile(sorted: number[], p: number): number {
  const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * p));
  return sorted[idx];
}

async function main(): Promise<void> {
  console.debug = () => {};

  const gameMap = MAP_BY_NAME[MAP_ARG];
  if (gameMap === undefined) throw new Error(`unknown map: ${MAP_ARG}`);

  const gameConfig: GameConfig = {
    gameMap,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Public,
    difficulty: Difficulty.Medium,
    nations: "default",
    donateGold: false,
    donateTroops: false,
    bots: BOTS,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
  };
  const gameStart: GameStartInfo = {
    gameID: SEED,
    lobbyCreatedAt: 0,
    config: gameConfig,
    players: [],
  };

  const config = new Config(gameConfig, null, false);
  const mapLoader = new NodeGameMapLoader(
    path.join(PROJECT_ROOT, "resources/maps"),
  );
  const terrain = await loadTerrainMap(
    gameConfig.gameMap,
    gameConfig.gameMapSize,
    mapLoader,
  );
  const random = new PseudoRandom(simpleHash(SEED));
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

  let fatalError: string | undefined;
  const runner = new GameRunner(game, new Executor(game, SEED, undefined), (gu) => {
    if ("errMsg" in gu) fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
  });
  runner.init();

  const map = game.map();
  let spawnTile = -1;
  for (let i = 0; i < map.width() * map.height(); i++) {
    if (!map.isLand(i) || map.hasOwner(i)) continue;
    let neutralNeighbours = 0;
    for (const n of map.neighbors(i)) {
      if (map.isLand(n) && !map.hasOwner(n)) neutralNeighbours++;
    }
    if (neutralNeighbours === 4) {
      spawnTile = i;
      break;
    }
  }
  if (spawnTile < 0) throw new Error("no free land tile found");

  let turnNumber = 0;
  const runTick = (intents: Turn["intents"] = []): void => {
    runner.addTurn({ turnNumber: turnNumber++, intents });
    runner.executeNextTick();
    if (fatalError !== undefined) throw new Error(`sim error: ${fatalError}`);
  };

  // Spawn, then let the game run a while so bots have built up territory,
  // structures, and gold — a cold empty map understates real mid-game cost.
  runTick([
    { clientID: "client0", playerID: AGENT_ID, type: "spawn", tile: spawnTile },
  ] as unknown as Turn["intents"]);
  while (game.inSpawnPhase()) {
    if (turnNumber > 1000) throw new Error("spawn phase never ended");
    runTick();
  }
  for (let i = 0; i < 300; i++) runTick();

  const spawnedAgent = game.player(AGENT_ID);
  // Expand into terra nullius (same as HumanSeatProbe.ts) so the owned-tile
  // pass measures a realistic mid-game territory instead of the ~28-tile
  // spawn footprint, which understates every ownership-gated cost below.
  runTick([
    {
      clientID: "client0",
      playerID: AGENT_ID,
      type: "attack",
      targetID: null,
      troops: spawnedAgent.troops() * 0.9,
    },
  ] as unknown as Turn["intents"]);
  for (let i = 0; i < 1700; i++) runTick();

  const agent = game.player(AGENT_ID);
  if (!agent.isAlive()) {
    console.log("agent died before benchmark window — re-run with a different seed");
    process.exit(1);
  }

  // Build the exact set of tile refs the observation encoder's letterboxed
  // 128x64 raster would query: one representative land tile per cell,
  // covering both owned and unowned land so the branch mix (spawn-phase
  // checks, structure-adjacency scans) isn't artificially uniform.
  const w = map.width();
  const h = map.height();
  const scale = Math.max(w / 128, h / 64);
  const tiles: number[] = [];
  outer: for (let cy = 0; cy < 64; cy++) {
    for (let cx = 0; cx < 128; cx++) {
      const x = Math.min(w - 1, Math.floor(cx * scale));
      const y = Math.min(h - 1, Math.floor(cy * scale));
      const ref = map.ref(x, y);
      if (map.isLand(ref)) tiles.push(ref);
      if (tiles.length >= RASTER_CELLS) break outer;
    }
  }
  console.log(
    `${tiles.length} land-tile samples out of ${RASTER_CELLS} raster cells ` +
      `(map ${MAP_ARG}, ${w}x${h}, scale=${scale.toFixed(2)})\n`,
  );

  // Warm up JIT, then measure.
  for (let i = 0; i < 200; i++) agent.buildableUnits(tiles[i % tiles.length]);

  const perCall: number[] = [];
  const t0 = performance.now();
  for (const tile of tiles) {
    const s = performance.now();
    agent.buildableUnits(tile);
    perCall.push(performance.now() - s);
  }
  const totalMs = performance.now() - t0;

  perCall.sort((a, b) => a - b);
  const mean = totalMs / tiles.length;
  console.log(`${tiles.length} calls, total ${totalMs.toFixed(2)}ms`);
  console.log(
    `  mean ${(mean * 1000).toFixed(1)}us/call  ` +
      `p50 ${(percentile(perCall, 0.5) * 1000).toFixed(1)}us  ` +
      `p95 ${(percentile(perCall, 0.95) * 1000).toFixed(1)}us  ` +
      `p99 ${(percentile(perCall, 0.99) * 1000).toFixed(1)}us  ` +
      `max ${(perCall[perCall.length - 1] * 1000).toFixed(1)}us`,
  );

  // Compare against the tick budget one step (10 ticks) would otherwise
  // spend, using the §2.2 measured throughput for this map as the
  // reference (World: 1075 ticks/sec -> ~9.3ms per 10-tick step).
  const measuredTicksPerSec: Record<string, number> = {
    world: 1075,
    baltics: 1545,
    europe: 566,
  };
  const stepBudgetMs = (10 / (measuredTicksPerSec[MAP_ARG] ?? 1000)) * 1000;
  console.log(
    `\nFull-grid mask cost: ${totalMs.toFixed(1)}ms vs. ~${stepBudgetMs.toFixed(
      1,
    )}ms of simulation time per 10-tick step (${MAP_ARG}) ` +
      `-> mask cost is ${(totalMs / stepBudgetMs).toFixed(1)}x the sim budget`,
  );

  // Second pass: only the agent's own owned tiles. `validStructureSpawnTiles`
  // short-circuits to [] for tiles the caller doesn't own, so most of the
  // grid above never ran the expensive radius-15 flood fill. This is the
  // worst case a candidate-list approach (border tiles / attack clusters
  // only, not the whole raster) would actually pay per call.
  const ownedTiles = [...agent.tiles()].slice(0, 200);
  if (ownedTiles.length > 0) {
    for (let i = 0; i < 50; i++) agent.buildableUnits(ownedTiles[i % ownedTiles.length]);
    const t1 = performance.now();
    for (const tile of ownedTiles) agent.buildableUnits(tile);
    const ownedMs = performance.now() - t1;
    console.log(
      `\nOwned-tile-only pass: ${ownedTiles.length} calls, total ${ownedMs.toFixed(2)}ms, ` +
        `mean ${((ownedMs / ownedTiles.length) * 1000).toFixed(1)}us/call ` +
        `(agent owns ${agent.numTilesOwned()} tiles)`,
    );
  } else {
    console.log("\nAgent owns no tiles — skipping owned-tile pass");
  }

  // Discriminating measurement: is the per-call cost the 11-type default
  // unit list, or fixed overhead that survives a single-type query? This
  // decides whether a small candidate list (one call per candidate tile,
  // one unit type) is affordable at all.
  const singleType = [UnitType.City] as const;
  for (let i = 0; i < 200; i++) agent.buildableUnits(tiles[i % tiles.length], singleType);
  const t2 = performance.now();
  for (const tile of tiles) agent.buildableUnits(tile, singleType);
  const singleMs = performance.now() - t2;
  console.log(
    `\nSingle-unit-type pass (grid tiles): ${tiles.length} calls, total ${singleMs.toFixed(2)}ms, ` +
      `mean ${((singleMs / tiles.length) * 1000).toFixed(1)}us/call`,
  );
  if (ownedTiles.length > 0) {
    for (let i = 0; i < 50; i++) agent.buildableUnits(ownedTiles[i % ownedTiles.length], singleType);
    const t3 = performance.now();
    for (const tile of ownedTiles) agent.buildableUnits(tile, singleType);
    const ownedSingleMs = performance.now() - t3;
    console.log(
      `Single-unit-type pass (owned tiles): ${ownedTiles.length} calls, total ${ownedSingleMs.toFixed(2)}ms, ` +
        `mean ${((ownedSingleMs / ownedTiles.length) * 1000).toFixed(1)}us/call`,
    );
  }

  // Per-type breakdown: which of the 11 default types actually costs
  // something, on the grid sample (mostly-unowned) and the owned sample.
  console.log("\nPer-type breakdown (grid tiles / owned tiles), mean us/call:");
  for (const ut of PlayerBuildable.types) {
    const single = [ut] as const;
    for (let i = 0; i < 50; i++) agent.buildableUnits(tiles[i % tiles.length], single);
    const g0 = performance.now();
    for (const tile of tiles) agent.buildableUnits(tile, single);
    const gridUs = ((performance.now() - g0) / tiles.length) * 1000;

    let ownedUs = NaN;
    if (ownedTiles.length > 0) {
      for (let i = 0; i < 20; i++) agent.buildableUnits(ownedTiles[i % ownedTiles.length], single);
      const o0 = performance.now();
      for (const tile of ownedTiles) agent.buildableUnits(tile, single);
      ownedUs = ((performance.now() - o0) / ownedTiles.length) * 1000;
    }
    // `ut` printed directly: UnitType[ut] reverse lookup silently fails for
    // any value containing a space ("Defense Post", "SAM Launcher",
    // "Transport Ship"), which looked like noise in an earlier run.
    console.log(`  ${ut}: ${gridUs.toFixed(2)} / ${ownedUs.toFixed(2)}`);
  }

  // bestTransportShipSpawn() is the engine's purpose-built query for naval
  // spawn legality — the plausible replacement for masking UnitType.TransportShip
  // via buildableUnits(). Time it directly rather than assuming it's cheap
  // because it exists; it may run the same reachability search.
  for (let i = 0; i < 50; i++) agent.bestTransportShipSpawn(tiles[i % tiles.length]);
  const bt0 = performance.now();
  for (const tile of tiles) agent.bestTransportShipSpawn(tile);
  const btMs = performance.now() - bt0;
  console.log(
    `\nbestTransportShipSpawn (grid tiles): ${tiles.length} calls, total ${btMs.toFixed(2)}ms, ` +
      `mean ${((btMs / tiles.length) * 1000).toFixed(1)}us/call`,
  );

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
