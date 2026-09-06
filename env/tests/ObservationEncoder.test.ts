import path from "path";
import { fileURLToPath } from "url";
import { describe, expect, it } from "vitest";
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
import { TROOP_SCALE } from "../src/Reward";
import {
  GlobalField,
  NUM_SLOT_FIELDS,
  NUM_SLOTS,
  OBSERVATION_SIZE,
  ObservationEncoder,
  PLAYER_TABLE_SIZE,
  Plane,
  RASTER_CELLS,
  RASTER_HEIGHT,
  RASTER_SIZE,
  RASTER_WIDTH,
  SlotField,
} from "../src/ObservationEncoder";
import { TurnFeeder } from "../src/TurnFeeder";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MAPS_DIR = path.join(__dirname, "../../OpenFrontIO/resources/maps");
const AGENT_ID = "agent0";

/**
 * Builds a real, production-map World game with a spawned human agent —
 * the encoder must work against exactly what GameEnv constructs (§5.1).
 *
 * While writing these tests we hit a reproducible "Cannot read properties
 * of undefined (reading '_borderTiles')" engine crash: GameImpl.ts's
 * updateBorderStatus calls owner(t) after hasOwner(t) passed, and owner()
 * (via playerBySmallID's `_playersBySmallID[id - 1]`) returns undefined —
 * a smallID present in the tile-state plane with no corresponding player.
 * It is **not eliminated by lowering bots** — it reproduces at bots=20
 * too, just less reliably (env/probes/SeedScan.ts measured ~7/17 seeds
 * failing within spawn + a 300-tick buffer at bots=20 on World, a small
 * sample and a floor rather than the true rate — see spec §9 criterion 1
 * and §11's open question 5). The seeds below (encoder-safe-N) were
 * selected with that scanner because they don't trigger it within its
 * buffer; a "safe" seed can still crash further into the game (it happened
 * once here — see the `encoder-safe2-0` swap below), so re-check a
 * specific seed against however many extra ticks your test actually runs,
 * don't just trust scanner output at face value.
 *
 * This is a genuine pre-existing OpenFrontIO bug, not something our code
 * introduces — worth reporting upstream. GameEnv.computeTerminal() already
 * treats a caught simulation exception as a non-scoring "sim_error"
 * terminal (§7) precisely for cases like this; §9's success criteria were
 * updated to count that rate rather than assume it's zero.
 */
async function buildWorldGame(seed: string, bots = 20) {
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
  const mapLoader = new NodeGameMapLoader(MAPS_DIR);
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
  const runner = new GameRunner(game, new Executor(game, seed, undefined), (gu) => {
    if ("errMsg" in gu) throw new Error(`sim error: ${gu.errMsg}`);
  });
  runner.init();
  const feeder = new TurnFeeder(runner);

  const map = game.map();
  // Same "surrounded by neutral land" quality filter GameEnv.pickSpawnTile()
  // and HumanSeatProbe.ts use — not just "the first unowned tile" scanning
  // from index 0, which can land the human adjacent to already-placed
  // nation territory and interact badly with border-tile bookkeeping.
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
  feeder.step(1, [
    { clientID: "client0", playerID: AGENT_ID, type: "spawn", tile: spawnTile },
  ] as unknown as Turn["intents"]);
  while (game.inSpawnPhase()) feeder.step(1, []);

  return { game, runner, feeder };
}

/** The correctness oracle §8 calls for: recompute every plane straight from
 *  `map.ownerID()`/`map.isLand()` with the simplest possible nested loop, no
 *  shared helpers with `ObservationEncoder`, and compare. If this and the
 *  encoder ever disagree, the encoder has a bug — this function is
 *  deliberately too simple to have the same bug. */
function referenceLandAndOwnershipFractions(game: ReturnType<typeof createGame>) {
  const map = game.map();
  const w = map.width();
  const h = map.height();
  const scale = Math.max(w / RASTER_WIDTH, h / RASTER_HEIGHT);
  const scaledW = w / scale;
  const scaledH = h / scale;
  const offsetX = Math.floor((RASTER_WIDTH - scaledW) / 2);
  const offsetY = Math.floor((RASTER_HEIGHT - scaledH) / 2);

  const total = new Float64Array(RASTER_CELLS);
  const land = new Float64Array(RASTER_CELLS);
  const unowned = new Float64Array(RASTER_CELLS);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const cx = Math.min(RASTER_WIDTH - 1, Math.max(0, offsetX + Math.floor(x / scale)));
      const cy = Math.min(RASTER_HEIGHT - 1, Math.max(0, offsetY + Math.floor(y / scale)));
      const cell = cy * RASTER_WIDTH + cx;
      const t = map.ref(x, y);
      total[cell]++;
      if (!map.isLand(t)) continue;
      land[cell]++;
      if (!map.hasOwner(t)) unowned[cell]++;
    }
  }
  return { total, land, unowned };
}

describe("ObservationEncoder", () => {
  it("produces exactly `size` values, all finite", async () => {
    const { game } = await buildWorldGame("encoder-safe-0");
    const encoder = new ObservationEncoder();
    const out = new Float32Array(encoder.size);

    encoder.encode(game, AGENT_ID, out);

    expect(out.length).toBe(OBSERVATION_SIZE);
    expect(out.every((v) => Number.isFinite(v))).toBe(true);
  });

  it("throws if given a buffer of the wrong length", async () => {
    const { game } = await buildWorldGame("encoder-safe-1");
    const encoder = new ObservationEncoder();

    expect(() => encoder.encode(game, AGENT_ID, new Float32Array(10))).toThrow();
  });

  it("matches an independent full-recompute oracle for land/unowned fractions", async () => {
    // The raster-correctness test §8 calls "the single most important
    // test": the encoder's per-cell accumulation must agree with a
    // completely separate implementation of the same downsampling.
    const { game } = await buildWorldGame("encoder-safe-4", 20);
    const encoder = new ObservationEncoder();
    const out = new Float32Array(encoder.size);
    encoder.encode(game, AGENT_ID, out);

    const oracle = referenceLandAndOwnershipFractions(game);
    for (let cell = 0; cell < RASTER_CELLS; cell++) {
      const expectedLand = oracle.total[cell] > 0 ? oracle.land[cell] / oracle.total[cell] : 0;
      expect(out[Plane.Land * RASTER_CELLS + cell]).toBeCloseTo(expectedLand, 6);

      if (oracle.land[cell] > 0) {
        const expectedUnowned = oracle.unowned[cell] / oracle.land[cell];
        expect(out[Plane.UnownedOwnership * RASTER_CELLS + cell]).toBeCloseTo(
          expectedUnowned,
          6,
        );
      }
    }
  });

  it("keeps self+ally+enemy+unowned ownership fractions summing to ~1 on every land cell", async () => {
    // Every land tile belongs to exactly one of these four categories, so
    // the four fractions must partition each cell's land, independent of
    // how many players or nations are on the map.
    const { game } = await buildWorldGame("encoder-safe-5", 20);
    const encoder = new ObservationEncoder();
    const out = new Float32Array(encoder.size);
    encoder.encode(game, AGENT_ID, out);

    for (let cell = 0; cell < RASTER_CELLS; cell++) {
      const landFraction = out[Plane.Land * RASTER_CELLS + cell];
      if (landFraction === 0) continue;
      const sum =
        out[Plane.SelfOwnership * RASTER_CELLS + cell] +
        out[Plane.AllyOwnership * RASTER_CELLS + cell] +
        out[Plane.EnemyOwnership * RASTER_CELLS + cell] +
        out[Plane.UnownedOwnership * RASTER_CELLS + cell];
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it("marks the agent's own spawn cell as self-owned", async () => {
    const { game } = await buildWorldGame("encoder-safe-6", 20);
    const encoder = new ObservationEncoder();
    const out = new Float32Array(encoder.size);
    encoder.encode(game, AGENT_ID, out);

    const agent = game.player(AGENT_ID);
    const map = game.map();
    const spawnTile = agent.spawnTile()!;
    const scale = Math.max(map.width() / RASTER_WIDTH, map.height() / RASTER_HEIGHT);
    const scaledW = map.width() / scale;
    const scaledH = map.height() / scale;
    const offsetX = Math.floor((RASTER_WIDTH - scaledW) / 2);
    const offsetY = Math.floor((RASTER_HEIGHT - scaledH) / 2);
    const cx = offsetX + Math.floor(map.x(spawnTile) / scale);
    const cy = offsetY + Math.floor(map.y(spawnTile) / scale);
    const cell = cy * RASTER_WIDTH + cx;

    expect(out[Plane.SelfOwnership * RASTER_CELLS + cell]).toBeGreaterThan(0);
  });

  it("reports troop density as troops-per-tile, not the player's total broadcast onto every owned cell", async () => {
    // Regression guard: an earlier version accumulated troops/TROOP_SCALE
    // per owned tile (not divided by the owner's tile count), so every
    // cell a player owned showed their *entire* troop reserve — identical
    // in every cell, carrying no more spatial information than the
    // ownership-fraction plane already does. Dividing by tile count makes
    // it "how thin are this player's troops spread", a real per-owner
    // spatial signal.
    const { game } = await buildWorldGame("encoder-safe-6", 20);
    const encoder = new ObservationEncoder();
    const out = new Float32Array(encoder.size);
    encoder.encode(game, AGENT_ID, out);

    const agent = game.player(AGENT_ID);
    const map = game.map();
    const spawnTile = agent.spawnTile()!;
    const scale = Math.max(map.width() / RASTER_WIDTH, map.height() / RASTER_HEIGHT);
    const scaledW = map.width() / scale;
    const scaledH = map.height() / scale;
    const offsetX = Math.floor((RASTER_WIDTH - scaledW) / 2);
    const offsetY = Math.floor((RASTER_HEIGHT - scaledH) / 2);
    const cx = offsetX + Math.floor(map.x(spawnTile) / scale);
    const cy = offsetY + Math.floor(map.y(spawnTile) / scale);
    const cell = cy * RASTER_WIDTH + cx;

    const selfOwnership = out[Plane.SelfOwnership * RASTER_CELLS + cell];
    const selfDensity = out[Plane.SelfTroopDensity * RASTER_CELLS + cell];
    const expectedDensity =
      selfOwnership * (agent.troops() / TROOP_SCALE / agent.numTilesOwned());

    expect(selfDensity).toBeCloseTo(expectedDensity, 6);
    // The bug's value would have been selfOwnership * (troops/TROOP_SCALE)
    // with no /numTilesOwned — catch a regression back to that directly.
    const buggyDensity = selfOwnership * (agent.troops() / TROOP_SCALE);
    if (agent.numTilesOwned() > 1) {
      expect(selfDensity).not.toBeCloseTo(buggyDensity, 6);
    }
  });

  it("never places the agent itself in the 1-16 opponent table", async () => {
    const { game } = await buildWorldGame("encoder-safe-8", 20);
    const encoder = new ObservationEncoder();
    const out = new Float32Array(encoder.size);
    encoder.encode(game, AGENT_ID, out);

    const agent = game.player(AGENT_ID);
    const totalLandTiles = game.map().numLandTiles();
    const agentTileFraction = agent.numTilesOwned() / totalLandTiles;
    const playerTable = out.subarray(RASTER_SIZE, RASTER_SIZE + PLAYER_TABLE_SIZE);

    let validSlots = 0;
    for (let slot = 1; slot < NUM_SLOTS; slot++) {
      const base = slot * NUM_SLOT_FIELDS;
      if (playerTable[base + SlotField.Valid] === 0) continue;
      validSlots++;
      // A real distinct opponent essentially never owns exactly the
      // agent's own tile count — if the agent had leaked into its own
      // opponent table, this is the field that would give it away.
      expect(playerTable[base + SlotField.Tiles]).not.toBeCloseTo(agentTileFraction, 9);
    }
    // With 20 bots + nations, there should be real opponents to check —
    // otherwise the loop above would pass vacuously.
    expect(validSlots).toBeGreaterThan(0);
  });

  it("packs plausible values into the global scalar block", async () => {
    const { game, feeder } = await buildWorldGame("encoder-safe2-0", 20);
    const encoder = new ObservationEncoder();
    const out = new Float32Array(encoder.size);
    const globals = out.subarray(RASTER_SIZE + PLAYER_TABLE_SIZE);

    encoder.encode(game, AGENT_ID, out);
    expect(globals[GlobalField.InSpawnPhase]).toBe(0); // buildWorldGame runs past spawn
    expect(globals[GlobalField.SelfAlive]).toBe(1);
    expect(globals[GlobalField.SelfTileFraction]).toBeGreaterThan(0);
    expect(globals[GlobalField.AlivePlayerFraction]).toBeGreaterThan(0);
    expect(globals[GlobalField.AlivePlayerFraction]).toBeLessThanOrEqual(1);
    expect(globals[GlobalField.SelfRank]).toBeGreaterThanOrEqual(0);
    expect(globals[GlobalField.SelfRank]).toBeLessThanOrEqual(1);

    const ticksBefore = game.ticks();
    feeder.step(10, []);
    encoder.encode(game, AGENT_ID, out);
    // TickFraction must track game.ticks() advancing, not be frozen at
    // whatever it was when the game was constructed.
    expect(game.ticks()).toBeGreaterThan(ticksBefore);
    expect(globals[GlobalField.TickFraction]).toBeCloseTo(
      Math.min(1, game.ticks() / 6000),
      6,
    );
  });

  it("runs a full 128x64 recompute fast enough for the per-step budget", async () => {
    // §11.1's resolution: full recompute was chosen over incremental
    // maintenance specifically because it measures cheap. Verify that
    // directly rather than assuming it from the buildableUnits() findings,
    // which were about a different, much more expensive engine call.
    const { game } = await buildWorldGame("encoder-safe-12", 20);
    const encoder = new ObservationEncoder();
    const out = new Float32Array(encoder.size);

    for (let i = 0; i < 5; i++) encoder.encode(game, AGENT_ID, out); // warm up

    const iterations = 20;
    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) encoder.encode(game, AGENT_ID, out);
    const meanMs = (performance.now() - t0) / iterations;

    console.log(`ObservationEncoder.encode() on World (20 bots): ${meanMs.toFixed(2)}ms/call`);
    // Measured ~13ms before two optimizations (flat typed-array lookups
    // instead of Maps/string-keyed Records; hoisting the per-row cy out of
    // the per-tile x loop), ~7ms after. The remaining cost is dominated by
    // map.isLand(t) — an interface method called per tile (up to ~2M for
    // World) — with no bulk terrain buffer exposed to read around it
    // (unlike ownerID, which tileStateBuffer() does expose). Recorded
    // honestly rather than gated on a number invented before measuring:
    // 7ms is real added cost on top of the ~9.3ms/10-tick sim budget (§2.2),
    // not the microseconds a naive "it's just array reads" guess assumed.
    // Threshold set with headroom above the measured value, not equal to a
    // guess.
    expect(meanMs).toBeLessThan(10);
  });
});
