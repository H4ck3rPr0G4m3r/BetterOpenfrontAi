import path from "path";
import {
  GameMapSize,
  GameMapType,
  TeamGameSpawnAreas,
} from "../../OpenFrontIO/src/core/game/Game";
import type { GameMap } from "../../OpenFrontIO/src/core/game/GameMap";
import {
  AdditionalNation,
  genTerrainFromBin,
  Nation,
} from "../../OpenFrontIO/src/core/game/TerrainMapLoader";
import { NodeGameMapLoader } from "../../OpenFrontIO/tests/perf/fullgame/NodeGameMapLoader";

export interface FreshTerrain {
  nations: Nation[];
  additionalNations: AdditionalNation[];
  gameMap: GameMap;
  miniGameMap: GameMap;
  teamGameSpawnAreas?: TeamGameSpawnAreas;
}

/**
 * Load production map terrain WITHOUT the engine's module-level cache.
 *
 * `loadTerrainMap()` memoizes by `map:size` in a private `loadedMaps` map
 * with no eviction API — and the cached object holds the LIVE GameMap, so a
 * second caller gets the first game's dirt: pre-owned tiles, a shrunken
 * spawn-candidate pool, shifted PRNG spawn picks, and eventually dangling
 * ownerIDs that crash border bookkeeping. That poisons every GameEnv.reset()
 * after the first in a worker process (a training fleet would play episode 1
 * clean and every later episode on a corrupted map) and makes any same-seed
 * rebuild in tests nondeterministic. This loader mirrors loadTerrainMap's
 * construction (same bins, same manifest, same Compact scaling) minus the
 * cache and minus layer PNGs (ImageBitmaps don't exist in Node and nothing
 * headless renders them).
 */
export async function loadFreshTerrainMap(
  map: GameMapType,
  mapSize: GameMapSize,
  mapsDir: string,
): Promise<FreshTerrain> {
  const mapLoader = new NodeGameMapLoader(path.resolve(mapsDir));
  const mapFiles = mapLoader.getMapData(map);
  const manifest = await mapFiles.manifest();

  const gameMap =
    mapSize === GameMapSize.Normal
      ? await genTerrainFromBin(manifest.map, await mapFiles.mapBin())
      : await genTerrainFromBin(manifest.map4x, await mapFiles.map4xBin());
  const miniGameMap =
    mapSize === GameMapSize.Normal
      ? await genTerrainFromBin(manifest.map4x, await mapFiles.map4xBin())
      : await genTerrainFromBin(manifest.map16x, await mapFiles.map16xBin());

  if (mapSize === GameMapSize.Compact) {
    manifest.nations.forEach((nation) => {
      if (nation.coordinates !== undefined) {
        nation.coordinates = [
          Math.floor(nation.coordinates[0] / 2),
          Math.floor(nation.coordinates[1] / 2),
        ];
      }
    });
    manifest.additionalNations?.forEach((nation) => {
      if (nation.coordinates !== undefined) {
        nation.coordinates = [
          Math.floor(nation.coordinates[0] / 2),
          Math.floor(nation.coordinates[1] / 2),
        ];
      }
    });
  }

  let teamGameSpawnAreas = manifest.teamGameSpawnAreas;
  if (mapSize === GameMapSize.Compact && teamGameSpawnAreas) {
    const scaled: TeamGameSpawnAreas = {};
    for (const [key, areas] of Object.entries(teamGameSpawnAreas)) {
      scaled[key] = areas.map((a) => ({
        x: Math.floor(a.x / 2),
        y: Math.floor(a.y / 2),
        width: Math.max(1, Math.floor(a.width / 2)),
        height: Math.max(1, Math.floor(a.height / 2)),
      }));
    }
    teamGameSpawnAreas = scaled;
  }

  return {
    nations: manifest.nations,
    additionalNations: manifest.additionalNations ?? [],
    gameMap,
    miniGameMap,
    teamGameSpawnAreas,
  };
}
