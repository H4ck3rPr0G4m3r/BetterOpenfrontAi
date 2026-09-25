import path from "path";
import { fileURLToPath } from "url";
import { describe, expect, it } from "vitest";
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  PlayerInfo,
  PlayerType,
} from "../../OpenFrontIO/src/core/game/Game";
import type { GameConfig, GameStartInfo, Turn } from "../../OpenFrontIO/src/core/Schemas";
import { ShadowSim } from "../src/live/shadow_sim";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MAPS_DIR = path.join(__dirname, "../../OpenFrontIO/resources/maps");

function startInfo(seed: string): GameStartInfo {
  const config: GameConfig = {
    gameMap: GameMapType.World,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Public,
    difficulty: Difficulty.Medium,
    nations: "disabled",
    donateGold: false,
    donateTroops: false,
    bots: 2,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
  };
  return { gameID: `shadow-${seed}`, lobbyCreatedAt: 0, config, players: [] };
}

function players(): PlayerInfo[] {
  return [new PlayerInfo("Shadow", PlayerType.Human, "client0", "shadow0")];
}

describe("ShadowSim", () => {
  it("reaches identical hashes on identical turn streams", async () => {
    const turns: Turn[] = Array.from({ length: 50 }, (_, turnNumber) => ({ turnNumber, intents: [] }));

    const first = new ShadowSim();
    await first.init(startInfo("twin"), MAPS_DIR, players());
    expect(first.feed(turns)).toBe(true);

    const second = new ShadowSim();
    await second.init(startInfo("twin"), MAPS_DIR, players());
    expect(second.feed(turns)).toBe(true);

    expect(first.lastHashTick()).toBeGreaterThanOrEqual(0);
    expect(second.lastHashTick()).toBe(first.lastHashTick());
    const tick = first.lastHashTick();
    const wire = first.checkHash(tick, 0);
    // Cross-instance agreement: the second sim's hash equals the first's.
    expect(second.checkHash(tick, wire.shadowHash as number).agree).toBe(true);
    // A forged wire hash disagrees.
    expect(second.checkHash(tick, (wire.shadowHash as number) + 1).agree).toBe(false);
  });
});
