import { Config } from "../../../OpenFrontIO/src/core/configuration/Config";
import { Executor } from "../../../OpenFrontIO/src/core/execution/ExecutionManager";
import type { Game, PlayerInfo } from "../../../OpenFrontIO/src/core/game/Game";
import {
  ErrorUpdate,
  GameUpdateType,
  GameUpdateViewData,
  HashUpdate,
} from "../../../OpenFrontIO/src/core/game/GameUpdates";
import { createGame } from "../../../OpenFrontIO/src/core/game/GameImpl";
import { createNationsForGame } from "../../../OpenFrontIO/src/core/game/NationCreation";
import { loadFreshTerrainMap } from "../Terrain";
import { GameRunner } from "../../../OpenFrontIO/src/core/GameRunner";
import { PseudoRandom } from "../../../OpenFrontIO/src/core/PseudoRandom";
import type { GameStartInfo, Turn } from "../../../OpenFrontIO/src/core/Schemas";
import { simpleHash } from "../../../OpenFrontIO/src/core/Util";
import { checkVersionParity } from "./rate_limiter";

/** Pinned training engine commit (spec §2: f51f165b...). */
export const PINNED_ENGINE_COMMIT = "f51f165b947a92cc683ade3a72c3800300e86e61";

/**
 * Shadow simulation for live lobbies (§10.2-10.3): the same GameRunner +
 * Executor as GameEnv, constructed from the lobby's GameStartInfo and fed
 * the intercepted Turn stream verbatim. Determinism (§2.3) is what makes
 * this work — same start info, same turns, same state.
 *
 * The load-bearing invariant is hash agreement: every hash turn on the
 * wire must equal the shadow's hash at that tick, or the episode aborts
 * and the agent stops acting (§10.3). Terrain comes from the CDN, not the
 * socket, so a mismatch surfaces here rather than as silent misplays.
 */
export class ShadowSim {
  private runner!: GameRunner;
  private hashes = new Map<number, number>();
  private fatalError: string | undefined;

  get game(): Game {
    return this.runner.game;
  }

  async init(start: GameStartInfo, mapsDir: string, players: PlayerInfo[]): Promise<void> {
    this.hashes = new Map();
    this.fatalError = undefined;
    const coreConfig = new Config(start.config, null, false);
    const terrain = await loadFreshTerrainMap(start.config.gameMap, start.config.gameMapSize, mapsDir);
    const random = new PseudoRandom(simpleHash(start.gameID));
    const nations = createNationsForGame(
      start,
      terrain.nations,
      terrain.additionalNations,
      0,
      random,
    );
    const game = createGame(
      [...players],
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      coreConfig,
      terrain.teamGameSpawnAreas,
    );
    this.runner = new GameRunner(
      game,
      new Executor(game, start.gameID, undefined),
      (gu: GameUpdateViewData | ErrorUpdate) => this.onUpdate(gu),
    );
    this.runner.init();
  }

  /** Feed intercepted turns verbatim; returns false on sim error. */
  feed(turns: Turn[]): boolean {
    for (const turn of turns) {
      this.runner.addTurn(turn);
      if (!this.runner.executeNextTick()) return false;
      if (this.fatalError !== undefined) return false;
    }
    return true;
  }

  /** Hash agreement check (§10.3): wire hash vs shadow hash at a tick. */
  checkHash(tick: number, wireHash: number): { agree: boolean; shadowHash: number | undefined } {
    const shadowHash = this.hashes.get(tick);
    return { agree: shadowHash === wireHash, shadowHash };
  }

  lastHashTick(): number {
    let last = -1;
    for (const tick of this.hashes.keys()) if (tick > last) last = tick;
    return last;
  }

  assertHealthy(): void {
    if (this.fatalError !== undefined) throw new Error(`shadow sim error: ${this.fatalError}`);
  }

  private onUpdate(gu: GameUpdateViewData | ErrorUpdate): void {
    if ("errMsg" in gu) {
      this.fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
      return;
    }
    for (const hu of gu.updates[GameUpdateType.Hash] as HashUpdate[]) {
      this.hashes.set(hu.tick, hu.hash);
    }
  }
}

export { checkVersionParity };
