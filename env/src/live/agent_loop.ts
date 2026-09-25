import type { Game, PlayerID } from "../../../OpenFrontIO/src/core/game/Game";
import type { Intent, Turn } from "../../../OpenFrontIO/src/core/Schemas";
import type { Action, MaskBuffers } from "../types";
import { RateLimiter } from "./rate_limiter";
import type { ShadowSim } from "./shadow_sim";

export interface AgentLoopConfig {
  /** Server turn interval budget in ms (~100ms live). Overrun drops the
   *  action; it never lands late against stale state (§10.4). */
  turnBudgetMs: number;
  /** Our own intent budget below the server's (§10.5). */
  maxIntentsPerMinute: number;
  ticksPerStep: number;
}

export interface Predictor {
  predict(obs: Float32Array, masks: MaskBuffers): Action;
}

export interface LoopHooks {
  encode(game: Game, playerID: PlayerID, out: Float32Array): void;
  masks(game: Game, playerID: PlayerID, out: MaskBuffers): void;
  decode(action: Action, game: Game, playerID: PlayerID): Intent[];
  createMaskBuffers(): MaskBuffers;
  obsSize(): number;
}

/**
 * Deadline-bound reactive loop for live lobbies (§10.4). The server owns
 * the clock: each intercepted Turn advances the shadow sim, then
 * encode → mask → infer → emit must finish inside one turn interval.
 *
 * Pure orchestration over injected hooks: fully unit-testable with a stub
 * sim, no engine needed. Returns the intents to emit, or [] on overrun /
 * hash disagreement / rate-limit exhaustion — the caller sends nothing
 * that tick rather than a stale or illegal action.
 */
export class AgentLoop {
  private limiter = new RateLimiter(8, 120);
  private lastTurnNumber = -1;

  constructor(
    private readonly config: AgentLoopConfig,
    private readonly hooks: LoopHooks,
    private readonly predictor: Predictor,
  ) {}

  onTurn(
    shadow: { game: Game; feed(turns: Turn[]): boolean },
    turn: Turn,
    agentPlayerID: PlayerID,
    wireHash: { tick: number; hash: number } | null,
    checkHash: (tick: number, hash: number) => boolean,
    nowMs: () => number = Date.now,
  ): { intents: Intent[]; dropped: string | null } {
    const start = nowMs();
    if (turn.turnNumber <= this.lastTurnNumber) {
      return { intents: [], dropped: "stale-turn" };
    }
    this.lastTurnNumber = turn.turnNumber;
    if (!shadow.feed([turn])) {
      return { intents: [], dropped: "shadow-error" };
    }
    if (wireHash !== null && !checkHash(wireHash.tick, wireHash.hash)) {
      // Hash disagreement: we are playing a different game. Stop acting.
      return { intents: [], dropped: "hash-mismatch" };
    }
    const obs = new Float32Array(this.hooks.obsSize());
    this.hooks.encode(shadow.game, agentPlayerID, obs);
    const masks = this.hooks.createMaskBuffers();
    this.hooks.masks(shadow.game, agentPlayerID, masks);
    const action = this.predictor.predict(obs, masks);
    const intents = this.hooks.decode(action, shadow.game, agentPlayerID);
    const elapsed = nowMs() - start;
    if (elapsed > this.config.turnBudgetMs) {
      return { intents: [], dropped: "overrun" };
    }
    const payloadBytes = JSON.stringify(intents).length;
    const waitMs = this.limiter.tryAcquire(nowMs(), payloadBytes);
    if (waitMs > 0) {
      return { intents: [], dropped: "rate-limited" };
    }
    return { intents, dropped: null };
  }
}

export type { ShadowSim };
