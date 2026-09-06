import type { GameRunner } from "../../OpenFrontIO/src/core/GameRunner";
import type { Turn } from "../../OpenFrontIO/src/core/Schemas";

export interface TurnFeederResult {
  /** False if the engine stopped advancing before all requested ticks ran
   *  (a caught simulation exception — see GameRunner.executeNextTick()). */
  ok: boolean;
  /** How many ticks actually executed before `ok` went false, or the full
   *  requested count on success. */
  ticksExecuted: number;
}

/**
 * Advances the simulation by exactly N ticks per `step()`.
 *
 * `GameRunner.executeNextTick()` returns false when no Turn is queued for
 * the current position — the simulation does not advance on its own. An
 * N-tick step therefore pushes N turns, all but the last carrying no
 * intents, and calls `executeNextTick()` N times. This is the loop
 * `FullGamePerf.ts` already implements (see spec §5.2); it's extracted here
 * so GameEnv can't reintroduce an off-by-one around it.
 */
export class TurnFeeder {
  private turnNumber = 0;

  constructor(private readonly runner: GameRunner) {}

  step(ticksPerStep: number, intents: Turn["intents"] = []): TurnFeederResult {
    if (!Number.isInteger(ticksPerStep) || ticksPerStep < 1) {
      throw new Error(`ticksPerStep must be a positive integer, got ${ticksPerStep}`);
    }

    let ticksExecuted = 0;
    for (let i = 0; i < ticksPerStep; i++) {
      const isLastTick = i === ticksPerStep - 1;
      this.runner.addTurn({
        turnNumber: this.turnNumber++,
        intents: isLastTick ? intents : [],
      });
      const ok = this.runner.executeNextTick();
      if (!ok) {
        return { ok: false, ticksExecuted };
      }
      ticksExecuted++;
    }
    return { ok: true, ticksExecuted };
  }
}
