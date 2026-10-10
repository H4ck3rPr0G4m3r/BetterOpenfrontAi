import { PlayerType, type Game, type Player } from "../../OpenFrontIO/src/core/game/Game";
import type { TileRef } from "../../OpenFrontIO/src/core/game/GameMap";

/** Share of our land an attack may take and still count as repelled. */
const REPEL_MAX_LOSS = 0.05;
/** An attack force smaller than this share of our troops is a probe, not a fight. */
const MIN_ATTACK_SHARE = 0.2;

interface TrackedAttack {
  attacker: number;
  real: boolean;
  lostAtStart: number;
  tilesAtStart: number;
}

/**
 * Per-seat combat bookkeeping for the advantage / defence reward terms.
 *
 * Tile captures come from a wrapper around the game's conquer() (the one
 * path every land transfer takes), so attribution is exact: who took the
 * tile, from whom, and whether the loser was a tribe or had fewer troops.
 * Repelled attacks are found by watching each incoming attack until it
 * disappears and checking how much land its attacker took meanwhile.
 */
export class CombatTracker {
  /** Tiles this seat took this step from tribes (bots) or from players with
   *  fewer troops than it: the easy, safe land to take first. */
  advantageTiles = 0;
  /** Real attacks (not probes) this seat repelled this step. */
  repelled = 0;
  private readonly lostTo = new Map<number, number>();
  private readonly attacks = new Map<string, TrackedAttack>();

  constructor(readonly smallID: number) {}

  /** Called for every land transfer in the game. */
  onConquer(winner: Player, loser: Player | null): void {
    if (loser === null) return;
    if (winner.smallID() === this.smallID) {
      if (loser.type() === PlayerType.Bot || loser.troops() < winner.troops()) this.advantageTiles++;
    } else if (loser.smallID() === this.smallID) {
      const id = winner.smallID();
      this.lostTo.set(id, (this.lostTo.get(id) ?? 0) + 1);
    }
  }

  /** Reconcile incoming attacks after the step's ticks have run. */
  update(me: Player): void {
    const live = new Set<string>();
    if (me.isAlive()) {
      const troops = me.troops();
      for (const attack of me.incomingAttacks()) {
        const id = String(attack.id());
        live.add(id);
        if (this.attacks.has(id)) continue;
        const attacker = attack.attacker();
        const force = attack.troops();
        this.attacks.set(id, {
          attacker: attacker.smallID(),
          real: force >= MIN_ATTACK_SHARE * troops,
          lostAtStart: this.lostTo.get(attacker.smallID()) ?? 0,
          tilesAtStart: me.numTilesOwned(),
        });
      }
    }
    for (const [id, a] of this.attacks) {
      if (live.has(id)) continue;
      this.attacks.delete(id);
      if (!a.real || !me.isAlive()) continue;
      const lost = (this.lostTo.get(a.attacker) ?? 0) - a.lostAtStart;
      if (lost < REPEL_MAX_LOSS * a.tilesAtStart) this.repelled++;
    }
  }

  /** Read and clear this step's counters. */
  take(): { advantageTiles: number; repelled: number } {
    const out = { advantageTiles: this.advantageTiles, repelled: this.repelled };
    this.advantageTiles = 0;
    this.repelled = 0;
    return out;
  }
}

/**
 * Route every land transfer in `game` to `listener`. Wraps the instance's
 * conquer(); the engine's behaviour is unchanged (the original runs first,
 * with the same arguments), so simulations and replays stay bit-identical.
 */
export function observeConquests(game: Game, listener: (winner: Player, loser: Player | null) => void): void {
  const impl = game as unknown as { conquer(owner: Player, tile: TileRef): void };
  const original = impl.conquer.bind(game);
  impl.conquer = (owner: Player, tile: TileRef) => {
    const previous = game.owner(tile);
    original(owner, tile);
    listener(owner, previous.isPlayer() ? (previous as Player) : null);
  };
}
