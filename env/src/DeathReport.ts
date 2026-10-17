import * as fs from "fs";
import type { Game, Player } from "../../OpenFrontIO/src/core/game/Game";
import { PlayerType, UnitType } from "../../OpenFrontIO/src/core/game/Game";
import { borderingSmallIDs } from "./TargetSlots";

const NUKE_TYPES = [UnitType.AtomBomb, UnitType.HydrogenBomb, UnitType.MIRVWarhead] as const;

/**
 * How a seat died, for diagnosis. Off unless OPENFRONT_DEATH_LOG names a
 * file; then every seat death appends one JSON line there: when, how big
 * the seat had been, who took its land (tiles per conqueror) and which
 * hostile nukes were aimed at it.
 */
export class DeathReport {
  private peakTiles = 0;
  private peakMinute = 0;
  private readonly lostTo = new Map<string, number>();
  private readonly nukes = new Map<number, string>();
  /** Troops / max troops, one sample per step (last ~3 minutes). */
  private readonly troopShare: number[] = [];
  private atPeak = { troopShare: 0, troopsVsBiggestNeighbour: 0 };
  private biggestAttack = 0;

  static file(): string | undefined {
    return process.env.OPENFRONT_DEATH_LOG || undefined;
  }

  onConquer(winner: Player): void {
    const key = `${winner.displayName()} (${kind(winner)})`;
    this.lostTo.set(key, (this.lostTo.get(key) ?? 0) + 1);
  }

  step(game: Game, player: Player, minute: number): void {
    const tiles = player.numTilesOwned();
    const share = player.troops() / Math.max(1, game.config().maxTroops(player));
    this.troopShare.push(share);
    if (this.troopShare.length > 180) this.troopShare.shift();
    for (const a of player.incomingAttacks()) {
      this.biggestAttack = Math.max(this.biggestAttack, a.troops() / Math.max(1, player.troops()));
    }
    if (tiles > this.peakTiles) {
      this.peakTiles = tiles;
      this.peakMinute = minute;
      const bordering = borderingSmallIDs(game, player);
      const strongest = game.players().reduce((m, n) => (bordering.has(n.smallID()) ? Math.max(m, n.troops()) : m), 0);
      this.atPeak = { troopShare: share, troopsVsBiggestNeighbour: player.troops() / Math.max(1, strongest) };
    }
    for (const nuke of game.units(...NUKE_TYPES)) {
      if (nuke.owner() === player || this.nukes.has(nuke.id())) continue;
      const target = nuke.targetTile();
      if (target !== undefined && game.owner(target) === player) {
        this.nukes.set(nuke.id(), `${nuke.type()} from ${nuke.owner().displayName()}`);
      }
    }
  }

  write(file: string, info: { seed: string; seat: number; minute: number; difficulty: string }): void {
    const lostTo = [...this.lostTo.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
    const line = {
      ...info,
      peakTiles: this.peakTiles,
      peakMinute: this.peakMinute,
      lostTo: Object.fromEntries(lostTo),
      nukesAimed: [...this.nukes.values()],
      troopShareAtPeak: round(this.atPeak.troopShare),
      ourTroopsVsStrongestNeighbourAtPeak: round(this.atPeak.troopsVsBiggestNeighbour),
      // Samples are one per step (6 s of game time).
      troopShareLastMinutes: [30, 20, 10, 1].map((back) => round(this.troopShare[this.troopShare.length - back] ?? NaN)),
      biggestIncomingAttackVsOurTroops: round(this.biggestAttack),
    };
    try {
      fs.appendFileSync(file, JSON.stringify(line) + "\n");
    } catch {
      // Diagnostics never break a game.
    }
  }
}

const round = (x: number): number => Math.round(x * 100) / 100;

function kind(p: Player): string {
  switch (p.type()) {
    case PlayerType.Nation:
      return "nation";
    case PlayerType.Bot:
      return "tribe";
    default:
      return "player";
  }
}
