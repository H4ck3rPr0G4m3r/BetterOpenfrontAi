import type { Game, Player, PlayerID } from "../../OpenFrontIO/src/core/game/Game";

/**
 * Target-slot assignment shared by the observation encoder and the action
 * codec (spec §5.3 player table, §5.4 target head). The policy reads player
 * slots from the encoder and acts on them through the codec, so the two must
 * agree on which player sits in which slot. One function, two callers, no
 * drift.
 *
 * Slot 0 is always terra nullius (represented as null — its engine id really
 * is null, see TerraNulliusImpl.id()). Slots 1-16 hold alive non-agent
 * players, bordering players first, then by tile count, mirroring the
 * encoder's long-standing order. Unfilled slots are null. Slot identity is
 * NOT stable across steps — deaths and border changes reshuffle it — which
 * the policy must tolerate.
 */
export function assignTargetSlots(
  game: Game,
  agent: Player,
  borderingSmallIDs: Set<number>,
): (Player | null)[] {
  const slots: (Player | null)[] = new Array(17).fill(null);
  const others = game
    .players()
    .filter((p) => p.smallID() !== agent.smallID() && p.isAlive());
  others.sort((a, b) => {
    const aBorders = borderingSmallIDs.has(a.smallID()) ? 0 : 1;
    const bBorders = borderingSmallIDs.has(b.smallID()) ? 0 : 1;
    if (aBorders !== bBorders) return aBorders - bBorders;
    return b.numTilesOwned() - a.numTilesOwned();
  });
  for (let i = 0; i < 16 && i < others.length; i++) {
    slots[i + 1] = others[i];
  }
  return slots;
}

/**
 * SmallIDs neighboring the agent's territory, from the incrementally
 * maintained border set rather than all owned tiles — same membership as a
 * full scan (any neighbor's tile touches a border tile by definition) at
 * O(perimeter) instead of O(territory). SmallID 0 in the result means
 * unowned land touches the agent's border (terra nullius has smallID 0).
 * Only land neighbors count: water also carries smallID 0, and open water
 * off a coast is not expandable territory.
 * Nonzero IDs with no registered player (dangling ownerIDs, see the engine
 * crash noted in ObservationEncoder) are left in the set; callers resolve
 * them against known players and ignore the rest.
 */
export function borderingSmallIDs(game: Game, agent: Player): Set<number> {
  const map = game.map();
  const agentSmallID = agent.smallID();
  const found = new Set<number>();
  for (const tile of agent.borderTiles()) {
    for (const n of map.neighbors(tile)) {
      if (!map.isLand(n)) continue;
      const id = map.ownerID(n);
      if (id !== agentSmallID) found.add(id);
    }
  }
  return found;
}

/** Look up a slotted player by head index, or null for terra nullius. */
export function slotPlayer(
  slots: (Player | null)[],
  agentID: PlayerID,
  game: Game,
  target: number,
): Player | null {
  if (target < 0 || target >= slots.length) {
    throw new RangeError(`target slot ${target} out of range`);
  }
  const p = slots[target];
  if (p === null || p === undefined) return null;
  // The slot table is a snapshot: a player may have died or the game may
  // have been reset since it was built. Resolve by id so decode never acts
  // through a stale object.
  if (!game.hasPlayer(p.id())) return null;
  const live = game.player(p.id());
  if (live.id() === agentID || !live.isAlive()) return null;
  return live;
}
