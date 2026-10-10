import { BUILD_TYPES, IntentType, NUM_INTENTS } from "./Layout";

/**
 * Reward weights, by name. Every term is either potential-based (a change
 * in a state value, so idling never mints return) or an event the seat
 * caused. See Reward.ts for the exact formula.
 */
export interface RewardWeights {
  /** Coefficient on Δ log(1 + tiles / LAND_REF). */
  land: number;
  /** Coefficient on Δ log(1 + own structure levels). */
  structures: number;
  /** Coefficient on Δ log(1 + own city levels), on top of `structures`.
   *  Cities are the economy: +250k max troops per level and train stations. */
  cities: number;
  /** Coefficient on Δ log(1 + max troops / POP_REF): the population cap,
   *  which land and city levels raise. */
  popCap: number;
  /** Coefficient on Δ log(1 + income per minute / INCOME_REF), income being
   *  gold earned beyond the flat worker rate (trade ships, trains, conquest),
   *  smoothed over about a minute. A potential like land: a strong economy
   *  pays while it lasts and is given back on death. */
  income: number;
  /** Per opponent this seat eliminated (engine ConquestEvent attribution). */
  kill: number;
  /** Terminal bonus for winning the game. */
  win: number;
  /** Terminal value for being eliminated (negative). */
  death: number;
  /** Tick-cap bonus scaled by land rank among living players (1 best, 0 worst). */
  placement: number;
  /** Per-step bonus while alive. Off by default: land already prices survival. */
  survive: number;
  /** Per-step penalty at zero troops, scaled linearly up to `lowTroopsBelow`
   *  of max troops (negative). Replays showed seats expanding themselves
   *  empty and then folding to the first attack. */
  lowTroops: number;
  /** Troop share of max troops below which `lowTroops` applies. */
  lowTroopsBelow: number;
  /** Per alliance this seat broke (negative). Breaking costs traitor status
   *  in game; without a price the policy farmed request/break cycles. */
  betrayal: number;
  /** Per diplomatic intent sent: alliance request, break, embargo, target
   *  (negative). Prices spam, not diplomacy: a few per game cost ~nothing. */
  diplomacy: number;
  /** Extra land value for tiles taken from tribes (bots) or from a player
   *  with fewer troops than us: worth (1 + advantage) times normal land.
   *  Clear the easy land before fighting real opponents. */
  advantage: number;
  /** Per real attack repelled (a force of at least 20% of our troops that
   *  ended having taken under 5% of our land). */
  defend: number;
}

export const DEFAULT_REWARD_WEIGHTS: RewardWeights = {
  land: 1.0,
  structures: 0.2,
  cities: 0.5,
  popCap: 0.3,
  income: 1.0,
  kill: 0.5,
  win: 5.0,
  death: -2.0,
  placement: 1.0,
  survive: 0.0,
  lowTroops: -0.05,
  lowTroopsBelow: 0.3,
  betrayal: -1.0,
  diplomacy: -0.005,
  advantage: 0.5,
  defend: 0.3,
};

/**
 * Capability limits for the AI control UI. Disabled intents and build types
 * are shut in masks() and dropped in decode(), so a policy trained under
 * them never sees them open. Noop can never be disabled.
 */
export interface BehaviorConfig {
  disabledIntents?: number[];
  /** Build type names from Layout.BUILD_TYPES, e.g. "Factory". */
  disabledBuilds?: string[];
  /** Gold and troop donations are off unless this is set. Real games rarely
   *  need them, and the v2 run gave away more troops to allied nations than
   *  it spent on every attack combined. */
  allowDonations?: boolean;
  /** Attack and boat ratios that would leave less than this share of max
   *  troops at home are masked (the smallest ratio always stays open). */
  troopReserve?: number;
}

export interface NormalizedBehavior {
  disabledIntents: Set<number>;
  disabledBuilds: Set<string>;
  troopReserve: number;
}

export const DEFAULT_TROOP_RESERVE = 0.1;

export function normalizeBehavior(config?: BehaviorConfig): NormalizedBehavior {
  const disabledIntents = new Set<number>();
  for (const raw of config?.disabledIntents ?? []) {
    if (!Number.isInteger(raw) || raw < 1 || raw >= NUM_INTENTS) {
      throw new Error(`disabled intent must be an integer 1..${NUM_INTENTS - 1}, got ${raw}`);
    }
    disabledIntents.add(raw);
  }
  if (disabledIntents.has(IntentType.Noop)) throw new Error("noop cannot be disabled");
  if (config?.allowDonations !== true) {
    disabledIntents.add(IntentType.DonateGold);
    disabledIntents.add(IntentType.DonateTroops);
  }
  const troopReserve = config?.troopReserve ?? DEFAULT_TROOP_RESERVE;
  if (!(troopReserve >= 0 && troopReserve < 1)) throw new Error(`troopReserve must be in [0, 1), got ${troopReserve}`);
  const disabledBuilds = new Set<string>();
  for (const name of config?.disabledBuilds ?? []) {
    if (!(BUILD_TYPES as readonly string[]).includes(name)) {
      throw new Error(`unknown build type ${name}`);
    }
    disabledBuilds.add(name);
  }
  return { disabledIntents, disabledBuilds, troopReserve };
}

/** Merge a partial override onto the defaults, rejecting unknown or non-finite values. */
export function normalizeRewardWeights(partial?: Partial<Record<string, unknown>>): RewardWeights {
  const out: RewardWeights = { ...DEFAULT_REWARD_WEIGHTS };
  for (const [key, value] of Object.entries(partial ?? {})) {
    if (!(key in out)) throw new Error(`unknown reward weight ${key}`);
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`reward weight ${key} must be a finite number, got ${String(value)}`);
    }
    out[key as keyof RewardWeights] = value;
  }
  return out;
}
