/**
 * Side-quest behavior controls for the AI control UI.
 *
 * Two enforcement points, both config-only with zero bridge schema change:
 *
 * - Capability limits (disabled intent types, disabled build types) are
 *   enforced inside `ActionCodecImpl`: disabled intents are forced shut in
 *   `masks()` and dropped to `[]` in `decode()`, and disabled build types
 *   are filtered out of the bounded candidate list in both places so slot
 *   indices stay aligned. A policy trained under these limits never sees
 *   the disabled actions as open, so it learns within them.
 * - Reward shaping weights override the phase-1 default (spec section 5.5)
 *   without changing the reward's shape. `Reward` reads them via
 *   `RewardConfig.weights`.
 *
 * Noop (intent 0) can never be disabled: it is the guaranteed-legal
 * fallback every mask keeps open.
 */

export interface RewardWeights {
  land: number;
  troop: number;
  gold: number;
  win: number;
  death: number;
  /** Bonus per eliminated opponent, counted in Reward between steps. */
  kill: number;
}

/** Phase-1 default shaping, unchanged unless explicitly overridden. Kill
 *  defaults to 0 so old 5-value configs behave exactly as before. */
export const DEFAULT_REWARD_WEIGHTS: RewardWeights = {
  land: 1.0,
  troop: 0.1,
  gold: 0.1,
  win: 1,
  death: -1,
  kill: 0,
};

export interface BehaviorConfig {
  /** IntentType values to close. 0 (Noop) is rejected. */
  disabledIntents?: number[];
  /** Buildable unit type names to exclude, e.g. "Factory". Exact match. */
  disabledBuilds?: string[];
  /** Partial override; missing fields keep phase-1 defaults. */
  rewardWeights?: Partial<RewardWeights>;
}

export interface NormalizedBehavior {
  disabledIntents: Set<number>;
  disabledBuilds: Set<string>;
  rewardWeights: RewardWeights;
}

/** Valid build-type names: the string values of BUILD_CANDIDATE_TYPES
 *  in ActionCodec (the UnitType enum uses spaced names like
 *  "Defense Post"). A test pins the two lists together. */
export const KNOWN_BUILD_TYPES = [
  "City",
  "Defense Post",
  "Port",
  "SAM Launcher",
  "Missile Silo",
  "Factory",
  "Warship",
] as const;

export function normalizeBehavior(config?: BehaviorConfig): NormalizedBehavior {
  const disabledIntents = new Set<number>();
  for (const raw of config?.disabledIntents ?? []) {
    if (!Number.isInteger(raw) || raw < 0 || raw > 13) {
      throw new Error(`disabled intent must be an integer 0..13, got ${raw}`);
    }
    if (raw === 0) throw new Error("intent 0 (Noop) cannot be disabled");
    disabledIntents.add(raw);
  }
  const disabledBuilds = new Set<string>();
  for (const name of config?.disabledBuilds ?? []) {
    if (!(KNOWN_BUILD_TYPES as readonly string[]).includes(name)) {
      throw new Error(`unknown build type ${name}`);
    }
    disabledBuilds.add(name);
  }
  const partial = config?.rewardWeights ?? {};
  const rewardWeights: RewardWeights = { ...DEFAULT_REWARD_WEIGHTS };
  for (const [key, value] of Object.entries(partial)) {
    if (!(key in rewardWeights)) throw new Error(`unknown reward weight ${key}`);
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`reward weight ${key} must be a finite number, got ${value}`);
    }
    rewardWeights[key as keyof RewardWeights] = value;
  }
  return { disabledIntents, disabledBuilds, rewardWeights };
}
