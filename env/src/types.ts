import type { Game, GameMapType, PlayerID } from "../../OpenFrontIO/src/core/game/Game";
import type { Intent } from "../../OpenFrontIO/src/core/Schemas";
import type { TerminalReason } from "./Reward";

export type Observation = Float32Array;

/**
 * The five action heads from spec §5.4. `buildSite` indexes the bounded
 * candidate list (§11.1 resolution), not a raster cell — dense per-cell
 * build masking was measured and rejected.
 */
export interface Action {
  intentType: number;
  target: number;
  ratioBucket: number;
  spatialCell: number;
  buildSite: number;
}

/** One boolean-as-0/1 mask per head, same indexing as `Action`. */
export interface MaskBuffers {
  intentType: Uint8Array;
  target: Uint8Array;
  ratioBucket: Uint8Array;
  spatialCell: Uint8Array;
  buildSite: Uint8Array;
}

export interface ObservationEncoder {
  /** Fixed length of the encoded observation this encoder produces. */
  readonly size: number;
  encode(game: Game, agentPlayerID: PlayerID, out: Float32Array): void;
}

export interface ActionCodec {
  decode(action: Action, game: Game, playerID: PlayerID): Intent[];
  masks(game: Game, playerID: PlayerID, out: MaskBuffers): void;
}

export interface Info {
  terminalReason: TerminalReason;
  ticks: number;
  /** Next-step legality masks, computed from the post-step state (§5.6). */
  masks: MaskBuffers;
}

export interface StepResult {
  obs: Observation;
  reward: number;
  done: boolean;
  info: Info;
}

export interface EnvConfig {
  map: GameMapType;
  mapsDir: string;
  bots: number;
  /** Simulation ticks advanced per `step()` call. Spec default: 10. */
  ticksPerStep: number;
  /** Post-spawn tick cap — nations don't eliminate each other within 3300
   *  ticks (§5.1), so episodes need a hard cap, not a natural winner. */
  tickCap: number;
  /** Scripted nation peoples: "default" (map manifest) or "disabled".
   *  Curriculum knob — disabled plus few bots is the easiest stage. */
  nations?: "default" | "disabled";
  /** Episode recording for data replays (phase 8). Absent means off. */
  replay?: ReplayOptions;
}

/**
 * Data-replay recording. Every Nth episode per env is written to `dir` as
 * a GameRecord-compatible JSON file the dev client can play back.
 */
export interface ReplayOptions {
  dir: string;
  /** Record every Nth episode (1 means every episode). Must be >= 1. */
  every: number;
}

export const DEFAULT_ENV_CONFIG: Omit<EnvConfig, "map" | "mapsDir"> = {
  bots: 50,
  ticksPerStep: 10,
  tickCap: 6000,
};
