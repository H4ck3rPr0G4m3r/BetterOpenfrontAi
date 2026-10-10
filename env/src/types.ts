import type { GameMapType } from "../../OpenFrontIO/src/core/game/Game";

/**
 * One seat's action: five heads, conditioned on intentType (see Layout.ts
 * HEAD_USAGE). Heads the intent does not use are ignored by decode().
 */
export interface Action {
  intentType: number;
  target: number;
  ratioBucket: number;
  spatialCell: number;
  buildSite: number;
}

export const NOOP_ACTION: Action = {
  intentType: 0,
  target: 0,
  ratioBucket: 0,
  spatialCell: 0,
  buildSite: 0,
};

/** Why a seat's episode ended. "aborted" = the trainer ended the game early. */
export type TerminalReason = "win" | "death" | "tick_cap" | "sim_error" | "aborted" | null;

export interface EnvConfig {
  map: GameMapType;
  mapsDir: string;
  bots: number;
  /** Simulation ticks advanced per `step()` call. */
  ticksPerStep: number;
  /** Post-spawn tick cap. Episodes end there with a placement reward. */
  tickCap: number;
  /** Agent-controlled human seats in one game (1 = solo vs bots). */
  seats: number;
  /** Scripted nations: "default" (map manifest) or "disabled". */
  nations?: "default" | "disabled";
  /** Episode recording for data replays. Absent means off. */
  replay?: ReplayOptions;
}

/** Every Nth game per env is written to `dir` as a GameRecord JSON file. */
export interface ReplayOptions {
  dir: string;
  every: number;
}
