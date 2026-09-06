import type { Game, PlayerID } from "../../../OpenFrontIO/src/core/game/Game";
import type { Intent } from "../../../OpenFrontIO/src/core/Schemas";
import type { Action, ActionCodec, MaskBuffers, ObservationEncoder } from "../../src/types";

/** Fixed-size zero-filled observation — enough to exercise GameEnv's
 *  lifecycle without depending on the real ObservationEncoder. */
export class StubObservationEncoder implements ObservationEncoder {
  readonly size = 4;
  encode(_game: Game, _agentPlayerID: PlayerID, out: Float32Array): void {
    out.fill(0);
  }
}

/** Always decodes to no intents — a "do nothing" agent, sufficient for
 *  testing turn advancement and episode-boundary detection. */
export class NoopActionCodec implements ActionCodec {
  decode(_action: Action, _game: Game, _playerID: PlayerID): Intent[] {
    return [];
  }
  masks(_game: Game, _playerID: PlayerID, out: MaskBuffers): void {
    out.intentType.fill(1);
    out.target.fill(1);
    out.ratioBucket.fill(1);
    out.spatialCell.fill(1);
    out.buildSite.fill(1);
  }
}

export const NOOP_ACTION: Action = {
  intentType: 0,
  target: 0,
  ratioBucket: 0,
  spatialCell: 0,
  buildSite: 0,
};
