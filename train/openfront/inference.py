"""One predictor interface for every consumer: evaluation, AI-vs-AI matches,
the dashboard live view, human-vs-AI play, and real openfront.io lobbies.

A predictor spec is a string:
  ``scripted`` | ``random`` | ``noop`` | a path to a v2 checkpoint (``.pt``)

``min_attack_ratio`` floors the ratio bucket of attacks at inference time
(the dashboard's aggression slider). It works on any predictor without
retraining.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import layout as L
from .agents import make_agent

AGENT_KINDS = ("scripted", "random", "noop")


def aggression_to_ratio_floor(level: int) -> int:
    """Aggression 0..100 into a minimum attack ratio bucket (shown in the UI):
    0..33 keeps every bucket, 34..66 floors attacks at 20%, 67..100 at 50%."""
    if not 0 <= level <= 100:
        raise ValueError(f"aggression must be 0..100, got {level}")
    return 0 if level <= 33 else 2 if level <= 66 else 4


class Predictor:
    def __init__(self, spec: str, device: str = "cpu", deterministic: bool = False, min_attack_ratio: int = 0, seed: int = 0):
        self.spec = str(spec)
        self.deterministic = deterministic
        self.min_attack_ratio = int(min_attack_ratio)
        if not 0 <= self.min_attack_ratio < L.NUM_RATIOS:
            raise ValueError("min_attack_ratio must be 0..7")
        self.model = None
        self.agent = None
        self.device = device
        if self.spec in AGENT_KINDS:
            self.agent = make_agent(self.spec, seed)
        else:
            path = Path(self.spec)
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {self.spec}")
            from .model import load_checkpoint

            self.model, payload = load_checkpoint(path, device)
            self.model.eval()
            self.meta = payload.get("meta", {})

    @property
    def label(self) -> str:
        return self.spec if self.agent is not None else Path(self.spec).stem

    def __call__(self, raster: np.ndarray, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.zeros((0, 5), np.int64)
        if self.agent is not None:
            actions = self.agent.act(raster, features, masks)
        else:
            import torch

            with torch.no_grad():
                out = self.model.act(
                    torch.as_tensor(np.ascontiguousarray(raster), device=self.device),
                    torch.as_tensor(np.ascontiguousarray(features), device=self.device),
                    torch.as_tensor(np.ascontiguousarray(masks), device=self.device),
                    deterministic=self.deterministic,
                )
            actions = out["actions"].cpu().numpy()
        return self._floor_attacks(actions, masks)

    def _floor_attacks(self, actions: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if self.min_attack_ratio == 0:
            return actions
        actions = np.array(actions, copy=True)
        ratio_mask = L.split_masks(np.asarray(masks, dtype=bool))["ratio"]
        for i in np.flatnonzero(actions[:, 0] == L.ATTACK):
            # Raise to the floor, but never past the troop reserve's mask.
            open_ = np.flatnonzero(ratio_mask[i])
            allowed = open_[open_ <= max(self.min_attack_ratio, actions[i, 2])]
            actions[i, 2] = int(allowed[-1]) if len(allowed) else 0
        return actions
