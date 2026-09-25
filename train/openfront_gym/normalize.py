"""Running observation normalization for the OpenFront PPO loop.

Diagnosis (2026-09-25): the flat obs mixes scales — ownership planes in
[0, 1] next to raw troop/gold fractions (troops/1e5, gold/1e7, unbounded,
tiny typical magnitude) with no standardization anywhere
(``ObservationEncoder.ts``, ``adapter.py:10-11`` documents the missing
normalization). The value loss is dominated by the large-magnitude
targets and the policy trunk sees mixed-scale inputs.

This module is a numpy-only VecNormalize equivalent: per-element running
mean/variance over the flat vector, applied as ``clip((x - mean) /
sqrt(var + eps))``. It wraps any vec with a ``reset``/``step``/``num_envs``
interface (duck-typed, so unit tests need no gymnasium or torch), updates
statistics on training steps only, and persists to ``.npz`` beside
checkpoints so eval can reuse frozen stats.
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Parallel Welford update over batches of vectors."""

    def __init__(self, size: int, epsilon: float = 1e-4):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, self.mean.shape[0])
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = (
            self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        ) / total
        self.count = total

    def normalize(self, x: np.ndarray, clip: float = 5.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = (x.astype(np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        if clip is not None:
            out = np.clip(out, -clip, clip)
        return out.astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])

    def save(self, path) -> None:
        np.savez(path, mean=self.mean, var=self.var, count=np.array(self.count))

    @staticmethod
    def load(path, size: int) -> "RunningMeanStd":
        raw = np.load(path, allow_pickle=False)
        stats = RunningMeanStd(size)
        stats.load_state_dict({"mean": raw["mean"], "var": raw["var"], "count": float(raw["count"])})
        return stats


class ObservationNormalizer:
    """Duck-typed vec wrapper: normalizes obs, passes everything else through.

    ``training=True`` updates statistics on every reset/step batch;
    ``training=False`` (eval) applies frozen stats.
    """

    def __init__(self, inner, clip: float = 5.0, training: bool = True):
        self.inner = inner
        self.clip = float(clip)
        self.training = bool(training)
        self.stats: RunningMeanStd | None = None

    @property
    def num_envs(self):
        return self.inner.num_envs

    def _ensure(self, obs: np.ndarray) -> RunningMeanStd:
        if self.stats is None:
            self.stats = RunningMeanStd(obs.shape[-1])
        return self.stats

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        stats = self._ensure(obs)
        if self.training:
            stats.update(np.asarray(obs))
        return stats.normalize(obs, self.clip)

    def reset(self, **kwargs):
        out = self.inner.reset(**kwargs)
        obs, infos = out if isinstance(out, tuple) else (out, None)
        normed = self._normalize(np.asarray(obs))
        return (normed, infos) if infos is not None else normed

    def step(self, actions):
        out = self.inner.step(actions)
        obs, rewards, terminated, truncated, infos = out
        return self._normalize(np.asarray(obs)), rewards, terminated, truncated, infos

    def seed(self, seed=None):
        if hasattr(self.inner, "seed"):
            return self.inner.seed(seed)
        return None

    def close(self) -> None:
        self.inner.close()

    def save(self, path) -> None:
        if self.stats is None:
            raise RuntimeError("no statistics collected yet, nothing to save")
        self.stats.save(path)

    def load(self, path, size: int) -> None:
        self.stats = RunningMeanStd.load(path, size)

    def __getattr__(self, name: str):
        # Masks, spaces, worker logs and SB3 protocol helpers live on the
        # inner vec; attribute lookup falls through to it.
        inner = object.__getattribute__(self, "inner")
        return getattr(inner, name)
