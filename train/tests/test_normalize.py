"""Tests for the running observation normalizer (numpy only, no gym/torch)."""

import numpy as np

from openfront_gym.normalize import ObservationNormalizer, RunningMeanStd


class FakeVec:
    def __init__(self, obs):
        self._obs = np.asarray(obs, dtype=np.float32)
        self.closed = False

    @property
    def num_envs(self):
        return self._obs.shape[0]

    def reset(self, **kwargs):
        return self._obs.copy(), {"action_masks": None}

    def step(self, actions):
        return self._obs.copy(), np.zeros(self.num_envs), np.zeros(self.num_envs, bool), np.zeros(
            self.num_envs, bool
        ), {}

    def close(self):
        self.closed = True


def test_running_stats_match_batch_moments():
    stats = RunningMeanStd(3)
    rng = np.random.default_rng(0)
    batch = rng.normal(loc=[1.0, 2.0, 3.0], scale=[2.0, 0.5, 4.0], size=(500, 3)).astype(np.float32)
    stats.update(batch)
    np.testing.assert_allclose(stats.mean, batch.mean(axis=0), rtol=1e-3)
    np.testing.assert_allclose(stats.var, batch.var(axis=0), rtol=1e-3)


def test_normalize_clips_and_standardizes():
    stats = RunningMeanStd(2)
    stats.update(np.array([[0.0, 100.0], [2.0, 102.0]], dtype=np.float32))
    out = stats.normalize(np.array([[1.0, 101.0]], dtype=np.float32), clip=5.0)
    # ~0 up to the 1e-4 epsilon prior pulling the first update slightly.
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=0.05)
    out = stats.normalize(np.array([[1000.0, -1000.0]], dtype=np.float32), clip=5.0)
    assert out.max() <= 5.0 and out.min() >= -5.0


def test_save_load_roundtrip(tmp_path):
    stats = RunningMeanStd(2)
    stats.update(np.ones((4, 2), dtype=np.float32))
    path = tmp_path / "norm.npz"
    stats.save(path)
    restored = RunningMeanStd.load(path, 2)
    np.testing.assert_allclose(restored.mean, stats.mean)
    np.testing.assert_allclose(restored.var, stats.var)


def test_wrapper_updates_while_training_and_freezes_in_eval():
    obs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    wrapper = ObservationNormalizer(FakeVec(obs), training=True)
    normed, infos = wrapper.reset()
    assert normed.shape == obs.shape
    assert infos == {"action_masks": None}
    count_after_reset = wrapper.stats.count
    wrapper.step([0, 0])
    assert wrapper.stats.count > count_after_reset

    frozen = ObservationNormalizer(FakeVec(obs), training=False)
    frozen.stats = wrapper.stats
    before = (frozen.stats.mean.copy(), frozen.stats.var.copy())
    frozen.reset()
    np.testing.assert_array_equal(frozen.stats.mean, before[0])
    np.testing.assert_array_equal(frozen.stats.var, before[1])


def test_wrapper_passes_through_everything_else():
    obs = np.zeros((1, 4), dtype=np.float32)
    inner = FakeVec(obs)
    wrapper = ObservationNormalizer(inner, training=True)
    assert wrapper.num_envs == 1
    _, rewards, terminated, truncated, _ = wrapper.step([0])
    assert rewards.shape == (1,)
    assert not terminated[0] and not truncated[0]
    wrapper.close()
    assert inner.closed
