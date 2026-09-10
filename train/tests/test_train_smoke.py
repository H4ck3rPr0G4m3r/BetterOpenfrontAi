"""Phase 6 smoke coverage: adapter, SB3 translation, PPO loop, eval.

Live Node workers stay out of unit tests. Everything here runs against
``FakeWorker``. The fake speaks the real wire protocol, so the adapter
and rollout paths are genuine and only the simulation is stubbed.
"""

import tempfile
import unittest

import numpy as np

from openfront_gym import protocol, spaces
from openfront_gym.adapter import (
    NUM_GLOBALS,
    OBSERVATION_SIZE,
    TABLE_SIZE,
    split_batch,
    split_observation,
)
from openfront_gym.vec import OpenFrontVecEnv
from openfront_gym.worker import WorkerClient, WorkerConfig
from sb3_adapter import SB3OpenFrontVec
from tests.fake_worker import FakeWorker


def factory_for(fakes: list[FakeWorker], num_envs: int):
    config = WorkerConfig(num_envs=num_envs, socket_timeout=5.0, spawn_timeout=5.0)

    def factory(worker_index: int) -> WorkerClient:
        return WorkerClient.connect("127.0.0.1", fakes[worker_index].port, config)

    return factory


def mask_with(*, intent=(), target=(), ratio=None, spatial=(), site=()):
    flat = np.zeros(spaces.MASK_TOTAL, dtype=bool)
    heads = spaces.split_flat_mask(flat)
    for index in intent:
        heads["intent_type"][index] = True
    for index in target:
        heads["target"][index] = True
    if ratio is None:
        heads["ratio_bucket"][:] = True
    else:
        for index in ratio:
            heads["ratio_bucket"][index] = True
    for index in spatial:
        heads["spatial_cell"][index] = True
    for index in site:
        heads["build_site"][index] = True
    return np.concatenate([heads[name] for name in spaces.HEAD_NAMES])


class AdapterTest(unittest.TestCase):
    def test_split_shapes(self):
        parts = split_observation(np.arange(OBSERVATION_SIZE, dtype=np.float32))
        self.assertEqual(parts["spatial"].shape, (8, 64, 128))
        self.assertEqual(parts["table"].shape, (17, 8))
        self.assertEqual(parts["globals"].shape, (12,))
        self.assertEqual(parts["spatial"][0, 0, 0], 0.0)
        self.assertEqual(parts["table"][0, 0], 65536.0)
        self.assertEqual(parts["globals"][0], 65536.0 + TABLE_SIZE)

    def test_split_batch(self):
        batch = np.zeros((3, OBSERVATION_SIZE), dtype=np.float32)
        parts = split_batch(batch)
        self.assertEqual(parts["spatial"].shape, (3, 8, 64, 128))
        self.assertEqual(parts["table"].shape, (3, 17, 8))
        self.assertEqual(parts["globals"].shape, (3, NUM_GLOBALS))

    def test_bad_size_raises(self):
        with self.assertRaises(ValueError):
            split_observation(np.zeros(8, dtype=np.float32))
        with self.assertRaises(ValueError):
            split_batch(np.zeros((2, 8), dtype=np.float32))


class SB3AdapterTest(unittest.TestCase):
    def test_reset_returns_obs_only(self):
        fake = FakeWorker(num_envs=2, obs_size=8)
        fake.start()
        try:
            inner = OpenFrontVecEnv(1, 2, worker_factory=factory_for([fake], 2))
            vec = SB3OpenFrontVec(inner)
            obs = vec.reset()
            self.assertEqual(obs.shape, (2, 8))
            vec.close()
        finally:
            fake.stop()

    def test_step_wait_matches_sb3_tuple(self):
        fake = FakeWorker(num_envs=2, obs_size=8)
        fake.start()
        try:
            inner = OpenFrontVecEnv(1, 2, worker_factory=factory_for([fake], 2))
            vec = SB3OpenFrontVec(inner)
            vec.reset()
            vec.step_async(np.zeros((2, 5), dtype=np.int64))
            obs, rewards, dones, infos = vec.step_wait()
            self.assertEqual(obs.shape, (2, 8))
            self.assertEqual(rewards.shape, (2,))
            self.assertEqual(dones.shape, (2,))
            self.assertFalse(dones.any())
            self.assertIsInstance(infos, list)
            self.assertIsNone(infos[0]["terminal_reason"])
            self.assertFalse(infos[0]["TimeLimit.truncated"])
            vec.close()
        finally:
            fake.stop()

    def test_tick_cap_bootstraps_from_fresh_obs(self):
        fake = FakeWorker(
            num_envs=1, obs_size=8, terminals={0: [protocol.TerminalCode.TICK_CAP]}
        )
        fake.start()
        try:
            inner = OpenFrontVecEnv(1, 1, worker_factory=factory_for([fake], 1))
            vec = SB3OpenFrontVec(inner)
            vec.reset()
            vec.step_async(np.zeros((1, 5), dtype=np.int64))
            obs, _, dones, infos = vec.step_wait()
            self.assertTrue(dones[0])
            self.assertEqual(infos[0]["terminal_reason"], "tick_cap")
            self.assertTrue(infos[0]["TimeLimit.truncated"])
            np.testing.assert_array_equal(infos[0]["terminal_observation"], obs[0])
            self.assertEqual(infos[0]["episode"]["l"], 1)
            vec.close()
        finally:
            fake.stop()

    def test_win_ends_advantage_chain(self):
        fake = FakeWorker(
            num_envs=1, obs_size=8, terminals={0: [protocol.TerminalCode.WIN]}
        )
        fake.start()
        try:
            inner = OpenFrontVecEnv(1, 1, worker_factory=factory_for([fake], 1))
            vec = SB3OpenFrontVec(inner)
            vec.reset()
            vec.step_async(np.zeros((1, 5), dtype=np.int64))
            _, rewards, dones, infos = vec.step_wait()
            self.assertTrue(dones[0])
            self.assertEqual(infos[0]["terminal_reason"], "win")
            self.assertFalse(infos[0]["TimeLimit.truncated"])
            self.assertNotIn("terminal_observation", infos[0])
            self.assertAlmostEqual(infos[0]["episode"]["r"], float(rewards[0]))
            self.assertEqual(vec.terminal_counts["win"], 1)
            vec.close()
        finally:
            fake.stop()

    def test_mask_convention(self):
        fake = FakeWorker(num_envs=2, obs_size=8)
        fake.start()
        try:
            inner = OpenFrontVecEnv(1, 2, worker_factory=factory_for([fake], 2))
            vec = SB3OpenFrontVec(inner)
            self.assertTrue(vec.has_attr("action_masks"))
            stacked = np.stack(vec.env_method("action_masks"))
            self.assertEqual(stacked.shape, (2, spaces.MASK_TOTAL))
            heads = spaces.split_flat_mask(stacked[0])
            self.assertEqual(
                [heads[name].shape[0] for name in spaces.HEAD_NAMES],
                [14, 17, 8, 8192, 32],
            )
            self.assertTrue(stacked.all())
            subset = vec.env_method("action_masks", indices=[1])
            self.assertEqual(len(subset), 1)
            np.testing.assert_array_equal(subset[0], stacked[1])
            vec.close()
        finally:
            fake.stop()

    def test_step_wait_without_async_raises(self):
        fake = FakeWorker(num_envs=1, obs_size=8)
        fake.start()
        try:
            inner = OpenFrontVecEnv(1, 1, worker_factory=factory_for([fake], 1))
            vec = SB3OpenFrontVec(inner)
            with self.assertRaises(RuntimeError):
                vec.step_wait()
            vec.close()
        finally:
            fake.stop()


class TrainSmokeTest(unittest.TestCase):
    def _make_model(self, vec):
        from sb3_contrib import MaskablePPO

        from openfront_policy import OpenFrontPolicy

        return MaskablePPO(
            OpenFrontPolicy,
            vec,
            verbose=0,
            seed=0,
            device="cpu",
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            policy_kwargs={
                "features_extractor_kwargs": {"features_dim": 128},
                "net_arch": dict(pi=[64], vf=[64]),
            },
        )

    def test_ppo_learns_against_fake_bridge(self):
        fake = FakeWorker(num_envs=1, obs_size=OBSERVATION_SIZE)
        fake.start()
        try:
            inner = OpenFrontVecEnv(1, 1, worker_factory=factory_for([fake], 1))
            vec = SB3OpenFrontVec(inner)
            model = self._make_model(vec)
            model.learn(total_timesteps=16)
            self.assertEqual(model.num_timesteps, 16)
            loss = model.logger.name_to_value.get("train/loss")
            self.assertIsNotNone(loss)
            self.assertTrue(np.isfinite(loss))
            vec.close()
        finally:
            fake.stop()

    def test_checkpoint_round_trips_identical_actions(self):
        from sb3_contrib import MaskablePPO

        fake = FakeWorker(num_envs=1, obs_size=OBSERVATION_SIZE)
        fake.start()
        try:
            inner = OpenFrontVecEnv(1, 1, worker_factory=factory_for([fake], 1))
            vec = SB3OpenFrontVec(inner)
            model = self._make_model(vec)
            model.learn(total_timesteps=16)
            obs = vec.reset()
            masks = vec.action_masks()
            first, _ = model.predict(obs, deterministic=True, action_masks=masks)
            with tempfile.TemporaryDirectory() as tmp:
                path = f"{tmp}/smoke"
                model.save(path)
                reloaded = MaskablePPO.load(path, device="cpu")
            second, _ = reloaded.predict(obs, deterministic=True, action_masks=masks)
            np.testing.assert_array_equal(first, second)
            vec.close()
        finally:
            fake.stop()

    def test_masked_sampling_never_picks_shut_entries(self):
        fake = FakeWorker(num_envs=1, obs_size=OBSERVATION_SIZE)
        fake.start()
        try:
            inner = OpenFrontVecEnv(1, 1, worker_factory=factory_for([fake], 1))
            vec = SB3OpenFrontVec(inner)
            model = self._make_model(vec)
            obs = vec.reset()
            # Exactly one open entry per head: any sample must equal it.
            masks = mask_with(intent=[3], target=[2], ratio=[4], spatial=[77], site=[5]).reshape(1, -1)
            for _ in range(3):
                actions, _ = model.predict(obs, deterministic=False, action_masks=masks)
                np.testing.assert_array_equal(actions[0], [3, 2, 4, 77, 5])
            vec.close()
        finally:
            fake.stop()

    def test_eval_loop_reports_terminals(self):
        from eval import policy_predictor, run_episodes

        from openfront_policy import OpenFrontPolicy  # noqa: F401

        fake = FakeWorker(
            num_envs=1, obs_size=OBSERVATION_SIZE, terminals={0: [protocol.TerminalCode.WIN]}
        )
        fake.start()
        try:
            inner = OpenFrontVecEnv(1, 1, worker_factory=factory_for([fake], 1))
            vec = SB3OpenFrontVec(inner)
            model = self._make_model(vec)
            stats = run_episodes(policy_predictor(model), inner, 1, ["e0"])
            self.assertEqual(stats["episodes"], 1)
            self.assertEqual(stats["terminals"], {"win": 1})
            self.assertEqual(stats["win_rate"], 1.0)
            self.assertEqual(stats["mean_length"], 1.0)
            vec.close()
        finally:
            fake.stop()


class ReplayEnvTest(unittest.TestCase):
    def test_apply_replay_env_exports_and_clears(self):
        import os
        from train_ppo import apply_replay_env

        os.environ.pop("OPENFRONT_REPLAY_DIR", None)
        os.environ.pop("OPENFRONT_REPLAY_EVERY", None)
        apply_replay_env({"run": {"replay": {"dir": "replays/x", "every": 50}}})
        self.assertTrue(
            os.environ["OPENFRONT_REPLAY_DIR"].endswith(os.path.join("replays", "x"))
        )
        self.assertEqual(os.environ["OPENFRONT_REPLAY_EVERY"], "50")
        apply_replay_env({"run": {}})
        self.assertNotIn("OPENFRONT_REPLAY_DIR", os.environ)
        self.assertNotIn("OPENFRONT_REPLAY_EVERY", os.environ)

    def test_apply_replay_env_rejects_bad_every(self):
        from train_ppo import apply_replay_env

        with self.assertRaises(ValueError):
            apply_replay_env({"run": {"replay": {"dir": "replays/x", "every": 0}}})


if __name__ == "__main__":
    unittest.main()
