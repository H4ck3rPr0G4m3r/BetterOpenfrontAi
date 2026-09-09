import unittest
import warnings

import numpy as np

from openfront_gym import protocol
from openfront_gym.vec import OpenFrontVecEnv
from openfront_gym.worker import WorkerClient, WorkerConfig
from tests.fake_worker import FakeWorker


def factory_for(fakes: list[FakeWorker], num_envs: int):
    config = WorkerConfig(num_envs=num_envs, socket_timeout=5.0, spawn_timeout=5.0)

    def factory(worker_index: int) -> WorkerClient:
        return WorkerClient.connect("127.0.0.1", fakes[worker_index].port, config)

    return factory


class VecEnvTest(unittest.TestCase):
    def test_batched_reset_and_step_shapes(self):
        fakes = [FakeWorker(num_envs=2, obs_size=8) for _ in range(2)]
        for fake in fakes:
            fake.start()
        try:
            vec = OpenFrontVecEnv(
                2, 2, worker_factory=factory_for(fakes, 2), seeds=["a", "b", "c", "d"]
            )
            obs, infos = vec.reset()
            self.assertEqual(obs.shape, (4, 8))
            self.assertEqual(infos["action_masks"].shape, (4, 8263))
            self.assertEqual(list(infos["ticks"]), [302] * 4)
            actions = np.zeros((4, 5), dtype=np.int64)
            obs, rewards, terminated, truncated, infos = vec.step(actions)
            self.assertEqual(obs.shape, (4, 8))
            self.assertEqual(rewards.shape, (4,))
            self.assertFalse(terminated.any())
            self.assertFalse(truncated.any())
            vec.close()
        finally:
            for fake in fakes:
                fake.stop()

    def test_done_rows_carry_fresh_episodes(self):
        fakes = [
            FakeWorker(num_envs=2, obs_size=8, terminals={0: [protocol.TerminalCode.WIN]}),
            FakeWorker(num_envs=2, obs_size=8),
        ]
        for fake in fakes:
            fake.start()
        try:
            vec = OpenFrontVecEnv(2, 2, worker_factory=factory_for(fakes, 2))
            vec.reset()
            _, _, terminated, _, infos = vec.step(np.zeros((4, 5), dtype=np.int64))
            self.assertEqual(list(terminated), [True, False, False, False])
            self.assertEqual(list(infos["terminal_reason"])[0], "win")
            _, _, terminated, _, _ = vec.step(np.zeros((4, 5), dtype=np.int64))
            self.assertFalse(terminated.any())
            vec.close()
        finally:
            for fake in fakes:
                fake.stop()

    def test_mask_retrieval_convention(self):
        fakes = [FakeWorker(num_envs=2, obs_size=8)]
        fakes[0].start()
        try:
            vec = OpenFrontVecEnv(1, 2, worker_factory=factory_for(fakes, 2))
            masks = vec.action_masks()
            self.assertEqual(masks.shape, (2, 8263))
            self.assertTrue(vec.has_attr("action_masks"))
            listed = vec.env_method("action_masks")
            self.assertEqual(len(listed), 2)
            np.testing.assert_array_equal(np.stack(listed), masks)
            vec.close()
        finally:
            fakes[0].stop()

    def test_worker_death_respawns_and_marks_rows_done(self):
        replacement = FakeWorker(num_envs=2, obs_size=8)
        replacement.start()
        doomed = FakeWorker(num_envs=2, obs_size=8)
        doomed.start()
        fakes = [FakeWorker(num_envs=2, obs_size=8), doomed]
        fakes[0].start()
        try:
            vec = OpenFrontVecEnv(2, 2, worker_factory=factory_for(fakes, 2))
            vec.reset()
            fakes[1] = replacement
            doomed.kill()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _, rewards, terminated, truncated, infos = vec.step(
                    np.zeros((4, 5), dtype=np.int64)
                )
            self.assertEqual(list(terminated), [False, False, True, True])
            self.assertFalse(truncated.any())
            self.assertEqual(
                list(infos["terminal_reason"])[2:], ["worker_death", "worker_death"]
            )
            # The respawned worker serves the next step normally.
            _, _, terminated, _, _ = vec.step(np.zeros((4, 5), dtype=np.int64))
            self.assertFalse(terminated.any())
            vec.close()
        finally:
            for fake in (fakes[0], replacement, doomed):
                try:
                    fake.stop()
                except OSError:
                    pass

    def test_reset_seed_is_deterministic(self):
        fakes = [FakeWorker(num_envs=2, obs_size=8)]
        fakes[0].start()
        try:
            vec = OpenFrontVecEnv(1, 2, worker_factory=factory_for(fakes, 2))
            first, _ = vec.reset(seed=7)
            second, _ = vec.reset(seed=7)
            np.testing.assert_array_equal(first, second)
            vec.close()
        finally:
            fakes[0].stop()

    def test_reset_accepts_per_env_seeds(self):
        fakes = [FakeWorker(num_envs=2, obs_size=8)]
        fakes[0].start()
        try:
            vec = OpenFrontVecEnv(1, 2, worker_factory=factory_for(fakes, 2))
            first, _ = vec.reset(seed=["alpha", "beta"])
            second, _ = vec.reset(seed=["alpha", "beta"])
            np.testing.assert_array_equal(first, second)
            third, _ = vec.reset(seed=["alpha", "gamma"])
            self.assertFalse(np.array_equal(first, third))
            vec.close()
        finally:
            fakes[0].stop()


if __name__ == "__main__":
    unittest.main()
