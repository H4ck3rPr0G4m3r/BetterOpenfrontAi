import unittest

import numpy as np

from openfront_gym import protocol
from openfront_gym.env import OpenFrontEnv
from openfront_gym.worker import WorkerClient, WorkerConfig
from tests.fake_worker import FakeWorker


def make_env(fake: FakeWorker, **overrides) -> OpenFrontEnv:
    config = WorkerConfig(
        num_envs=fake.num_envs, socket_timeout=10.0, spawn_timeout=10.0
    )

    def factory() -> WorkerClient:
        return WorkerClient.connect("127.0.0.1", fake.port, config)

    return OpenFrontEnv(worker_factory=factory, **overrides)


class SingleEnvTest(unittest.TestCase):
    def test_reset_returns_shaped_obs_and_masks(self):
        fake = FakeWorker(num_envs=1, obs_size=8)
        fake.start()
        try:
            env = make_env(fake)
            obs, info = env.reset(seed="hello")
            self.assertEqual(obs.shape, (8,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertIn("action_masks", info)
            self.assertEqual(info["action_masks"].shape, (8263,))
            self.assertEqual(info["ticks"], 302)
            self.assertIsNone(info["terminal_reason"])
            np.testing.assert_array_equal(info["action_masks"], env.action_masks())
            env.close()
        finally:
            fake.stop()

    def test_step_advances_ticks_with_shaping_reward(self):
        fake = FakeWorker(num_envs=1, obs_size=8)
        fake.start()
        try:
            env = make_env(fake)
            env.reset(seed="s")
            obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.int64))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(reward, 0.1)
            self.assertEqual(info["ticks"], 312)
            env.close()
        finally:
            fake.stop()

    def test_terminal_mapping(self):
        cases = [
            (protocol.TerminalCode.WIN, True, False, "win"),
            (protocol.TerminalCode.DEATH, True, False, "death"),
            (protocol.TerminalCode.TICK_CAP, False, True, "tick_cap"),
            (protocol.TerminalCode.SIM_ERROR, True, False, "sim_error"),
        ]
        for code, terminated, truncated, reason in cases:
            fake = FakeWorker(num_envs=1, obs_size=8, terminals={0: [code]})
            fake.start()
            try:
                env = make_env(fake)
                env.reset(seed="s")
                _, _, is_term, is_trunc, info = env.step(np.zeros(5, dtype=np.int64))
                self.assertEqual(is_term, terminated, reason)
                self.assertEqual(is_trunc, truncated, reason)
                self.assertEqual(info["terminal_reason"], reason)
                env.close()
            finally:
                fake.stop()

    def test_done_reply_carries_fresh_episode(self):
        fake = FakeWorker(
            num_envs=1, obs_size=8, terminals={0: [protocol.TerminalCode.WIN]}
        )
        fake.start()
        try:
            env = make_env(fake)
            first, _ = env.reset(seed="s")
            _, _, terminated, _, _ = env.step(np.zeros(5, dtype=np.int64))
            self.assertTrue(terminated)
            second, _, terminated, _, info = env.step(np.zeros(5, dtype=np.int64))
            self.assertFalse(terminated)
            self.assertFalse(np.array_equal(first, second))
            self.assertEqual(info["ticks"], 312)
            env.close()
        finally:
            fake.stop()

    def test_gymnasium_compliance(self):
        from gymnasium.utils.env_checker import check_env

        fake = FakeWorker(num_envs=1, obs_size=8)
        fake.start()
        try:
            check_env(make_env(fake), skip_render_check=True)
        finally:
            fake.stop()

    def test_worker_death_ends_episode_and_recovers(self):
        first = FakeWorker(num_envs=1, obs_size=8)
        first.start()
        second_holder: list[FakeWorker] = []

        config = WorkerConfig(num_envs=1, socket_timeout=5.0, spawn_timeout=5.0)
        fakes = [first]

        def factory() -> WorkerClient:
            return WorkerClient.connect("127.0.0.1", fakes[0].port, config)

        env = OpenFrontEnv(worker_factory=factory)
        try:
            env.reset(seed="s")
            replacement = FakeWorker(num_envs=1, obs_size=8)
            replacement.start()
            second_holder.append(replacement)
            fakes[0] = replacement
            first.kill()
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(5, dtype=np.int64)
            )
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["terminal_reason"], "worker_death")
            self.assertEqual(obs.shape, (8,))
            # The env works again on the replacement worker.
            _, _, terminated, _, _ = env.step(np.zeros(5, dtype=np.int64))
            self.assertFalse(terminated)
            env.close()
        finally:
            first.stop()
            for fake in second_holder:
                fake.stop()

    def test_close_is_idempotent(self):
        fake = FakeWorker(num_envs=1, obs_size=8)
        fake.start()
        try:
            env = make_env(fake)
            env.reset(seed="s")
            env.close()
            env.close()
        finally:
            fake.stop()


if __name__ == "__main__":
    unittest.main()
