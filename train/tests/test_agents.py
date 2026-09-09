import unittest

import numpy as np

from agents.random_agent import MaskedRandomAgent
from agents.scripted_agent import ScriptedBaselineAgent
from openfront_gym import spaces


def mask_with(*, intent=(), target=(), spatial=(), site=()):
    flat = np.zeros(spaces.MASK_TOTAL, dtype=bool)
    heads = spaces.split_flat_mask(flat)
    for index in intent:
        heads["intent_type"][index] = True
    for index in target:
        heads["target"][index] = True
    heads["ratio_bucket"][:] = True
    for index in spatial:
        heads["spatial_cell"][index] = True
    for index in site:
        heads["build_site"][index] = True
    return np.concatenate([heads[name] for name in spaces.HEAD_NAMES])


class AgentsTest(unittest.TestCase):
    def test_random_agent_only_picks_open_entries(self):
        agent = MaskedRandomAgent(seed=0)
        masks = mask_with(intent=[1], target=[0], spatial=[5], site=[2])
        for _ in range(20):
            action = agent.act(np.zeros(8, dtype=np.float32), masks)
            self.assertEqual(list(action), [1, 0, int(action[2]), 5, 2])
            self.assertIn(int(action[2]), range(8))

    def test_random_agent_handles_fully_shut_heads(self):
        agent = MaskedRandomAgent(seed=0)
        masks = np.zeros(spaces.MASK_TOTAL, dtype=bool)
        action = agent.act(np.zeros(8, dtype=np.float32), masks)
        np.testing.assert_array_equal(action, np.zeros(5, dtype=np.int64))

    def test_random_agent_is_seeded(self):
        masks = mask_with(
            intent=[1, 2], target=[0, 1], spatial=[5, 6], site=[2, 3]
        )
        first = MaskedRandomAgent(seed=3).act(np.zeros(8, dtype=np.float32), masks)
        second = MaskedRandomAgent(seed=3).act(np.zeros(8, dtype=np.float32), masks)
        np.testing.assert_array_equal(first, second)

    def test_scripted_agent_expands_first(self):
        agent = ScriptedBaselineAgent()
        masks = mask_with(intent=[0, 1], target=[0, 4], spatial=[9], site=[1])
        action = agent.act(np.zeros(8, dtype=np.float32), masks)
        self.assertEqual(int(action[0]), spaces.IntentType.ATTACK)
        self.assertEqual(int(action[1]), 0)

    def test_scripted_agent_attacks_players_when_no_open_land(self):
        agent = ScriptedBaselineAgent()
        masks = mask_with(intent=[0, 1], target=[4], spatial=[9], site=[1])
        action = agent.act(np.zeros(8, dtype=np.float32), masks)
        self.assertEqual((int(action[0]), int(action[1])), (1, 4))

    def test_scripted_agent_falls_back_down_the_priority_list(self):
        agent = ScriptedBaselineAgent()
        obs = np.zeros(8, dtype=np.float32)
        boat = agent.act(obs, mask_with(intent=[0, 2], spatial=[77]))
        self.assertEqual((int(boat[0]), int(boat[3])), (2, 77))
        build = agent.act(obs, mask_with(intent=[0, 3], site=[11]))
        self.assertEqual((int(build[0]), int(build[4])), (3, 11))
        noop = agent.act(obs, mask_with(intent=[0]))
        self.assertEqual(int(noop[0]), 0)

    def test_scripted_agent_is_deterministic(self):
        agent = ScriptedBaselineAgent()
        masks = mask_with(intent=[0, 1, 2, 3], target=[0, 4], spatial=[9], site=[1])
        obs = np.zeros(8, dtype=np.float32)
        np.testing.assert_array_equal(agent.act(obs, masks), agent.act(obs, masks))

    def test_batch_acting_shapes(self):
        obs_batch = np.zeros((3, 8), dtype=np.float32)
        masks_batch = np.stack(
            [mask_with(intent=[0, 1], target=[0]) for _ in range(3)]
        )
        for agent in (MaskedRandomAgent(seed=1), ScriptedBaselineAgent()):
            actions = agent.act_batch(obs_batch, masks_batch)
            self.assertEqual(actions.shape, (3, 5))


if __name__ == "__main__":
    unittest.main()
