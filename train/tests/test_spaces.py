import unittest

import numpy as np

from openfront_gym import spaces


class SpacesTest(unittest.TestCase):
    def test_head_layout_matches_wire_order(self):
        self.assertEqual(
            spaces.HEAD_SIZES, (14, 17, 8, 8192, 32), "codec/encoder head order changed"
        )
        self.assertEqual(sum(spaces.HEAD_SIZES), 8263)

    def test_action_space_shape(self):
        space = spaces.make_action_space()
        self.assertEqual(tuple(space.nvec), spaces.HEAD_SIZES)
        sample = space.sample()
        self.assertEqual(spaces.action_array_to_dict(sample)["intent_type"], int(sample[0]))

    def test_observation_space_shape(self):
        space = spaces.make_observation_space(8)
        self.assertEqual(space.shape, (8,))
        self.assertEqual(space.dtype, np.float32)

    def test_action_conversion_rejects_short_vectors(self):
        with self.assertRaises(ValueError):
            spaces.action_array_to_dict([1, 2, 3])

    def test_mask_round_trip(self):
        flat = np.zeros(8263, dtype=bool)
        flat[0] = True
        flat[14] = True
        heads = spaces.split_flat_mask(flat)
        self.assertEqual(heads["intent_type"].shape, (14,))
        self.assertEqual(heads["spatial_cell"].shape, (8192,))
        self.assertTrue(heads["intent_type"][0])
        self.assertTrue(heads["target"][0])
        rebuilt = spaces.flatten_mask_dict(
            {name: heads[name].astype(np.uint8) for name in spaces.HEAD_NAMES}
        )
        np.testing.assert_array_equal(rebuilt, flat)

    def test_mask_size_guard(self):
        with self.assertRaises(ValueError):
            spaces.split_flat_mask(np.zeros(10, dtype=bool))


if __name__ == "__main__":
    unittest.main()
