import unittest

import numpy as np

from Script.stage1.render_bvh_to_mp4 import root_motion_display_positions


class Stage1RenderBVHTests(unittest.TestCase):
    def test_world_space_keeps_root_xz_motion(self):
        sampled = np.zeros((2, 2, 3), dtype=np.float64)
        sampled[0, 0] = [1.0, 0.0, 2.0]
        sampled[0, 1] = [1.5, 0.5, 2.5]
        sampled[1, 0] = [3.0, 0.0, 5.0]
        sampled[1, 1] = [3.5, 0.5, 5.5]

        world = root_motion_display_positions(sampled, keep_root_motion=True)

        np.testing.assert_array_equal(world, sampled)


if __name__ == "__main__":
    unittest.main()
