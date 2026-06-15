import unittest

import numpy as np

from Script.stage1.make_bvh_contact_sheet import select_quality_rows
from pathlib import Path

from Script.stage1.render_bvh_to_mp4 import display_label_for_bvh, root_motion_display_positions


class Stage1RenderBVHTests(unittest.TestCase):
    def test_world_space_keeps_root_xz_motion(self):
        sampled = np.zeros((2, 2, 3), dtype=np.float64)
        sampled[0, 0] = [1.0, 0.0, 2.0]
        sampled[0, 1] = [1.5, 0.5, 2.5]
        sampled[1, 0] = [3.0, 0.0, 5.0]
        sampled[1, 1] = [3.5, 0.5, 5.5]

        world = root_motion_display_positions(sampled, keep_root_motion=True)

        np.testing.assert_array_equal(world, sampled)

    def test_display_label_hides_humanml3d_train_prefix(self):
        label = display_label_for_bvh(Path("train_000057__baseline_top_p.bvh"))

        self.assertEqual(label, "Prompt 057 | Baseline")
        self.assertNotIn("train", label.lower())

    def test_display_label_formats_long100_prompt(self):
        label = display_label_for_bvh(Path("test_long_012__finetuned_top_p.bvh"))

        self.assertEqual(label, "Long100 012 | Fine-tuned")

    def test_display_label_formats_clean_prompt_name(self):
        label = display_label_for_bvh(Path("prompt_099__finetuned_top_p.bvh"))

        self.assertEqual(label, "Prompt 099 | Fine-tuned")

    def test_contact_sheet_selects_limited_accepted_and_rejected_rows(self):
        payload = {
            "rows": [
                {"label": "a0", "accepted": True},
                {"label": "a1", "accepted": True},
                {"label": "r0", "accepted": False},
                {"label": "r1", "accepted": False},
                {"label": "r2", "accepted": False},
            ]
        }

        rows = select_quality_rows(payload, selection="both", limit_per_class=2)

        self.assertEqual([row["label"] for row in rows], ["a0", "a1", "r0", "r1"])


if __name__ == "__main__":
    unittest.main()
