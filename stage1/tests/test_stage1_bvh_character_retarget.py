import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch


class _FakeAgent:
    obs_mean = torch.zeros(323, dtype=torch.float32)
    obs_std = torch.ones(323, dtype=torch.float32)

    def encode_seq_all(self, obs, target):
        length = int(target.shape[0])
        indices = torch.arange(length * 4, dtype=torch.long).reshape(4, 1, length) % 512
        return {"indexs": indices}


class Stage1BVHCharacterRetargetTests(unittest.TestCase):
    def test_diagnose_bvh_character_retarget_compares_to_native_distribution(self):
        from Script.stage1 import diagnose_bvh_character_retarget as diag

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            native_h5 = tmp / "native.h5"
            with h5py.File(native_h5, "w") as h5:
                h5.create_dataset("motion/observation", data=np.zeros((6, 323), dtype=np.float32))

            fake_motion = {
                "state": np.zeros((5, 20, 13), dtype=np.float32),
                "observation": np.ones((5, 323), dtype=np.float32) * 0.25,
                "done": np.zeros((5, 1), dtype=bool),
            }

            with mock.patch.object(diag, "extract_bvh_with_moconvq_character", return_value=fake_motion):
                summary = diag.diagnose_bvh_character_retarget(
                    [tmp / "toy.bvh"],
                    agent=_FakeAgent(),
                    fps=20,
                    rvq_depth=4,
                    native_h5=native_h5,
                    native_observation_key="motion/observation",
                )

        self.assertEqual(len(summary["summaries"]), 2)
        self.assertEqual(summary["summaries"][0]["kind"], "bvh_character")
        self.assertEqual(summary["summaries"][0]["state_shape"], [5, 20, 13])
        self.assertEqual(summary["summaries"][0]["observation_shape"], [5, 323])
        self.assertIn("aggregate_abs_z", summary["summaries"][0]["observation_z"])
        self.assertEqual(len(summary["summaries"][0]["stats"]), 4)
        self.assertEqual(len(summary["comparisons"]), 1)
        self.assertEqual(len(summary["comparisons"][0]["by_depth"]), 4)


if __name__ == "__main__":
    unittest.main()
