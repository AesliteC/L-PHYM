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

    def eval(self):
        return self

    def encode_seq_all(self, obs, target):
        length = int(target.shape[0])
        indices = torch.arange(length * 4, dtype=torch.long).reshape(4, 1, length) % 512
        return {"indexs": indices}


class Stage1BVHCharacterRetargetTests(unittest.TestCase):
    def test_direct_script_execution_prefers_own_repo_root(self):
        from pathlib import Path

        import Script.stage1.diagnose_bvh_character_retarget as diag

        expected_root = Path(diag.__file__).resolve().parents[2]
        old_path = list(diag.sys.path)
        try:
            diag.sys.path[:] = ["/tmp/other_checkout"] + [
                path for path in old_path if Path(path or ".").resolve() != expected_root
            ]
            diag._ensure_own_repo_root_on_path(package="")
            self.assertEqual(Path(diag.sys.path[0]).resolve(), expected_root)
        finally:
            diag.sys.path[:] = old_path

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

    def test_main_forwards_motion_dataset_to_agent_loader(self):
        from Script.stage1 import diagnose_bvh_character_retarget as diag

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "summary.json"
            motion_dataset = tmp / "simple_motion_data.h5"
            motion_dataset.write_bytes(b"")

            fake_payload = {
                "summaries": [
                    {
                        "kind": "bvh_character",
                        "paths": ["toy.bvh"],
                        "fps": 20,
                        "flip": False,
                        "state_shape": [1, 20, 13],
                        "observation_shape": [1, 323],
                        "shape": [1, 4],
                        "stats": [],
                    }
                ],
                "comparisons": [],
            }

            with mock.patch.object(diag, "build_loaded_moconvq_agent", return_value=_FakeAgent()) as loader:
                with mock.patch.object(diag, "diagnose_bvh_character_retarget", return_value=fake_payload):
                    diag.main(
                        [
                            "toy.bvh",
                            "--base-data",
                            "base.data",
                            "--motion-dataset",
                            str(motion_dataset),
                            "--output-json",
                            str(out),
                        ]
                    )

            loader.assert_called_once()
            self.assertEqual(loader.call_args.kwargs["motion_dataset"], motion_dataset)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
