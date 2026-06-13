from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class Stage1BVHToHumanML3DFeaturesTests(unittest.TestCase):
    def test_resample_positions_downsamples_to_target_fps(self):
        from Script.stage1.bvh_to_humanml3d_features import resample_positions

        positions = np.zeros((121, 2, 3), dtype=np.float32)
        positions[:, 0, 0] = np.arange(121, dtype=np.float32)
        resampled = resample_positions(positions, source_fps=120.0, target_fps=20.0)

        self.assertEqual(resampled.shape, (21, 2, 3))
        np.testing.assert_allclose(resampled[:, 0, 0], np.arange(0, 121, 6), atol=1e-5)

    def test_bvh_fk_positions_map_to_humanml3d_joint_shape(self):
        from Script.stage1.bvh_to_humanml3d_features import bvh_positions_to_humanml3d_joints
        from Script.stage1.render_bvh_to_mp4 import frame_positions, parse_bvh

        template = Path(__file__).resolve().parents[1] / "base.bvh"
        nodes, motion, _frame_time = parse_bvh(template)
        positions = np.stack([frame_positions(nodes, motion[0]) for _ in range(3)], axis=0)
        joints = bvh_positions_to_humanml3d_joints(nodes, positions)

        self.assertEqual(joints.shape, (3, 22, 3))
        node_by_name = {node.name: idx for idx, node in enumerate(nodes)}
        np.testing.assert_allclose(joints[:, 0], positions[:, node_by_name["RootJoint"]])
        np.testing.assert_allclose(joints[:, 1], positions[:, node_by_name["lHip"]])
        np.testing.assert_allclose(joints[:, 2], positions[:, node_by_name["rHip"]])
        self.assertTrue(np.isfinite(joints).all())

    def test_real_bvh_smoke_writes_263d_features(self):
        from Script.stage1.bvh_to_humanml3d_features import convert_bvh_to_humanml3d_features

        bvh = Path(__file__).resolve().parents[1] / "base.bvh"
        humanml_root = Path("/home/chenjie/cc/robotics/HumanML3D/HumanML3D")
        if not (humanml_root / "new_joints" / "000021.npy").exists():
            self.skipTest("local HumanML3D reference sample is unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "base.npy"
            joints = Path(tmpdir) / "base_joints.npy"
            summary = convert_bvh_to_humanml3d_features(
                bvh,
                humanml_data_root=humanml_root,
                output_vecs=out,
                output_joints=joints,
                target_fps=20.0,
            )
            features = np.load(out)
            converted_joints = np.load(joints)

        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], 263)
        self.assertEqual(summary["feature_dim"], 263)
        self.assertEqual(converted_joints.shape[1:], (22, 3))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
