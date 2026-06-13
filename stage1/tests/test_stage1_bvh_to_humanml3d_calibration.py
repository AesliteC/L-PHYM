from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class Stage1BVHToHumanML3DCalibrationTests(unittest.TestCase):
    def test_compare_feature_arrays_reports_zero_for_identical_inputs(self):
        from Script.stage1.calibrate_bvh_to_humanml3d_adapter import compare_feature_arrays

        original = np.ones((4, 263), dtype=np.float32)
        mean = np.zeros(263, dtype=np.float32)
        std = np.ones(263, dtype=np.float32)

        stats = compare_feature_arrays(original, original.copy(), mean=mean, std=std)

        self.assertEqual(stats["original_feature_frames"], 4)
        self.assertEqual(stats["roundtrip_feature_frames"], 4)
        self.assertEqual(stats["compared_feature_frames"], 4)
        self.assertEqual(stats["feature_mae"], 0.0)
        self.assertEqual(stats["feature_rmse"], 0.0)
        self.assertEqual(stats["feature_z_mae"], 0.0)
        self.assertEqual(stats["feature_z_rmse"], 0.0)

    def test_compare_feature_arrays_uses_common_prefix_length(self):
        from Script.stage1.calibrate_bvh_to_humanml3d_adapter import compare_feature_arrays

        original = np.zeros((3, 263), dtype=np.float32)
        roundtrip = np.ones((2, 263), dtype=np.float32)
        mean = np.zeros(263, dtype=np.float32)
        std = np.ones(263, dtype=np.float32)

        stats = compare_feature_arrays(original, roundtrip, mean=mean, std=std)

        self.assertEqual(stats["original_feature_frames"], 3)
        self.assertEqual(stats["roundtrip_feature_frames"], 2)
        self.assertEqual(stats["compared_feature_frames"], 2)
        self.assertAlmostEqual(stats["feature_mae"], 1.0)
        self.assertAlmostEqual(stats["feature_rmse"], 1.0)

    def test_compare_joint_arrays_reports_zero_for_identical_inputs(self):
        from Script.stage1.calibrate_bvh_to_humanml3d_adapter import compare_joint_arrays

        joints = np.zeros((5, 22, 3), dtype=np.float32)
        joints[:, :, 0] = np.arange(22, dtype=np.float32)

        stats = compare_joint_arrays(joints, joints.copy())

        self.assertEqual(stats["original_joint_frames"], 5)
        self.assertEqual(stats["roundtrip_joint_frames"], 5)
        self.assertEqual(stats["compared_joint_frames"], 5)
        self.assertEqual(stats["joint_mpjpe_mean"], 0.0)
        self.assertEqual(stats["local_joint_mpjpe_mean"], 0.0)
        self.assertEqual(stats["root_position_error_mean"], 0.0)

    def test_summarize_rows_averages_numeric_metrics(self):
        from Script.stage1.calibrate_bvh_to_humanml3d_adapter import summarize_rows

        rows = [
            {
                "feature_mae": 1.0,
                "feature_rmse": 2.0,
                "feature_p95_abs": 3.0,
                "feature_z_mae": 4.0,
                "feature_z_rmse": 5.0,
                "feature_z_p95_abs": 6.0,
                "joint_mpjpe_mean": 7.0,
                "joint_mpjpe_p95": 8.0,
                "local_joint_mpjpe_mean": 9.0,
                "local_joint_mpjpe_p95": 10.0,
                "root_position_error_mean": 11.0,
                "root_position_error_p95": 12.0,
            },
            {
                "feature_mae": 3.0,
                "feature_rmse": 4.0,
                "feature_p95_abs": 5.0,
                "feature_z_mae": 6.0,
                "feature_z_rmse": 7.0,
                "feature_z_p95_abs": 8.0,
                "joint_mpjpe_mean": 9.0,
                "joint_mpjpe_p95": 10.0,
                "local_joint_mpjpe_mean": 11.0,
                "local_joint_mpjpe_p95": 12.0,
                "root_position_error_mean": 13.0,
                "root_position_error_p95": 14.0,
            },
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary["samples"], 2.0)
        self.assertEqual(summary["avg_feature_mae"], 2.0)
        self.assertEqual(summary["max_feature_mae"], 3.0)
        self.assertEqual(summary["avg_root_position_error_p95"], 13.0)
        self.assertEqual(summary["max_root_position_error_p95"], 14.0)

    def test_calibrate_real_sample_smoke_writes_summary_metrics(self):
        from Script.stage1.calibrate_bvh_to_humanml3d_adapter import calibrate_sample

        humanml_root = Path("/home/chenjie/cc/robotics/HumanML3D/HumanML3D")
        sample_id = "000021"
        if not (humanml_root / "new_joint_vecs" / f"{sample_id}.npy").exists():
            self.skipTest("local HumanML3D reference sample is unavailable")

        mean = np.load(humanml_root / "Mean.npy").astype(np.float32)
        std = np.maximum(np.load(humanml_root / "Std.npy").astype(np.float32), 1e-8)
        with tempfile.TemporaryDirectory() as tmpdir:
            row = calibrate_sample(
                sample_id=sample_id,
                humanml_root=humanml_root,
                output_dir=Path(tmpdir),
                mean=mean,
                std=std,
            )

        self.assertEqual(row["sample_id"], sample_id)
        self.assertGreater(row["compared_feature_frames"], 0)
        self.assertGreater(row["compared_joint_frames"], 0)
        self.assertTrue(np.isfinite(float(row["feature_mae"])))
        self.assertTrue(np.isfinite(float(row["joint_mpjpe_mean"])))


if __name__ == "__main__":
    unittest.main()
