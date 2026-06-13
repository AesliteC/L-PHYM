import tempfile
import unittest
from pathlib import Path


def _write_toy_bvh(path: Path, frames: int = 8) -> None:
    rows = []
    for idx in range(frames):
        rows.append(f"{0.1 * idx:.6f} 0.000000 0.000000 {idx:.6f} 0.000000 0.000000")
    path.write_text(
        "\n".join(
            [
                "HIERARCHY",
                "ROOT RootJoint",
                "{",
                "  OFFSET 0.0 0.0 0.0",
                "  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation",
                "  End Site",
                "  {",
                "    OFFSET 0.0 1.0 0.0",
                "  }",
                "}",
                "MOTION",
                f"Frames: {frames}",
                "Frame Time: 0.008333",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class Stage1BVHMetricTests(unittest.TestCase):
    def test_evaluate_bvh_files_reports_stage1_engineering_metrics(self):
        from Script.stage1.evaluate_bvh_metrics import evaluate_bvh_files

        with tempfile.TemporaryDirectory() as tmpdir:
            bvh = Path(tmpdir) / "toy.bvh"
            _write_toy_bvh(bvh, frames=8)
            summary = evaluate_bvh_files(
                [str(bvh)],
                sample_stride=1,
                lags=(1, 2),
                expected_min_frames=10,
            )

        self.assertIn("metric_notes", summary)
        self.assertEqual(len(summary["rows"]), 1)
        row = summary["rows"][0]
        self.assertEqual(row["frames"], 8)
        self.assertTrue(row["early_stop"])
        self.assertGreater(row["duration_sec"], 0.0)
        self.assertGreater(row["root_path_length"], 0.0)
        self.assertIn("lag_1_mean_cosine", row)

    def test_load_bvh_motion_trims_extra_rows_to_header_frame_count(self):
        from Script.stage1.evaluate_bvh_metrics import load_bvh_motion

        with tempfile.TemporaryDirectory() as tmpdir:
            bvh = Path(tmpdir) / "extra_rows.bvh"
            _write_toy_bvh(bvh, frames=8)
            with bvh.open("a", encoding="utf-8") as handle:
                handle.write("999 0 0 0 0 0\n")

            motion, _frame_time = load_bvh_motion(bvh)

        self.assertEqual(motion.shape[0], 8)
        self.assertNotEqual(motion[-1, 0], 999)


if __name__ == "__main__":
    unittest.main()
