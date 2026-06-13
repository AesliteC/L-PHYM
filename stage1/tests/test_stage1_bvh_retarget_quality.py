import json
import tempfile
import unittest
from pathlib import Path


def _per_file_row(label: str, *, frames: int, tokens: int, p99: float, max_z: float, top_frac: float, unique: int):
    return {
        "path": f"exports/{label}.bvh",
        "state_shape": [frames, 20, 13],
        "observation_shape": [frames, 323],
        "shape": [tokens, 4],
        "observation_z": {
            "aggregate_abs_z": {
                "p95": 2.0,
                "p99": p99,
                "max": max_z,
                "frac_gt_5": 0.01,
                "frac_gt_10": 0.0,
            }
        },
        "stats": [
            {
                "depth": 0,
                "tokens": tokens,
                "unique": unique,
                "entropy_bits": 4.0,
                "top_fracs": [top_frac],
            }
        ],
    }


class Stage1BVHRetargetQualityTests(unittest.TestCase):
    def test_quality_summary_accepts_and_rejects_with_reasons(self):
        from Script.stage1.summarize_bvh_retarget_quality import summarize_retarget_quality

        payload = {
            "per_file": [
                _per_file_row("good", frames=180, tokens=45, p99=5.0, max_z=12.0, top_frac=0.12, unique=24),
                _per_file_row("bad_z", frames=180, tokens=45, p99=12.0, max_z=80.0, top_frac=0.12, unique=24),
                _per_file_row("collapsed", frames=180, tokens=45, p99=3.0, max_z=5.0, top_frac=0.9, unique=3),
                _per_file_row("short", frames=80, tokens=19, p99=3.0, max_z=5.0, top_frac=0.12, unique=24),
            ]
        }
        metrics = {
            "rows": [
                {"label": "good", "duration_sec": 9.0, "early_stop": False, "pose_velocity_mean": 1.0},
            ]
        }
        export = {"exports": [{"sample_id": "good", "caption": "a person walks"}]}

        summary = summarize_retarget_quality(payload, metrics_payload=metrics, export_payload=export)

        self.assertEqual(summary["counts"], {"total": 4, "accepted": 1, "rejected": 3})
        self.assertEqual(summary["accepted_paths"], ["exports/good.bvh"])
        by_label = {row["label"]: row for row in summary["rows"]}
        self.assertTrue(by_label["good"]["accepted"])
        self.assertEqual(by_label["good"]["caption"], "a person walks")
        self.assertIn("p99_abs_z>8", by_label["bad_z"]["reject_reasons"])
        self.assertIn("max_abs_z>50", by_label["bad_z"]["reject_reasons"])
        self.assertIn("depth0_top_frac>0.25", by_label["collapsed"]["reject_reasons"])
        self.assertIn("depth0_unique<16", by_label["collapsed"]["reject_reasons"])
        self.assertIn("frames<120", by_label["short"]["reject_reasons"])
        self.assertIn("tokens<20", by_label["short"]["reject_reasons"])

    def test_cli_writes_json_and_csv(self):
        from Script.stage1 import summarize_bvh_retarget_quality as quality

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            retarget = tmp / "retarget.json"
            metrics = tmp / "metrics.json"
            export = tmp / "export.json"
            output_json = tmp / "quality.json"
            output_csv = tmp / "quality.csv"
            retarget.write_text(
                json.dumps(
                    {
                        "per_file": [
                            _per_file_row("good", frames=180, tokens=45, p99=5.0, max_z=12.0, top_frac=0.12, unique=24)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            metrics.write_text(json.dumps({"rows": [{"label": "good", "duration_sec": 9.0}]}), encoding="utf-8")
            export.write_text(json.dumps({"exports": [{"sample_id": "good", "caption": "walk"}]}), encoding="utf-8")

            quality.main(
                [
                    "--retarget-json",
                    str(retarget),
                    "--bvh-metrics-json",
                    str(metrics),
                    "--export-summary",
                    str(export),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertTrue(output_json.exists())
            self.assertTrue(output_csv.exists())
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["counts"]["accepted"], 1)
            self.assertIn("good", output_csv.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
