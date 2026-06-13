import json
import tempfile
import unittest
from pathlib import Path


class Stage1RunSummaryTests(unittest.TestCase):
    def test_build_stage1_run_report_collects_core_evidence(self):
        from Script.stage1.summarize_stage1_run import build_stage1_run_report, report_to_markdown

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            quality = tmp / "quality.json"
            train_cache = tmp / "train_cache.json"
            val_cache = tmp / "val_cache.json"
            train_tokens = tmp / "train_tokens.json"
            val_tokens = tmp / "val_tokens.json"
            train_log = tmp / "train_log.jsonl"
            metrics = tmp / "metrics.json"
            videos = tmp / "video_summary.json"
            readiness = tmp / "readiness.json"

            quality.write_text(
                json.dumps(
                    {
                        "counts": {"total": 4, "accepted": 3, "rejected": 1},
                        "rows": [{"reject_reasons": ["max_abs_z>50"]}],
                    }
                ),
                encoding="utf-8",
            )
            train_cache.write_text(
                json.dumps({"windows": 10, "valid_tokens": 100, "unique_sequences": 3}),
                encoding="utf-8",
            )
            val_cache.write_text(
                json.dumps({"windows": 2, "valid_tokens": 20, "unique_sequences": 1}),
                encoding="utf-8",
            )
            token_payload = {
                "summaries": [
                    {
                        "stats": [
                            {"depth": 0, "top_fracs": [0.1]},
                            {"depth": 1, "top_fracs": [0.2]},
                        ]
                    }
                ]
            }
            train_tokens.write_text(json.dumps(token_payload), encoding="utf-8")
            val_tokens.write_text(json.dumps(token_payload), encoding="utf-8")
            train_log.write_text(
                "\n".join(
                    [
                        json.dumps({"epoch": 0, "train": {"loss": 2.0}, "val": {"loss": 3.0, "token_accuracy": 0.1}}),
                        json.dumps({"epoch": 1, "train": {"loss": 1.0}, "val": {"loss": 2.0, "token_accuracy": 0.2}}),
                    ]
                ),
                encoding="utf-8",
            )
            metrics.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "label": "walk__baseline_top_p",
                                "frames": 100,
                                "early_stop": True,
                                "root_path_length": 1.0,
                                "pose_velocity_mean": 2.0,
                                "pose_variance_mean": 3.0,
                            },
                            {
                                "label": "walk__finetuned_top_p",
                                "frames": 150,
                                "early_stop": False,
                                "root_path_length": 1.5,
                                "pose_velocity_mean": 2.5,
                                "pose_variance_mean": 4.0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            videos.write_text(json.dumps({"side_by_side_videos": ["a.mp4"]}), encoding="utf-8")
            readiness.write_text(
                json.dumps({"paper_metrics_ready": False, "paper_metrics_missing": ["evaluator checkpoint"]}),
                encoding="utf-8",
            )

            report = build_stage1_run_report(
                run_name="toy",
                quality_summary=quality,
                train_cache_summary=train_cache,
                val_cache_summary=val_cache,
                train_token_distribution=train_tokens,
                val_token_distribution=val_tokens,
                train_log=train_log,
                metrics_json=metrics,
                comparison_video_summary=videos,
                evaluation_readiness=readiness,
                checkpoint="ckpt.pth",
                notes="note",
            )
            markdown = report_to_markdown(report)

        self.assertAlmostEqual(report["quality"]["accepted_rate"], 0.75)
        self.assertEqual(report["cache"]["train"]["windows"], 10)
        self.assertAlmostEqual(report["token_top_fractions"]["train"]["depth0"], 0.1)
        self.assertEqual(report["training"]["summary"]["epochs"], 2)
        self.assertAlmostEqual(report["generation"]["improvements"]["frames"]["delta"], 50.0)
        self.assertFalse(report["paper_metrics"]["ready"])
        self.assertIn("Paper-level FID/R-precision should not be claimed", markdown)
        self.assertIn("a.mp4", markdown)


if __name__ == "__main__":
    unittest.main()
