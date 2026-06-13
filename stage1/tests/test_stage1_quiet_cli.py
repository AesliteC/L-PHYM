import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


class _FakeAgent:
    def eval(self):
        return self


def _capture_json(fn, argv):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        fn(argv)
    return json.loads(stream.getvalue())


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


def _quality_per_file_row(label: str):
    return {
        "path": f"exports/{label}.bvh",
        "state_shape": [180, 20, 13],
        "observation_shape": [180, 323],
        "shape": [45, 4],
        "observation_z": {
            "aggregate_abs_z": {
                "p95": 2.0,
                "p99": 5.0,
                "max": 12.0,
                "frac_gt_5": 0.01,
                "frac_gt_10": 0.0,
            }
        },
        "stats": [
            {
                "depth": 0,
                "tokens": 45,
                "unique": 24,
                "entropy_bits": 4.0,
                "top_fracs": [0.12],
            }
        ],
    }


class Stage1QuietCliTests(unittest.TestCase):
    def test_evaluate_bvh_metrics_quiet_prints_compact_json(self):
        from Script.stage1 import evaluate_bvh_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bvh = tmp / "toy.bvh"
            output = tmp / "metrics.json"
            _write_toy_bvh(bvh, frames=8)

            payload = _capture_json(
                evaluate_bvh_metrics.main,
                [str(bvh), "--expected-min-frames", "10", "--output", str(output), "--quiet"],
            )

        self.assertEqual(payload["rows"], 1)
        self.assertEqual(payload["early_stop"], 1)
        self.assertTrue(str(payload["output"]).endswith("metrics.json"))

    def test_summarize_retarget_quality_quiet_prints_counts(self):
        from Script.stage1 import summarize_bvh_retarget_quality

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            retarget = tmp / "retarget.json"
            metrics = tmp / "metrics.json"
            export = tmp / "export.json"
            output_json = tmp / "quality.json"
            output_csv = tmp / "quality.csv"
            retarget.write_text(json.dumps({"per_file": [_quality_per_file_row("good")]}), encoding="utf-8")
            metrics.write_text(json.dumps({"rows": [{"label": "good", "duration_sec": 9.0}]}), encoding="utf-8")
            export.write_text(json.dumps({"exports": [{"sample_id": "good", "caption": "walk"}]}), encoding="utf-8")

            payload = _capture_json(
                summarize_bvh_retarget_quality.main,
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
                    "--quiet",
                ],
            )

        self.assertEqual(payload["counts"], {"total": 1, "accepted": 1, "rejected": 0})
        self.assertTrue(str(payload["output_json"]).endswith("quality.json"))
        self.assertTrue(str(payload["output_csv"]).endswith("quality.csv"))

    def test_diagnose_token_distribution_quiet_prints_depth_summaries(self):
        from Script.stage1 import diagnose_token_distribution

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cache = tmp / "cache.pt"
            output = tmp / "tokens.json"
            torch.save(
                {
                    "indices": torch.tensor([[[1, 2, 3, 4], [1, 5, 6, 7]]], dtype=torch.long),
                    "target_masks": torch.tensor([[True, True]], dtype=torch.bool),
                },
                cache,
            )

            payload = _capture_json(
                diagnose_token_distribution.main,
                ["--cache", str(cache), "--output-json", str(output), "--quiet"],
            )

        self.assertEqual(payload["comparisons"], 0)
        self.assertEqual(payload["summaries"][0]["shape"], [1, 2, 4])
        self.assertEqual(payload["summaries"][0]["depths"][0]["tokens"], 2)
        self.assertEqual(payload["summaries"][0]["depths"][0]["unique"], 1)

    def test_bvh_character_retarget_quiet_prints_core_shapes(self):
        from Script.stage1 import diagnose_bvh_character_retarget

        fake_payload = {
            "summaries": [
                {
                    "kind": "bvh_character",
                    "paths": ["toy.bvh"],
                    "state_shape": [9, 20, 13],
                    "observation_shape": [9, 323],
                    "shape": [2, 4],
                    "observation_z": {"aggregate_abs_z": {"p99": 3.5, "max": 8.0}},
                    "stats": [],
                }
            ],
            "comparisons": [{}],
            "per_file": [{"path": "toy.bvh"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output = tmp / "retarget.json"
            with mock.patch.object(diagnose_bvh_character_retarget, "build_loaded_moconvq_agent", return_value=_FakeAgent()):
                with mock.patch.object(
                    diagnose_bvh_character_retarget,
                    "diagnose_bvh_character_retarget",
                    return_value=fake_payload,
                ):
                    payload = _capture_json(
                        diagnose_bvh_character_retarget.main,
                        ["toy.bvh", "--base-data", "base.data", "--output-json", str(output), "--quiet"],
                    )

        self.assertEqual(payload["paths"], 1)
        self.assertEqual(payload["state_shape"], [9, 20, 13])
        self.assertEqual(payload["observation_shape"], [9, 323])
        self.assertEqual(payload["token_shape"], [2, 4])
        self.assertEqual(payload["per_file"], 1)
        self.assertEqual(payload["comparisons"], 1)

    def test_bvh_character_cache_quiet_prints_cache_counts(self):
        from Script.stage1 import build_bvh_character_gpt_cache

        fake_cache = {
            "indices": torch.zeros((2, 3, 4), dtype=torch.long),
            "latents": torch.zeros((2, 3, 768), dtype=torch.float32),
            "text_features": torch.zeros((2, 4, 1024), dtype=torch.float32),
            "text_masks": torch.ones((2, 4), dtype=torch.bool),
            "captions": ["walk", "walk"],
            "sequence_ids": ["a", "b"],
            "target_masks": torch.ones((2, 3), dtype=torch.bool),
            "config": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bvh = tmp / "walk.bvh"
            bvh.write_text("HIERARCHY\nMOTION\nFrames: 0\nFrame Time: 0.05\n", encoding="utf-8")
            output = tmp / "cache.pt"
            observation = tmp / "obs.h5"
            summary = tmp / "summary.json"
            with mock.patch.object(build_bvh_character_gpt_cache, "build_loaded_moconvq_agent", return_value=object()):
                with mock.patch.object(build_bvh_character_gpt_cache, "build_t5_text_encoder", return_value=object()):
                    with mock.patch.object(
                        build_bvh_character_gpt_cache,
                        "build_bvh_character_cache",
                        return_value=fake_cache,
                    ):
                        payload = _capture_json(
                            build_bvh_character_gpt_cache.main,
                            [
                                "--bvh",
                                f"{bvh}=a person walks",
                                "--base-data",
                                "base.data",
                                "--output",
                                str(output),
                                "--observation-h5",
                                str(observation),
                                "--summary",
                                str(summary),
                                "--quiet",
                            ],
                        )

        self.assertEqual(payload["windows"], 2)
        self.assertEqual(payload["valid_tokens"], 24)
        self.assertEqual(payload["unique_sequences"], 2)
        self.assertTrue(str(payload["output"]).endswith("cache.pt"))
        self.assertTrue(str(payload["summary"]).endswith("summary.json"))

    def test_export_humanml3d_to_bvh_quiet_prints_export_count(self):
        from tests.test_stage1_humanml3d_bvh_export import _write_minimal_humanml
        from Script.stage1 import export_humanml3d_to_bvh

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            humanml = _write_minimal_humanml(tmp, "000001", frames=5)
            output_dir = tmp / "exported"
            summary = tmp / "summary.json"

            payload = _capture_json(
                export_humanml3d_to_bvh.main,
                [
                    "--humanml-root",
                    str(humanml),
                    "--sample-id",
                    "000001",
                    "--output-dir",
                    str(output_dir),
                    "--summary",
                    str(summary),
                    "--quiet",
                ],
            )

        self.assertEqual(payload["exports"], 1)
        self.assertEqual(payload["rotation_source"], "joints_ik")
        self.assertTrue(str(payload["summary"]).endswith("summary.json"))


if __name__ == "__main__":
    unittest.main()
