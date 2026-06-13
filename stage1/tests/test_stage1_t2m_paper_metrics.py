from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class Stage1T2MPaperMetricsTests(unittest.TestCase):
    def test_prompt_tsv_supports_optional_humanml_tokens(self):
        from Script.stage1.evaluate_t2m_paper_metrics import read_prompt_specs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.tsv"
            path.write_text(
                "walk_turn\ta person walks then turns\twalk/VERB turn/VERB\n"
                "jump\ta person jumps high\n",
                encoding="utf-8",
            )
            prompts = read_prompt_specs(path)

        self.assertEqual(prompts["walk_turn"].tokens, ("walk/VERB", "turn/VERB"))
        self.assertIn("person/OTHER", prompts["jump"].tokens)
        self.assertIn("jumps/OTHER", prompts["jump"].tokens)

    def test_generated_bvh_name_parser_splits_prompt_and_model(self):
        from Script.stage1.evaluate_t2m_paper_metrics import parse_generated_bvh_name

        self.assertEqual(
            parse_generated_bvh_name(Path("walk_turn_wave__finetuned_top_p.bvh")),
            ("walk_turn_wave", "finetuned_top_p"),
        )
        self.assertEqual(parse_generated_bvh_name(Path("loose_name.bvh")), ("loose_name", "generated"))

    def test_prepare_motion_array_aligns_truncates_normalizes_and_pads(self):
        from Script.stage1.evaluate_t2m_paper_metrics import prepare_motion_array

        with tempfile.TemporaryDirectory() as tmpdir:
            feature_path = Path(tmpdir) / "motion.npy"
            features = np.ones((11, 263), dtype=np.float32) * 3.0
            np.save(feature_path, features)

            motion, length = prepare_motion_array(
                feature_path,
                mean=np.ones(263, dtype=np.float32),
                std=np.ones(263, dtype=np.float32) * 2.0,
                max_motion_length=8,
                unit_length=4,
            )

        self.assertEqual(length, 8)
        self.assertEqual(motion.shape, (8, 263))
        np.testing.assert_allclose(motion, np.ones((8, 263), dtype=np.float32))

    def test_r_precision_matches_diagonal_pairs(self):
        from Script.stage1.evaluate_t2m_paper_metrics import calculate_r_precision

        text = np.eye(3, dtype=np.float32)
        motion = np.eye(3, dtype=np.float32)

        r_precision, matching_score = calculate_r_precision(text, motion, top_k=3)

        np.testing.assert_allclose(r_precision, np.ones(3))
        self.assertAlmostEqual(matching_score, 0.0)

    def test_check_only_reports_missing_assets_without_importing_evaluator(self):
        from Script.stage1.evaluate_t2m_paper_metrics import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompts = root / "prompts.tsv"
            prompts.write_text("walk\ta person walks\n", encoding="utf-8")
            output_dir = root / "out"
            summary = output_dir / "summary.json"

            main(
                [
                    str(root / "walk__baseline_top_p.bvh"),
                    "--prompts",
                    str(prompts),
                    "--humanml-root",
                    "/home/chenjie/cc/robotics/HumanML3D",
                    "--evaluator-root",
                    str(root / "missing_assets"),
                    "--output-dir",
                    str(output_dir),
                    "--summary",
                    str(summary),
                    "--check-only",
                ]
            )
            payload = json.loads(summary.read_text(encoding="utf-8"))

        self.assertFalse(payload["ready"])
        self.assertIn("walk", payload["planned_prompts"])
        self.assertIn("baseline_top_p", payload["planned_models"])
        self.assertIn("missing_assets", payload["readiness"])


if __name__ == "__main__":
    unittest.main()
