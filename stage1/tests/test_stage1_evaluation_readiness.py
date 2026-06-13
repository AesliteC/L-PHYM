import tempfile
import unittest
from pathlib import Path


class Stage1EvaluationReadinessTests(unittest.TestCase):
    def test_readiness_reports_missing_paper_evaluator_assets(self):
        from Script.stage1.check_evaluation_readiness import check_evaluation_readiness

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            humanml = root / "HumanML3D"
            (repo / "Script/stage1").mkdir(parents=True)
            (repo / "Script/stage1/evaluate_bvh_metrics.py").write_text("", encoding="utf-8")
            humanml.mkdir()

            payload = check_evaluation_readiness(repo_root=repo, humanml_root=humanml)

        self.assertFalse(payload["paper_metrics_ready"])
        self.assertIn("HumanML3D text-motion evaluator source files", payload["paper_metrics_missing"])
        self.assertIn("pretrained HumanML3D evaluator", payload["paper_metrics_missing"][1])
        self.assertTrue(payload["engineering_tools"]["bvh_metrics"]["exists"])

    def test_readiness_detects_evaluator_source_and_checkpoint(self):
        from Script.stage1.check_evaluation_readiness import check_evaluation_readiness

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            humanml = root / "HumanML3D"
            humanml.mkdir()
            (humanml / "eval_t2m.py").write_text("", encoding="utf-8")
            (humanml / "evaluator.pth").write_bytes(b"")

            payload = check_evaluation_readiness(repo_root=repo, humanml_root=humanml)

        self.assertTrue(payload["paper_metrics_ready"])
        self.assertEqual(payload["paper_metrics_missing"], [])


if __name__ == "__main__":
    unittest.main()
