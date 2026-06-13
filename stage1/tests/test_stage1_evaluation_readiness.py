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
            (repo / "Script/stage1/bvh_to_humanml3d_features.py").write_text("", encoding="utf-8")
            humanml.mkdir()

            payload = check_evaluation_readiness(repo_root=repo, humanml_root=humanml)

        self.assertFalse(payload["paper_metrics_ready"])
        self.assertIn("HumanML3D text-motion evaluator source files", payload["paper_metrics_missing"])
        self.assertIn("pretrained HumanML3D evaluator", payload["paper_metrics_missing"][1])
        self.assertTrue(payload["engineering_tools"]["bvh_metrics"]["exists"])
        self.assertIn("checkpoints/t2m/text_mot_match/model/finest.tar", payload["t2m_evaluator"]["missing_assets"])
        self.assertTrue(payload["bvh_to_humanml3d_adapter"]["exists"])

    def test_readiness_detects_evaluator_source_and_checkpoint(self):
        from Script.stage1.check_evaluation_readiness import check_evaluation_readiness

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            humanml = root / "HumanML3D"
            (repo / "Script/stage1").mkdir(parents=True)
            (repo / "Script/stage1/bvh_to_humanml3d_features.py").write_text("", encoding="utf-8")
            humanml.mkdir()
            (humanml / "eval_t2m.py").write_text("", encoding="utf-8")
            (humanml / "evaluator.pth").write_bytes(b"")

            payload = check_evaluation_readiness(repo_root=repo, humanml_root=humanml)

        self.assertTrue(payload["paper_metrics_ready"])
        self.assertEqual(payload["paper_metrics_missing"], [])

    def test_readiness_detects_t2m_evaluator_layout(self):
        from Script.stage1.check_evaluation_readiness import check_evaluation_readiness

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            humanml = root / "HumanML3D"
            evaluator = root / "T2M-GPT"
            (repo / "Script/stage1").mkdir(parents=True)
            (repo / "Script/stage1/bvh_to_humanml3d_features.py").write_text("", encoding="utf-8")
            humanml.mkdir()
            for relative in (
                "models/evaluator_wrapper.py",
                "utils/eval_trans.py",
                "options/get_eval_option.py",
                "checkpoints/t2m/text_mot_match/model/finest.tar",
                "checkpoints/t2m/text_mot_match/opt.txt",
                "glove/our_vab_data.npy",
                "glove/our_vab_words.pkl",
            ):
                path = evaluator / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            payload = check_evaluation_readiness(repo_root=repo, humanml_root=humanml, evaluator_root=evaluator)

        self.assertTrue(payload["t2m_evaluator"]["ready"])
        self.assertTrue(payload["paper_metrics_ready"])
        self.assertEqual(payload["paper_metrics_missing"], [])
        self.assertIn("available_approximate_adapter", payload["bvh_to_humanml3d_adapter"]["status"])

    def test_readiness_requires_bvh_to_humanml3d_adapter(self):
        from Script.stage1.check_evaluation_readiness import check_evaluation_readiness

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            humanml = root / "HumanML3D"
            evaluator = root / "T2M-GPT"
            humanml.mkdir()
            for relative in (
                "models/evaluator_wrapper.py",
                "utils/eval_trans.py",
                "options/get_eval_option.py",
                "checkpoints/t2m/text_mot_match/model/finest.tar",
                "checkpoints/t2m/text_mot_match/opt.txt",
                "glove/our_vab_data.npy",
                "glove/our_vab_words.pkl",
            ):
                path = evaluator / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            payload = check_evaluation_readiness(repo_root=repo, humanml_root=humanml, evaluator_root=evaluator)

        self.assertFalse(payload["paper_metrics_ready"])
        self.assertIn("MoConVQ BVH/character motion", payload["paper_metrics_missing"][0])
        self.assertFalse(payload["bvh_to_humanml3d_adapter"]["exists"])


if __name__ == "__main__":
    unittest.main()
