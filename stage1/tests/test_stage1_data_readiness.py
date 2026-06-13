from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


def _write_processed_humanml(root: Path, sample_ids: list[str]) -> Path:
    humanml = root / "HumanML3D"
    for dirname in ("texts", "new_joints", "new_joint_vecs"):
        (humanml / dirname).mkdir(parents=True, exist_ok=True)
    (humanml / "all.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    for split in ("train", "val", "test", "train_val"):
        (humanml / f"{split}.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    np.save(humanml / "Mean.npy", np.zeros(263, dtype=np.float32))
    np.save(humanml / "Std.npy", np.ones(263, dtype=np.float32))
    for sample_id in sample_ids:
        (humanml / "texts" / f"{sample_id}.txt").write_text("walk forward#walk forward#0#0\n", encoding="utf-8")
        np.save(humanml / "new_joints" / f"{sample_id}.npy", np.zeros((4, 22, 3), dtype=np.float32))
        np.save(humanml / "new_joint_vecs" / f"{sample_id}.npy", np.zeros((4, 263), dtype=np.float32))
    return humanml


class Stage1DataReadinessTests(unittest.TestCase):
    def test_processed_humanml_without_source_motion_is_not_mainline_ready(self):
        from Script.stage1.check_stage1_data_readiness import check_stage1_data_readiness

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            (repo / "Script/stage1").mkdir(parents=True)
            (repo / "Script/stage1/build_bvh_character_gpt_cache.py").write_text("", encoding="utf-8")
            (repo / "Script/stage1/diagnose_bvh_character_retarget.py").write_text("", encoding="utf-8")
            humanml = _write_processed_humanml(root / "HumanML3DRoot", ["000001", "000002"])
            (humanml.parent / "index.csv").write_text(
                "source_path,start_frame,end_frame,new_name\n"
                "./pose_data/KIT/example_poses.npy,0,10,000001.npy\n",
                encoding="utf-8",
            )

            payload = check_stage1_data_readiness(repo_root=repo, humanml_root=humanml.parent)

        self.assertTrue(payload["processed_corpus"]["ready"])
        self.assertFalse(payload["source_motion_available_for_export"])
        self.assertFalse(payload["native_bvh_cache_ready"])
        self.assertFalse(payload["stage1_mainline_ready"])
        self.assertIn("HumanML3D/AMASS source motion files or BVH exports", payload["missing"])

    def test_bvh_source_and_tools_make_native_cache_route_ready(self):
        from Script.stage1.check_stage1_data_readiness import check_stage1_data_readiness

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            (repo / "Script/stage1").mkdir(parents=True)
            (repo / "Script/stage1/build_bvh_character_gpt_cache.py").write_text("", encoding="utf-8")
            (repo / "Script/stage1/diagnose_bvh_character_retarget.py").write_text("", encoding="utf-8")
            humanml = _write_processed_humanml(root / "HumanML3DRoot", ["000001"])
            bvh_root = root / "bvh"
            bvh_root.mkdir()
            (bvh_root / "sample.bvh").write_text("HIERARCHY\nMOTION\nFrames: 0\nFrame Time: 0.033333\n", encoding="utf-8")

            payload = check_stage1_data_readiness(
                repo_root=repo,
                humanml_root=humanml,
                source_roots=[bvh_root],
            )

        self.assertTrue(payload["processed_corpus"]["ready"])
        self.assertTrue(payload["source_motion_available_for_export"])
        self.assertTrue(payload["native_bvh_cache_ready"])
        self.assertTrue(payload["stage1_mainline_ready"])


if __name__ == "__main__":
    unittest.main()
