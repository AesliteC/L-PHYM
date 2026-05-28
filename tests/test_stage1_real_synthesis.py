import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


def _make_clip(length: int, root_x: float) -> np.ndarray:
    joints = np.zeros((length, 22, 3), dtype=np.float32)
    joints[:, 0, 0] = root_x + np.linspace(0.0, 0.1, length)
    joints[:, 0, 1] = 1.0
    joints[:, 1, :] = joints[:, 0, :] + np.array([0.1, -0.2, 0.0], dtype=np.float32)
    joints[:, 2, :] = joints[:, 0, :] + np.array([-0.1, -0.2, 0.0], dtype=np.float32)
    joints[:, 7, :] = joints[:, 0, :] + np.array([0.1, -0.9, 0.05], dtype=np.float32)
    joints[:, 8, :] = joints[:, 0, :] + np.array([-0.1, -0.9, 0.05], dtype=np.float32)
    joints[:, 10, :] = joints[:, 7, :] + np.array([0.0, -0.05, 0.12], dtype=np.float32)
    joints[:, 11, :] = joints[:, 8, :] + np.array([0.0, -0.05, 0.12], dtype=np.float32)
    joints[:, 13, :] = joints[:, 0, :] + np.array([0.22, 0.35, 0.0], dtype=np.float32)
    joints[:, 14, :] = joints[:, 0, :] + np.array([-0.22, 0.35, 0.0], dtype=np.float32)
    return joints


def _write_humanml_fixture(base: Path) -> Path:
    root = base / "HumanML3D"
    for dirname in ("new_joints", "new_joint_vecs", "texts"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    ids = ["000001", "000002", "000003"]
    for split in ("all", "train", "val", "test", "train_val"):
        (root / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    for idx, sample_id in enumerate(ids):
        np.save(root / "new_joints" / f"{sample_id}.npy", _make_clip(8 + idx, float(idx)))
        np.save(root / "new_joint_vecs" / f"{sample_id}.npy", np.full((8 + idx, 263), idx, dtype=np.float32))
        (root / "texts" / f"{sample_id}.txt").write_text(
            f"caption {sample_id}#caption/NOUN {sample_id}/NUM#0.0#0.0\n",
            encoding="utf-8",
        )
    with (base / "index.csv").open("w", encoding="utf-8") as f:
        f.write("source_path,start_frame,end_frame,new_name\n")
        for sample_id in ids:
            f.write(f"./pose_data/{sample_id}.npy,0,8,{sample_id}.npy\n")
    return root


class Stage1RealSynthesisTests(unittest.TestCase):
    def test_synthesize_dataset_writes_manifest_h5_and_summary(self):
        from Script.stage1.synthesize_long_humanml3d import synthesize_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = _write_humanml_fixture(tmp)
            out = tmp / "out"

            summary = synthesize_dataset(
                humanml_root=root,
                split="train",
                num_sequences=3,
                min_clips=2,
                max_clips=2,
                seed=123,
                candidate_pool=2,
                transition_max_score=0.35,
                blend_frames=2,
                caption_joiner=" then ",
                output_dir=out,
            )

            self.assertEqual(summary["num_sequences"], 3)
            rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(rows[0]["sample_ids"]), 2)
            self.assertIn(" then ", rows[0]["caption"])
            self.assertIn("transition_forced", rows[0])

            with h5py.File(out / "long_sequences.h5", "r") as h5:
                self.assertEqual(len(h5.keys()), 3)
                group = h5[rows[0]["sequence_id"]]
                self.assertEqual(group["joints_22"].shape[1:], (22, 3))
                self.assertEqual(group["joint_vecs_263"].shape[1], 263)
                self.assertEqual(group["clip_boundaries"].shape, (2, 2))
                self.assertEqual(group.attrs["split"], "train")

            loaded_summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(loaded_summary["config"]["candidate_pool"], 2)

    def test_synthesize_dataset_filters_clips_with_invalid_joint_shape(self):
        from Script.stage1.synthesize_long_humanml3d import synthesize_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = _write_humanml_fixture(tmp)
            bad_id = "bad001"
            for split in ("all", "train", "val", "test", "train_val"):
                with (root / f"{split}.txt").open("a", encoding="utf-8") as f:
                    f.write(f"{bad_id}\n")
            np.save(root / "new_joints" / f"{bad_id}.npy", _make_clip(1, 9.0)[0])
            np.save(root / "new_joint_vecs" / f"{bad_id}.npy", np.zeros((1, 263), dtype=np.float32))
            (root / "texts" / f"{bad_id}.txt").write_text(
                "bad clip#bad clip#0.0#0.0\n",
                encoding="utf-8",
            )

            summary = synthesize_dataset(
                humanml_root=root,
                split="train",
                num_sequences=4,
                min_clips=2,
                max_clips=2,
                seed=7,
                candidate_pool=4,
                transition_max_score=0.35,
                blend_frames=2,
                caption_joiner=" then ",
                output_dir=tmp / "out",
            )

            self.assertEqual(summary["filtered_invalid_clips"], 1)
            rows = [json.loads(line) for line in (tmp / "out" / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertNotIn(bad_id, row["sample_ids"])


if __name__ == "__main__":
    unittest.main()
