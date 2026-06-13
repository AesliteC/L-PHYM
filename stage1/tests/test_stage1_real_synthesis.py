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


def _write_disconnected_humanml_fixture(base: Path) -> Path:
    root = base / "HumanML3D"
    for dirname in ("new_joints", "new_joint_vecs", "texts"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    ids = ["000001", "000002"]
    for split in ("all", "train", "val", "test", "train_val"):
        (root / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    for idx, sample_id in enumerate(ids):
        clip = _make_clip(8, float(idx) * 100.0)
        if idx == 1:
            fast_drift = np.linspace(0.0, 5.0, 8, dtype=np.float32)
            clip[:, :, 0] += fast_drift[:, None]
        np.save(root / "new_joints" / f"{sample_id}.npy", clip)
        np.save(root / "new_joint_vecs" / f"{sample_id}.npy", np.full((8, 263), idx, dtype=np.float32))
        (root / "texts" / f"{sample_id}.txt").write_text(
            f"caption {sample_id}#caption/NOUN {sample_id}/NUM#0.0#0.0\n",
            encoding="utf-8",
        )
    (base / "index.csv").write_text("source_path,start_frame,end_frame,new_name\n", encoding="utf-8")
    return root


def _write_mixed_transition_humanml_fixture(base: Path) -> Path:
    root = base / "HumanML3D"
    for dirname in ("new_joints", "new_joint_vecs", "texts"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    ids = ["000001", "000002", "000003"]
    for split in ("all", "train", "val", "test", "train_val"):
        (root / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    for idx, sample_id in enumerate(ids):
        clip = _make_clip(8, float(idx))
        if idx == 2:
            fast_drift = np.linspace(0.0, 5.0, 8, dtype=np.float32)
            clip[:, :, 0] += fast_drift[:, None]
        np.save(root / "new_joints" / f"{sample_id}.npy", clip)
        np.save(root / "new_joint_vecs" / f"{sample_id}.npy", np.full((8, 263), idx, dtype=np.float32))
        (root / "texts" / f"{sample_id}.txt").write_text(
            f"caption {sample_id}#caption/NOUN {sample_id}/NUM#0.0#0.0\n",
            encoding="utf-8",
        )
    (base / "index.csv").write_text("source_path,start_frame,end_frame,new_name\n", encoding="utf-8")
    return root


class Stage1RealSynthesisTests(unittest.TestCase):
    def test_transition_score_is_computed_after_boundary_alignment(self):
        from Script.stage1.synthesize_long_humanml3d import transition_score

        prev = _make_clip(8, root_x=0.0)
        same_motion_far_away = _make_clip(8, root_x=100.0)

        score = transition_score(prev, same_motion_far_away)

        self.assertLess(score["root_position"], 1e-4)
        self.assertLess(score["yaw"], 1e-4)
        self.assertLess(score["total"], 0.05)

    def test_synthesize_dataset_writes_manifest_h5_and_summary(self):
        from Script.stage1.synthesize_long_humanml3d import synthesize_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = _write_mixed_transition_humanml_fixture(tmp)
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
                allow_forced_transitions=True,
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
                self.assertEqual(group["dropped_prefix_frames"].shape, (2,))
                self.assertEqual(group.attrs["split"], "train")

            loaded_summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(loaded_summary["config"]["candidate_pool"], 2)
            self.assertEqual(
                loaded_summary["transitions"],
                sum(len(row["transition_scores"]) for row in rows),
            )
            self.assertEqual(loaded_summary["duplicate_sequences"], 0)

            log_text = (out / "synthesize.log").read_text(encoding="utf-8")
            self.assertIn("[start]", log_text)
            self.assertIn("[sequence_written]", log_text)
            self.assertIn("[summary]", log_text)

            progress_events = [
                json.loads(line)["event"]
                for line in (out / "synthesize_progress.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIn("start", progress_events)
            self.assertIn("sequence_written", progress_events)
            self.assertIn("summary", progress_events)

    def test_diagnose_long_humanml3d_quality_reports_boundary_stats(self):
        from Script.stage1.diagnose_long_humanml3d_quality import diagnose_long_humanml3d_quality
        from Script.stage1.synthesize_long_humanml3d import synthesize_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = _write_mixed_transition_humanml_fixture(tmp)
            out = tmp / "out"
            synthesize_dataset(
                humanml_root=root,
                split="train",
                num_sequences=2,
                min_clips=2,
                max_clips=2,
                seed=123,
                candidate_pool=2,
                transition_max_score=0.35,
                blend_frames=2,
                caption_joiner=" then ",
                output_dir=out,
                allow_forced_transitions=True,
            )

            summary = diagnose_long_humanml3d_quality(
                long_h5_path=out / "long_sequences.h5",
                manifest_path=out / "manifest.jsonl",
                output_json=tmp / "quality.json",
                transition_jsonl=tmp / "transitions.jsonl",
                root_gap_warn=0.0,
                root_velocity_warn=0.0,
                yaw_warn_rad=0.0,
                foot_velocity_warn=0.0,
            )

            self.assertEqual(summary["sequences"], 2)
            self.assertEqual(summary["transitions"], 2)
            self.assertGreaterEqual(summary["bad_transition_count"], 0)
            self.assertTrue((tmp / "quality.json").exists())
            self.assertTrue((tmp / "transitions.jsonl").exists())

    def test_synthesize_dataset_filters_clips_with_invalid_joint_shape(self):
        from Script.stage1.synthesize_long_humanml3d import synthesize_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = _write_mixed_transition_humanml_fixture(tmp)
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
                seed=0,
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

    def test_synthesize_dataset_rejects_forced_transitions_by_default(self):
        from Script.stage1.synthesize_long_humanml3d import synthesize_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = _write_disconnected_humanml_fixture(tmp)

            with self.assertRaises(RuntimeError):
                synthesize_dataset(
                    humanml_root=root,
                    split="train",
                    num_sequences=1,
                    min_clips=2,
                    max_clips=2,
                    seed=7,
                    candidate_pool=2,
                    transition_max_score=0.001,
                    blend_frames=2,
                    caption_joiner=" then ",
                    output_dir=tmp / "out",
                )

    def test_synthesize_dataset_can_keep_forced_transitions_when_explicitly_allowed(self):
        from Script.stage1.synthesize_long_humanml3d import synthesize_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = _write_disconnected_humanml_fixture(tmp)

            summary = synthesize_dataset(
                humanml_root=root,
                split="train",
                num_sequences=1,
                min_clips=2,
                max_clips=2,
                seed=0,
                candidate_pool=2,
                transition_max_score=0.001,
                blend_frames=2,
                caption_joiner=" then ",
                output_dir=tmp / "out",
                allow_forced_transitions=True,
            )

            self.assertEqual(summary["forced_transitions"], 1)

    def test_synthesize_dataset_retries_failed_sequences_until_target_count(self):
        from Script.stage1.synthesize_long_humanml3d import synthesize_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = _write_mixed_transition_humanml_fixture(tmp)

            summary = synthesize_dataset(
                humanml_root=root,
                split="train",
                num_sequences=2,
                min_clips=2,
                max_clips=2,
                seed=0,
                candidate_pool=1,
                transition_max_score=0.001,
                blend_frames=2,
                caption_joiner=" then ",
                output_dir=tmp / "out",
                max_sequence_attempts=50,
            )

            rows = [json.loads(line) for line in (tmp / "out" / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(summary["num_sequences"], 2)
            self.assertEqual(len(rows), 2)
            self.assertGreater(summary["failed_sequences"], 0)
            self.assertGreater(summary["attempted_sequences"], summary["num_sequences"])

    def test_caption_filter_prefer_atomic_selects_simpler_caption(self):
        from Script.stage1.synthesize_long_humanml3d import synthesize_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = _write_mixed_transition_humanml_fixture(tmp)
            target_text = root / "texts" / "000001.txt"
            target_text.write_text(
                "\n".join(
                    [
                        "a person walks then turns around#a/DET person/NOUN walk/VERB then/ADV turn/VERB around/ADV#0.0#0.0",
                        "a person walks forward#a/DET person/NOUN walk/VERB forward/ADV#0.0#0.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            synthesize_dataset(
                humanml_root=root,
                split="train",
                num_sequences=1,
                min_clips=2,
                max_clips=2,
                seed=0,
                candidate_pool=3,
                transition_max_score=0.35,
                blend_frames=2,
                caption_joiner=" then ",
                caption_filter_mode="prefer_atomic",
                output_dir=tmp / "out",
            )

            rows = [json.loads(line) for line in (tmp / "out" / "manifest.jsonl").read_text().splitlines()]
            all_captions = [caption for row in rows for caption in row["clip_captions"]]
            self.assertIn("a person walks forward", all_captions)
            self.assertNotIn("a person walks then turns around", all_captions)

    def test_caption_filter_atomic_removes_samples_without_atomic_caption(self):
        from Script.stage1.synthesize_long_humanml3d import synthesize_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = _write_mixed_transition_humanml_fixture(tmp)
            for sample_id in ("000001", "000002"):
                (root / "texts" / f"{sample_id}.txt").write_text(
                    "a person walks then turns#a/DET person/NOUN walk/VERB then/ADV turn/VERB#0.0#0.0\n",
                    encoding="utf-8",
                )
            (root / "texts" / "000003.txt").write_text(
                "a person jumps#a/DET person/NOUN jump/VERB#0.0#0.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "not enough valid clips"):
                synthesize_dataset(
                    humanml_root=root,
                    split="train",
                    num_sequences=1,
                    min_clips=2,
                    max_clips=2,
                    seed=0,
                    candidate_pool=3,
                    transition_max_score=0.35,
                    blend_frames=2,
                    caption_joiner=" then ",
                    caption_filter_mode="atomic",
                    output_dir=tmp / "out",
                )


if __name__ == "__main__":
    unittest.main()
