from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


def _write_minimal_humanml(root: Path, sample_id: str, frames: int = 4) -> Path:
    humanml = root / "HumanML3D"
    for dirname in ("texts", "new_joints", "new_joint_vecs"):
        (humanml / dirname).mkdir(parents=True, exist_ok=True)
    for split in ("all", "train", "val", "test", "train_val"):
        (humanml / f"{split}.txt").write_text(f"{sample_id}\n", encoding="utf-8")
    np.save(humanml / "Mean.npy", np.zeros(263, dtype=np.float32))
    np.save(humanml / "Std.npy", np.ones(263, dtype=np.float32))
    (humanml / "texts" / f"{sample_id}.txt").write_text("a person walks#a person walks#0#0\n", encoding="utf-8")

    joints = np.zeros((frames, 22, 3), dtype=np.float32)
    joints[:, 0, 0] = np.linspace(0.0, 0.3, frames)
    joints[:, 0, 1] = 0.9
    joints[:, 0, 2] = np.linspace(0.0, 0.6, frames)
    for joint_id in range(1, 22):
        joints[:, joint_id] = joints[:, 0] + np.array([0.01 * joint_id, 0.02 * joint_id, 0.0], dtype=np.float32)

    vecs = np.zeros((frames, 263), dtype=np.float32)
    rot_start = 4 + (22 - 1) * 3
    for joint_id in range(21):
        vecs[:, rot_start + joint_id * 6 : rot_start + joint_id * 6 + 6] = np.array(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            dtype=np.float32,
        )
    np.save(humanml / "new_joints" / f"{sample_id}.npy", joints)
    np.save(humanml / "new_joint_vecs" / f"{sample_id}.npy", vecs)
    return humanml


class Stage1HumanML3DBVHExportTests(unittest.TestCase):
    def test_direct_script_execution_prefers_own_repo_root(self):
        from Script.stage1 import export_humanml3d_to_bvh as exporter

        expected_root = Path(exporter.__file__).resolve().parents[2]
        old_path = list(exporter.sys.path)
        try:
            exporter.sys.path[:] = ["/tmp/other_checkout"] + [
                path for path in old_path if Path(path or ".").resolve() != expected_root
            ]
            exporter._ensure_own_repo_root_on_path(package="")
            self.assertEqual(Path(exporter.sys.path[0]).resolve(), expected_root)
        finally:
            exporter.sys.path[:] = old_path

    def test_exported_bvh_uses_template_channels_and_humanml_root_translation(self):
        from Script.stage1.export_humanml3d_to_bvh import write_humanml3d_bvh
        from Script.stage1.render_bvh_to_mp4 import parse_bvh

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            humanml = _write_minimal_humanml(root, "000001", frames=5)
            out = root / "exported" / "000001.bvh"

            summary = write_humanml3d_bvh(
                sample_id="000001",
                humanml_root=humanml,
                output_bvh=out,
                template_bvh=Path(__file__).resolve().parents[1] / "base.bvh",
                output_fps=20.0,
            )
            nodes, motion, frame_time = parse_bvh(out)

            joints = np.load(humanml / "new_joints" / "000001.npy")
            self.assertEqual(summary["frames"], 5)
            self.assertEqual(summary["channels"], 63)
            self.assertEqual(summary["rotation_source"], "joints_ik")
            self.assertTrue(summary["unwrap_euler"])
            self.assertEqual(motion.shape, (5, 63))
            self.assertEqual(len(nodes), 25)
            self.assertAlmostEqual(frame_time, 0.05)
            np.testing.assert_allclose(motion[:, :3], joints[:, 0, :], atol=1e-6)

    def test_joints_ik_export_aligns_major_child_bone_directions(self):
        from Script.stage1.export_humanml3d_to_bvh import (
            HUMANML3D_TO_MOCONVQ,
            humanml3d_sample_to_bvh_motion,
        )
        from Script.stage1.render_bvh_to_mp4 import frame_positions, parse_bvh

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            humanml = _write_minimal_humanml(root, "000001", frames=5)
            joints = np.load(humanml / "new_joints" / "000001.npy")
            vecs = np.load(humanml / "new_joint_vecs" / "000001.npy")
            template = Path(__file__).resolve().parents[1] / "base.bvh"

            motion = humanml3d_sample_to_bvh_motion(
                joints,
                vecs,
                template_bvh=template,
                rotation_source="joints_ik",
            )
            nodes, _template_motion, _frame_time = parse_bvh(template)
            positions = frame_positions(nodes, motion[0])
            mapped_joints = joints[:, HUMANML3D_TO_MOCONVQ]

            for parent_node, child_node, parent_body, child_body in [
                (0, 1, 0, 1),
                (0, 5, 0, 3),
                (0, 9, 0, 4),
            ]:
                exported_vec = positions[child_node] - positions[parent_node]
                target_vec = mapped_joints[0, child_body] - mapped_joints[0, parent_body]
                cosine = float(np.dot(exported_vec, target_vec) / (np.linalg.norm(exported_vec) * np.linalg.norm(target_vec)))
                self.assertGreater(cosine, 0.99)

    def test_vec6d_rotation_source_is_still_available_for_reproducibility(self):
        from Script.stage1.export_humanml3d_to_bvh import write_humanml3d_bvh

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            humanml = _write_minimal_humanml(root, "000001", frames=5)
            out = root / "exported" / "000001.bvh"

            summary = write_humanml3d_bvh(
                sample_id="000001",
                humanml_root=humanml,
                output_bvh=out,
                template_bvh=Path(__file__).resolve().parents[1] / "base.bvh",
                rotation_source="vec6d",
                unwrap_euler=False,
            )

            self.assertEqual(summary["rotation_source"], "vec6d")
            self.assertFalse(summary["unwrap_euler"])

    def test_export_rejects_unknown_rotation_source(self):
        from Script.stage1.export_humanml3d_to_bvh import humanml3d_sample_to_bvh_motion

        joints = np.zeros((3, 22, 3), dtype=np.float32)
        vecs = np.zeros((3, 263), dtype=np.float32)
        with self.assertRaises(ValueError):
            humanml3d_sample_to_bvh_motion(
                joints,
                vecs,
                template_bvh=Path(__file__).resolve().parents[1] / "base.bvh",
                rotation_source="bad",
            )

    def test_export_rejects_too_short_processed_motion(self):
        from Script.stage1.export_humanml3d_to_bvh import write_humanml3d_bvh

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            humanml = _write_minimal_humanml(root, "000001", frames=1)

            with self.assertRaises(ValueError):
                write_humanml3d_bvh(
                    sample_id="000001",
                    humanml_root=humanml,
                    output_bvh=root / "out.bvh",
                    template_bvh=Path(__file__).resolve().parents[1] / "base.bvh",
                )


if __name__ == "__main__":
    unittest.main()
