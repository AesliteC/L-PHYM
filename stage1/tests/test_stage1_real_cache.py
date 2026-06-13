import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch


def _toy_joints(length: int = 24) -> np.ndarray:
    joints = np.zeros((length, 22, 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    joints[:, 0, :] = np.stack([0.1 * t, np.ones_like(t), np.zeros_like(t)], axis=-1)
    offsets = {
        1: [0.12, -0.18, 0.0],
        2: [-0.12, -0.18, 0.0],
        3: [0.0, 0.15, 0.0],
        4: [0.12, -0.45, 0.0],
        5: [-0.12, -0.45, 0.0],
        6: [0.0, 0.35, 0.0],
        7: [0.12, -0.75, 0.08],
        8: [-0.12, -0.75, 0.08],
        9: [0.0, 0.55, 0.0],
        10: [0.12, -0.82, 0.2],
        11: [-0.12, -0.82, 0.2],
        12: [0.0, 0.78, 0.0],
        13: [0.22, 0.5, 0.0],
        14: [-0.22, 0.5, 0.0],
        15: [0.0, 0.92, 0.0],
        16: [0.45, 0.42, 0.0],
        17: [-0.45, 0.42, 0.0],
        18: [0.66, 0.25, 0.0],
        19: [-0.66, 0.25, 0.0],
        20: [0.82, 0.12, 0.0],
        21: [-0.82, 0.12, 0.0],
    }
    for joint_id, offset in offsets.items():
        joints[:, joint_id, :] = joints[:, 0, :] + np.asarray(offset, dtype=np.float32)
    return joints


def _toy_joint_vecs(length: int = 24) -> np.ndarray:
    vecs = np.zeros((length, 263), dtype=np.float32)
    rot_start = 4 + 21 * 3
    rot = np.zeros((length, 21, 6), dtype=np.float32)
    rot[..., 0] = 1.0
    rot[..., 4] = 1.0
    vecs[:, rot_start : rot_start + 21 * 6] = rot.reshape(length, -1)
    return vecs


class _FakeAgent:
    obs_mean = torch.zeros(323, dtype=torch.float32)
    obs_std = torch.ones(323, dtype=torch.float32)

    def encode_seq_all(self, obs, target):
        self.last_target_shape = target.shape
        latent = torch.ones((1, 6, 768), dtype=torch.float32)
        indices = torch.arange(6 * 8, dtype=torch.long).reshape(8, 1, 6) % 512
        return {"latent_vq": latent, "indexs": indices}


class Stage1RealCacheTests(unittest.TestCase):
    def test_humanml_mapping_preserves_left_right_body_order(self):
        from Script.stage1.real_moconvq_cache import (
            HUMANML3D_TO_MOCONVQ,
            MOCONVQ_BODY_NAMES,
        )

        mapping = dict(zip(MOCONVQ_BODY_NAMES, HUMANML3D_TO_MOCONVQ))
        self.assertEqual(mapping["rUpperLeg"], 2)
        self.assertEqual(mapping["lUpperLeg"], 1)
        self.assertEqual(mapping["rFoot"], 8)
        self.assertEqual(mapping["lFoot"], 7)
        self.assertEqual(mapping["rToes"], 11)
        self.assertEqual(mapping["lToes"], 10)
        self.assertEqual(mapping["rUpperArm"], 17)
        self.assertEqual(mapping["lUpperArm"], 16)
        self.assertEqual(mapping["rHand"], 21)
        self.assertEqual(mapping["lHand"], 20)

    def test_retarget_state_and_observation_have_moconvq_shapes(self):
        from Script.stage1.real_moconvq_cache import (
            humanml3d_joints_to_moconvq_state,
            moconvq_state_to_observation,
        )

        state = humanml3d_joints_to_moconvq_state(_toy_joints(), fps=20)
        self.assertEqual(state.shape, (24, 20, 13))
        self.assertTrue(np.isfinite(state).all())

        observation = moconvq_state_to_observation(state)
        self.assertEqual(observation.shape, (24, 323))
        self.assertTrue(np.isfinite(observation).all())

    def test_humanml_vec6d_rotation_source_has_moconvq_shapes(self):
        from Script.stage1.real_moconvq_cache import humanml3d_joints_to_moconvq_state

        state = humanml3d_joints_to_moconvq_state(
            _toy_joints(),
            joint_vecs_263=_toy_joint_vecs(),
            fps=20,
            rotation_source="humanml_vec6d",
        )

        self.assertEqual(state.shape, (24, 20, 13))
        self.assertTrue(np.isfinite(state).all())

    def test_humanml_vec6d_recovery_matches_processed_humanml_joints(self):
        hml_root = Path(__file__).resolve().parents[2] / "HumanML3D" / "HumanML3D"
        vec_paths = sorted((hml_root / "new_joint_vecs").glob("*.npy"))
        if not vec_paths:
            self.skipTest("HumanML3D new_joint_vecs are not available")

        from Script.stage1.real_moconvq_cache import _cont6d_to_matrix, _humanml3d_root_yaw_matrices

        sample_id = vec_paths[0].stem
        vecs = np.load(hml_root / "new_joint_vecs" / f"{sample_id}.npy")
        joints = np.load(hml_root / "new_joints" / f"{sample_id}.npy")

        frames = len(vecs)
        rot_start = 4 + 21 * 3
        cont6d = np.zeros((frames, 22, 6), dtype=np.float32)
        cont6d[:, 0, 0] = 1.0
        cont6d[:, 0, 4] = 1.0
        cont6d[:, 1:] = vecs[:, rot_start : rot_start + 21 * 6].reshape(frames, 21, 6)
        local_mats = _cont6d_to_matrix(cont6d)
        root_mats = _humanml3d_root_yaw_matrices(vecs)
        chains = (
            (0, 2, 5, 8, 11),
            (0, 1, 4, 7, 10),
            (0, 3, 6, 9, 12, 15),
            (9, 14, 17, 19, 21),
            (9, 13, 16, 18, 20),
        )
        raw_offsets = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )
        parents = [-1] * 22
        for chain in chains:
            for idx in range(1, len(chain)):
                parents[chain[idx]] = chain[idx - 1]
        offsets = raw_offsets.copy()
        for joint_id in range(1, 22):
            bone_length = np.linalg.norm(joints[0, joint_id] - joints[0, parents[joint_id]])
            offsets[joint_id] = raw_offsets[joint_id] * bone_length
        recovered = np.zeros_like(joints)
        recovered[:, 0] = joints[:, 0]
        for chain in chains:
            current = root_mats.copy()
            for idx in range(1, len(chain)):
                joint_id = chain[idx]
                parent_id = chain[idx - 1]
                current = np.matmul(current, local_mats[:, joint_id])
                recovered[:, joint_id] = (
                    current @ offsets[joint_id][None, :, None]
                ).squeeze(-1) + recovered[:, parent_id]

        self.assertLess(float(np.linalg.norm(recovered - joints, axis=-1).mean()), 1e-4)

    def test_rest_rotation_calibration_aligns_heuristic_rest_with_moconvq_world(self):
        from Script.stage1.real_moconvq_cache import (
            apply_moconvq_rotation_calibration,
            moconvq_rest_rotation_reference,
        )

        heuristic_rest, target_rest = moconvq_rest_rotation_reference()
        calibrated = apply_moconvq_rotation_calibration(heuristic_rest[None], mode="rest")[0]
        quat_dots = np.abs(np.sum(calibrated * target_rest, axis=-1))
        self.assertTrue(np.allclose(quat_dots, np.ones_like(quat_dots), atol=1e-5))

    def test_build_cache_with_injected_agent_and_text_encoder_pads_short_windows(self):
        from Script.stage1.real_moconvq_cache import build_cache_from_long_h5

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_h5 = tmp / "long.h5"
            manifest = tmp / "manifest.jsonl"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints())
                group.create_dataset("joint_vecs_263", data=_toy_joint_vecs())
                group.attrs["caption"] = "walk then turn"
                group.attrs["sample_ids"] = "000001,000002"
            manifest.write_text(
                '{"sequence_id":"seq_000","caption":"walk then turn","sample_ids":["000001","000002"]}\n',
                encoding="utf-8",
            )

            def fake_text_encoder(captions):
                self.assertEqual(captions, ["walk then turn"])
                return (
                    np.ones((1, 8, 1024), dtype=np.float32),
                    np.zeros((1, 8), dtype=bool),
                )

            cache, failures = build_cache_from_long_h5(
                long_h5_path=long_h5,
                manifest_path=manifest,
                agent=_FakeAgent(),
                text_encoder=fake_text_encoder,
                window_size=10,
                window_stride=5,
                rvq_depth=4,
                fps=20,
                rotation_source="humanml_vec6d",
            )

            self.assertEqual(failures, [])
            self.assertEqual(cache["latents"].shape, (1, 10, 768))
            self.assertEqual(cache["indices"].shape, (1, 10, 4))
            self.assertEqual(cache["text_features"].shape, (1, 8, 1024))
            self.assertEqual(cache["indices"][0, 6:].unique().item(), 513)
            self.assertEqual(cache["window_ranges"], [(0, 6)])
            self.assertEqual(cache["config"]["rotation_source"], "humanml_vec6d")
            self.assertEqual(cache["config"]["rotation_calibration"], "rest")

    def test_build_cache_can_filter_observation_outliers_without_conversion_failure(self):
        from Script.stage1.real_moconvq_cache import build_cache_from_long_h5

        class OutlierAgent(_FakeAgent):
            obs_mean = torch.full((323,), -1000.0, dtype=torch.float32)
            obs_std = torch.ones(323, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_h5 = tmp / "long.h5"
            manifest = tmp / "manifest.jsonl"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints())
                group.attrs["caption"] = "walk"
                group.attrs["sample_ids"] = "000001"
            manifest.write_text(
                '{"sequence_id":"seq_000","caption":"walk","sample_ids":["000001"]}\n',
                encoding="utf-8",
            )

            def fake_text_encoder(captions):
                return (
                    np.ones((len(captions), 8, 1024), dtype=np.float32),
                    np.zeros((len(captions), 8), dtype=bool),
                )

            cache, failures = build_cache_from_long_h5(
                long_h5_path=long_h5,
                manifest_path=manifest,
                agent=OutlierAgent(),
                text_encoder=fake_text_encoder,
                window_size=10,
                window_stride=5,
                rvq_depth=4,
                fps=20,
                max_observation_p99_abs_z=10.0,
            )

            self.assertEqual(failures, [])
            self.assertEqual(cache["latents"].shape[0], 0)
            self.assertEqual(len(cache["filtered_sequences"]), 1)
            self.assertEqual(cache["filtered_sequences"][0]["sequence_id"], "seq_000")
            self.assertIn("p99_abs_z", cache["filtered_sequences"][0]["reason"])
            self.assertEqual(cache["config"]["max_observation_p99_abs_z"], 10.0)

    def test_cache_rejects_windows_that_exceed_gpt_temporal_context(self):
        from Script.stage1.real_moconvq_cache import build_cache_from_long_h5

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_h5 = tmp / "long.h5"
            manifest = tmp / "manifest.jsonl"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints())
                group.attrs["caption"] = "walk"
                group.attrs["sample_ids"] = "000001"
            manifest.write_text(
                '{"sequence_id":"seq_000","caption":"walk","sample_ids":["000001"]}\n',
                encoding="utf-8",
            )

            def fake_text_encoder(captions):
                return (
                    np.ones((len(captions), 8, 1024), dtype=np.float32),
                    np.zeros((len(captions), 8), dtype=bool),
                )

            with self.assertRaisesRegex(ValueError, "window_size"):
                build_cache_from_long_h5(
                    long_h5_path=long_h5,
                    manifest_path=manifest,
                    agent=_FakeAgent(),
                    text_encoder=fake_text_encoder,
                    window_size=52,
                    window_stride=5,
                    rvq_depth=4,
                    fps=20,
                )

    def test_build_cache_can_use_boundary_aligned_window_captions(self):
        from Script.stage1.real_moconvq_cache import build_cache_from_long_h5

        class WindowAgent:
            def encode_seq_all(self, obs, target):
                latent = torch.ones((1, 12, 768), dtype=torch.float32)
                indices = torch.arange(12 * 8, dtype=torch.long).reshape(8, 1, 12) % 512
                return {"latent_vq": latent, "indexs": indices}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_h5 = tmp / "long.h5"
            manifest = tmp / "manifest.jsonl"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints(length=48))
                group.create_dataset("clip_boundaries", data=np.asarray([[0, 24], [24, 48]], dtype=np.int32))
                group.attrs["caption"] = "walk then kick"
                group.attrs["sample_ids"] = "000001,000002"
            manifest.write_text(
                json_line := (
                    '{"sequence_id":"seq_000","caption":"walk then kick",'
                    '"clip_captions":["walk","kick"],'
                    '"clip_boundaries":[[0,24],[24,48]],'
                    '"sample_ids":["000001","000002"]}\n'
                ),
                encoding="utf-8",
            )
            self.assertIn("clip_captions", json_line)

            seen = []

            def fake_text_encoder(captions):
                seen.extend(captions)
                return (
                    np.ones((len(captions), 8, 1024), dtype=np.float32),
                    np.zeros((len(captions), 8), dtype=bool),
                )

            cache, failures = build_cache_from_long_h5(
                long_h5_path=long_h5,
                manifest_path=manifest,
                agent=WindowAgent(),
                text_encoder=fake_text_encoder,
                window_size=6,
                window_stride=6,
                rvq_depth=4,
                fps=20,
                caption_mode="window",
            )

            self.assertEqual(failures, [])
            self.assertEqual(cache["window_ranges"], [(0, 6), (6, 12)])
            self.assertEqual(cache["captions"], ["walk", "kick"])
            self.assertEqual(seen, ["walk", "kick"])

    def test_build_cache_defaults_to_boundary_aligned_window_captions(self):
        from Script.stage1.real_moconvq_cache import build_cache_from_long_h5

        class WindowAgent:
            def encode_seq_all(self, obs, target):
                latent = torch.ones((1, 12, 768), dtype=torch.float32)
                indices = torch.arange(12 * 8, dtype=torch.long).reshape(8, 1, 12) % 512
                return {"latent_vq": latent, "indexs": indices}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_h5 = tmp / "long.h5"
            manifest = tmp / "manifest.jsonl"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints(length=48))
                group.create_dataset("clip_boundaries", data=np.asarray([[0, 24], [24, 48]], dtype=np.int32))
                group.attrs["caption"] = "walk then kick"
                group.attrs["sample_ids"] = "000001,000002"
            manifest.write_text(
                (
                    '{"sequence_id":"seq_000","caption":"walk then kick",'
                    '"clip_captions":["walk","kick"],'
                    '"clip_boundaries":[[0,24],[24,48]],'
                    '"sample_ids":["000001","000002"]}\n'
                ),
                encoding="utf-8",
            )

            seen = []

            def fake_text_encoder(captions):
                seen.extend(captions)
                return (
                    np.ones((len(captions), 8, 1024), dtype=np.float32),
                    np.zeros((len(captions), 8), dtype=bool),
                )

            cache, failures = build_cache_from_long_h5(
                long_h5_path=long_h5,
                manifest_path=manifest,
                agent=WindowAgent(),
                text_encoder=fake_text_encoder,
                window_size=6,
                window_stride=6,
                rvq_depth=4,
                fps=20,
            )

            self.assertEqual(failures, [])
            self.assertEqual(cache["config"]["caption_mode"], "window")
            self.assertEqual(cache["captions"], ["walk", "kick"])
            self.assertEqual(seen, ["walk", "kick"])

    def test_build_cache_defaults_to_clip_aligned_windows(self):
        from Script.stage1.real_moconvq_cache import build_cache_from_long_h5

        class WindowAgent:
            def encode_seq_all(self, obs, target):
                latent = torch.ones((1, 14, 768), dtype=torch.float32)
                indices = torch.arange(14 * 8, dtype=torch.long).reshape(8, 1, 14) % 512
                return {"latent_vq": latent, "indexs": indices}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_h5 = tmp / "long.h5"
            manifest = tmp / "manifest.jsonl"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints(length=56))
                group.create_dataset("clip_boundaries", data=np.asarray([[0, 28], [28, 56]], dtype=np.int32))
                group.attrs["caption"] = "walk then kick"
                group.attrs["sample_ids"] = "000001,000002"
            manifest.write_text(
                (
                    '{"sequence_id":"seq_000","caption":"walk then kick",'
                    '"clip_captions":["walk","kick"],'
                    '"clip_boundaries":[[0,28],[28,56]],'
                    '"sample_ids":["000001","000002"]}\n'
                ),
                encoding="utf-8",
            )

            def fake_text_encoder(captions):
                return (
                    np.ones((len(captions), 8, 1024), dtype=np.float32),
                    np.zeros((len(captions), 8), dtype=bool),
                )

            cache, failures = build_cache_from_long_h5(
                long_h5_path=long_h5,
                manifest_path=manifest,
                agent=WindowAgent(),
                text_encoder=fake_text_encoder,
                window_size=5,
                window_stride=5,
                rvq_depth=4,
                fps=20,
            )

            self.assertEqual(failures, [])
            self.assertEqual(cache["config"]["window_policy"], "clip")
            self.assertEqual(cache["window_ranges"], [(0, 5), (2, 7), (7, 12), (9, 14)])
            self.assertEqual(cache["captions"], ["walk", "walk", "kick", "kick"])

    def test_build_cache_can_drop_windows_around_forced_transitions(self):
        from Script.stage1.real_moconvq_cache import build_cache_from_long_h5

        class WindowAgent:
            def encode_seq_all(self, obs, target):
                latent = torch.ones((1, 14, 768), dtype=torch.float32)
                indices = torch.arange(14 * 8, dtype=torch.long).reshape(8, 1, 14) % 512
                return {"latent_vq": latent, "indexs": indices}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_h5 = tmp / "long.h5"
            manifest = tmp / "manifest.jsonl"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints(length=56))
                group.create_dataset("clip_boundaries", data=np.asarray([[0, 28], [28, 56]], dtype=np.int32))
                group.attrs["caption"] = "walk then kick"
                group.attrs["sample_ids"] = "000001,000002"
            manifest.write_text(
                (
                    '{"sequence_id":"seq_000","caption":"walk then kick",'
                    '"clip_captions":["walk","kick"],'
                    '"clip_boundaries":[[0,28],[28,56]],'
                    '"transition_forced":[true],'
                    '"sample_ids":["000001","000002"]}\n'
                ),
                encoding="utf-8",
            )

            def fake_text_encoder(captions):
                return (
                    np.ones((len(captions), 8, 1024), dtype=np.float32),
                    np.zeros((len(captions), 8), dtype=bool),
                )

            cache, failures = build_cache_from_long_h5(
                long_h5_path=long_h5,
                manifest_path=manifest,
                agent=WindowAgent(),
                text_encoder=fake_text_encoder,
                window_size=5,
                window_stride=5,
                rvq_depth=4,
                fps=20,
                forced_transition_margin=1,
            )

            self.assertEqual(failures, [])
            self.assertEqual(cache["window_ranges"], [(0, 5), (8, 13)])
            self.assertEqual(cache["captions"], ["walk", "kick"])

    def test_build_cache_can_make_segment_prefix_samples(self):
        from Script.stage1.real_moconvq_cache import build_cache_from_long_h5

        class WindowAgent:
            def encode_seq_all(self, obs, target):
                latent = torch.arange(1 * 12 * 768, dtype=torch.float32).reshape(1, 12, 768)
                indices = torch.arange(12 * 8, dtype=torch.long).reshape(8, 1, 12) % 512
                return {"latent_vq": latent, "indexs": indices}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_h5 = tmp / "long.h5"
            manifest = tmp / "manifest.jsonl"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints(length=48))
                group.create_dataset("clip_boundaries", data=np.asarray([[0, 24], [24, 48]], dtype=np.int32))
                group.attrs["caption"] = "walk then kick"
                group.attrs["sample_ids"] = "000001,000002"
            manifest.write_text(
                (
                    '{"sequence_id":"seq_000","caption":"walk then kick",'
                    '"clip_captions":["walk","kick"],'
                    '"clip_boundaries":[[0,24],[24,48]],'
                    '"sample_ids":["000001","000002"]}\n'
                ),
                encoding="utf-8",
            )

            def fake_text_encoder(captions):
                return (
                    np.ones((len(captions), 8, 1024), dtype=np.float32),
                    np.zeros((len(captions), 8), dtype=bool),
                )

            cache, failures = build_cache_from_long_h5(
                long_h5_path=long_h5,
                manifest_path=manifest,
                agent=WindowAgent(),
                text_encoder=fake_text_encoder,
                window_size=5,
                window_stride=3,
                rvq_depth=4,
                fps=20,
                sample_mode="segment_prefix",
                prefix_size=2,
            )

            self.assertEqual(failures, [])
            self.assertEqual(cache["config"]["sample_mode"], "segment_prefix")
            self.assertEqual(cache["captions"][:3], ["walk", "walk", "kick"])
            self.assertEqual(cache["prefix_ranges"][0], (0, 0))
            self.assertEqual(cache["prefix_ranges"][2], (4, 6))
            self.assertEqual(cache["target_ranges"][2], (6, 9))
            self.assertEqual(cache["segment_idxs"].tolist()[:3], [0, 0, 1])
            self.assertEqual(cache["num_segments"].tolist()[:3], [2, 2, 2])
            self.assertAlmostEqual(float(cache["segment_progress"][2]), 1.0)
            self.assertEqual(cache["target_masks"].shape, (4, 5))
            self.assertEqual(cache["end_masks"].shape, (4, 5))
            self.assertTrue(torch.equal(cache["target_masks"][2], torch.tensor([False, False, True, True, True])))
            self.assertFalse(bool(cache["end_masks"][2].any()))
            self.assertFalse(bool(cache["end_masks"][3].any()))

    def test_segment_prefix_windows_mark_end_only_when_padding_slot_exists(self):
        from Script.stage1.real_moconvq_cache import make_segment_prefix_windows

        latents = np.zeros((11, 2), dtype=np.float32)
        indices = np.zeros((11, 1), dtype=np.int64)

        windows = make_segment_prefix_windows(
            latents=latents,
            indices=indices,
            window_size=5,
            window_stride=3,
            clip_boundaries=[(0, 6), (6, 11)],
            prefix_size=2,
        )

        self.assertFalse(bool(windows[1]["end_mask"].any()))
        self.assertTrue(any(bool(window["end_mask"].any()) for window in windows))

    def test_cache_main_records_text_encoder_configuration(self):
        from Script.stage1 import real_moconvq_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_h5 = tmp / "long.h5"
            manifest = tmp / "manifest.jsonl"
            output = tmp / "cache.pt"
            failure_log = tmp / "failures.jsonl"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints())
                group.attrs["caption"] = "walk"
                group.attrs["sample_ids"] = "000001"
            manifest.write_text(
                '{"sequence_id":"seq_000","caption":"walk","sample_ids":["000001"]}\n',
                encoding="utf-8",
            )

            old_agent_builder = real_moconvq_cache.build_loaded_moconvq_agent
            old_text_builder = real_moconvq_cache.build_t5_text_encoder
            try:
                real_moconvq_cache.build_loaded_moconvq_agent = lambda gpu, base_data: _FakeAgent()
                real_moconvq_cache.build_t5_text_encoder = lambda model_name, device, max_length: (
                    lambda captions: (
                        np.ones((len(captions), max_length, 1024), dtype=np.float32),
                        np.zeros((len(captions), max_length), dtype=bool),
                    )
                )
                real_moconvq_cache.main(
                    [
                        "--long-h5",
                        str(long_h5),
                        "--manifest",
                        str(manifest),
                        "--base-data",
                        "fake.data",
                        "--text-model",
                        "fake-t5",
                        "--max-text-length",
                        "17",
                        "--window-size",
                        "10",
                        "--window-stride",
                        "5",
                        "--output",
                        str(output),
                        "--failure-log",
                        str(failure_log),
                    ]
                )
            finally:
                real_moconvq_cache.build_loaded_moconvq_agent = old_agent_builder
                real_moconvq_cache.build_t5_text_encoder = old_text_builder

            cache = torch.load(output, map_location="cpu")
            self.assertEqual(cache["config"]["text_model"], "fake-t5")
            self.assertEqual(cache["config"]["max_text_length"], 17)
            self.assertEqual(cache["text_features"].shape, (1, 17, 1024))

    def test_convert_long_h5_to_moconvq_observation_writes_inspection_h5(self):
        from Script.stage1.convert_humanml3d_to_moconvq_observation import (
            convert_long_h5_to_moconvq_observation,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_h5 = tmp / "long.h5"
            manifest = tmp / "manifest.jsonl"
            output_h5 = tmp / "moconvq_obs.h5"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints())
                group.create_dataset("clip_boundaries", data=np.asarray([[0, 24]], dtype=np.int32))
                group.attrs["caption"] = "walk"
                group.attrs["sample_ids"] = "000001"
                group.attrs["split"] = "train"
            manifest.write_text(
                '{"sequence_id":"seq_000","caption":"walk","sample_ids":["000001"],"split":"train"}\n',
                encoding="utf-8",
            )

            summary = convert_long_h5_to_moconvq_observation(
                long_h5_path=long_h5,
                manifest_path=manifest,
                output_h5_path=output_h5,
                fps=20,
            )

            self.assertEqual(summary["converted_sequences"], 1)
            self.assertEqual(summary["failed_sequences"], 0)
            with h5py.File(output_h5, "r") as h5:
                group = h5["seq_000"]
                self.assertEqual(group["state_20x13"].shape, (24, 20, 13))
                self.assertEqual(group["observation_323"].shape, (24, 323))
                self.assertEqual(group.attrs["caption"], "walk")


if __name__ == "__main__":
    unittest.main()
