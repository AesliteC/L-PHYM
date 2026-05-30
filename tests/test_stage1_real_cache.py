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


class _FakeAgent:
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

    def test_build_cache_with_injected_agent_and_text_encoder_pads_short_windows(self):
        from Script.stage1.real_moconvq_cache import build_cache_from_long_h5

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_h5 = tmp / "long.h5"
            manifest = tmp / "manifest.jsonl"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints())
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
            )

            self.assertEqual(failures, [])
            self.assertEqual(cache["latents"].shape, (1, 10, 768))
            self.assertEqual(cache["indices"].shape, (1, 10, 4))
            self.assertEqual(cache["text_features"].shape, (1, 8, 1024))
            self.assertEqual(cache["indices"][0, 6:].unique().item(), 513)
            self.assertEqual(cache["window_ranges"], [(0, 6)])

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
