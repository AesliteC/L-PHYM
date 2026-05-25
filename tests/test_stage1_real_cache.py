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


if __name__ == "__main__":
    unittest.main()
