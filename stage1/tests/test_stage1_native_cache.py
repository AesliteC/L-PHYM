import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch


class _FakeAgent:
    def encode_seq_all(self, obs, target):
        observation = torch.as_tensor(target)
        latent_len = max(1, observation.shape[0] // 4)
        latent = torch.ones((1, latent_len, 768), dtype=torch.float32)
        indices = torch.arange(latent_len * 8, dtype=torch.long).reshape(8, 1, latent_len) % 512
        return {"latent_vq": latent, "indexs": indices}


class Stage1NativeCacheTests(unittest.TestCase):
    def test_parse_motion_specs_requires_key_caption_pairs(self):
        from Script.stage1.build_native_moconvq_gpt_cache import parse_motion_specs

        self.assertEqual(
            parse_motion_specs(["walk=a person walks", "kick=a person kicks"]),
            [("walk", "a person walks"), ("kick", "a person kicks")],
        )
        with self.assertRaises(ValueError):
            parse_motion_specs(["missing_separator"])

    def test_build_native_cache_from_h5_matches_training_cache_schema(self):
        from Script.stage1.build_native_moconvq_gpt_cache import build_native_cache_from_h5

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            native_h5 = tmp / "native.h5"
            with h5py.File(native_h5, "w") as handle:
                group = handle.create_group("walk")
                group.create_dataset("observation", data=np.zeros((24, 323), dtype=np.float32))

            def fake_text_encoder(captions):
                self.assertEqual(captions, ["a person walks"])
                return (
                    np.ones((1, 8, 1024), dtype=np.float32),
                    np.zeros((1, 8), dtype=bool),
                )

            cache = build_native_cache_from_h5(
                native_h5_path=native_h5,
                motion_specs=[("walk", "a person walks")],
                agent=_FakeAgent(),
                text_encoder=fake_text_encoder,
                window_size=10,
                window_stride=5,
                rvq_depth=4,
            )

        self.assertEqual(cache["latents"].shape, (1, 10, 768))
        self.assertEqual(cache["indices"].shape, (1, 10, 4))
        self.assertEqual(cache["text_features"].shape, (1, 8, 1024))
        self.assertEqual(cache["text_masks"].shape, (1, 8))
        self.assertEqual(cache["target_masks"].shape, (1, 10))
        self.assertEqual(cache["end_masks"].shape, (1, 10))
        self.assertEqual(cache["captions"], ["a person walks"])
        self.assertEqual(cache["sequence_ids"], ["walk"])
        self.assertEqual(cache["window_ranges"], [(0, 6)])
        self.assertEqual(cache["config"]["source"], "native_moconvq_observation_h5")
        self.assertEqual(cache["config"]["source_observation_shapes"]["walk"], [24, 323])

    def test_build_native_cache_rejects_oversized_motion_window(self):
        from Script.stage1.build_native_moconvq_gpt_cache import build_native_cache_from_h5

        with tempfile.TemporaryDirectory() as tmpdir:
            native_h5 = Path(tmpdir) / "native.h5"
            with h5py.File(native_h5, "w") as handle:
                group = handle.create_group("walk")
                group.create_dataset("observation", data=np.zeros((24, 323), dtype=np.float32))

            def fake_text_encoder(captions):
                return (
                    np.ones((len(captions), 8, 1024), dtype=np.float32),
                    np.zeros((len(captions), 8), dtype=bool),
                )

            with self.assertRaisesRegex(ValueError, "window_size"):
                build_native_cache_from_h5(
                    native_h5_path=native_h5,
                    motion_specs=[("walk", "a person walks")],
                    agent=_FakeAgent(),
                    text_encoder=fake_text_encoder,
                    window_size=52,
                    window_stride=5,
                    rvq_depth=4,
                )


if __name__ == "__main__":
    unittest.main()
