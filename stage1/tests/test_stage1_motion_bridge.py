import unittest
from pathlib import Path

import numpy as np

from Script.stage1.motion_bridge import lift_motion_vec_to_latent, quantize_rvq_sequence


class MotionBridgeStage1Tests(unittest.TestCase):
    def test_lift_motion_vec_to_latent_uses_768_dimensions(self):
        root = Path(__file__).resolve().parents[2] / "HumanML3D" / "HumanML3D"
        vec = np.load(root / "new_joint_vecs" / "000001.npy")[:2]
        mean = np.load(root / "Mean.npy")
        std = np.load(root / "Std.npy")

        latent = lift_motion_vec_to_latent(vec, mean, std)
        self.assertEqual(latent.shape, (2, 768))
        self.assertTrue(np.isfinite(latent).all())

    def test_quantize_rvq_sequence_returns_four_depth_indices(self):
        latent = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        codebooks = [
            np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]], dtype=np.float32),
        ]

        result = quantize_rvq_sequence(latent, codebooks)
        self.assertEqual(result.indices.shape, (2, 4))
        self.assertEqual(result.latent_vq.shape, (2, 4))
        self.assertEqual(result.latent_vq.dtype, np.float32)

    def test_extract_rvq_embeddings_from_state_dict_reads_four_codebooks(self):
        import torch
        from Script.stage1.motion_bridge import extract_rvq_embeddings_from_state_dict

        state = torch.load(Path(__file__).resolve().parents[1] / "moconvq_base.data", map_location="cpu")
        embeddings = extract_rvq_embeddings_from_state_dict(state)
        self.assertEqual(len(embeddings), 8)
        self.assertEqual(embeddings[0].shape, (512, 768))


if __name__ == "__main__":
    unittest.main()
