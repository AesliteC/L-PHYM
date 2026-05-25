import tempfile
import unittest
from pathlib import Path

import torch


class Stage1RealTrainTests(unittest.TestCase):
    def test_loss_and_metrics_ignore_padding_token(self):
        from Script.stage1.train_real_text_gpt import compute_loss_and_metrics

        logits = torch.zeros((1, 2, 4, 514), dtype=torch.float32)
        targets = torch.tensor([[[1, 2, 513, 4], [1, 513, 3, 4]]], dtype=torch.long)
        for t in range(2):
            for d in range(4):
                target = int(targets[0, t, d])
                if target != 513:
                    logits[0, t, d, target] = 20.0

        loss, metrics = compute_loss_and_metrics(logits, targets, ignore_index=513)

        self.assertLess(float(loss), 0.01)
        self.assertAlmostEqual(metrics["token_accuracy"], 1.0)
        self.assertEqual(metrics["valid_tokens"], 6)
        self.assertAlmostEqual(metrics["depth_accuracy"][2], 1.0)

    def test_real_cache_dataset_loads_expected_fields(self):
        from Script.stage1.train_real_text_gpt import RealStage1CacheDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.pt"
            torch.save(
                {
                    "latents": torch.zeros((1, 5, 768), dtype=torch.float32),
                    "indices": torch.zeros((1, 5, 4), dtype=torch.long),
                    "text_features": torch.zeros((1, 8, 1024), dtype=torch.float32),
                    "text_masks": torch.ones((1, 8), dtype=torch.bool),
                    "captions": ["walk"],
                    "sequence_ids": ["seq"],
                    "window_ranges": [(0, 5)],
                    "sample_ids": [["000001"]],
                    "config": {},
                },
                path,
            )

            dataset = RealStage1CacheDataset(str(path))
            item = dataset[0]
            self.assertEqual(item["latent"].shape, (5, 768))
            self.assertEqual(item["indices"].shape, (5, 4))
            self.assertEqual(item["text_feature"].shape, (8, 1024))
            self.assertEqual(item["caption"], "walk")


if __name__ == "__main__":
    unittest.main()
