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
        self.assertIn("ce_loss", metrics)
        self.assertIn("kl_loss", metrics)
        self.assertIn("end_loss", metrics)

    def test_loss_and_metrics_applies_depth_weights(self):
        from Script.stage1.train_real_text_gpt import compute_loss_and_metrics

        logits = torch.tensor(
            [
                [
                    [
                        [2.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 2.0],
                        [0.0, 0.0, 2.0],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([[[0, 1, 2, 0]]], dtype=torch.long)

        loss, metrics = compute_loss_and_metrics(
            logits,
            targets,
            depth_weights=[1.0, 0.5, 0.25, 0.0],
        )

        per_depth = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 3),
            targets.reshape(-1),
            reduction="none",
        ).reshape(1, 1, 4)
        expected = (
            per_depth[0, 0, 0] * 1.0
            + per_depth[0, 0, 1] * 0.5
            + per_depth[0, 0, 2] * 0.25
        ) / 1.75
        self.assertAlmostEqual(float(loss), float(expected), places=6)
        self.assertAlmostEqual(metrics["ce_loss"], float(expected), places=6)

    def test_loss_and_metrics_adds_baseline_kl_distillation(self):
        from Script.stage1.train_real_text_gpt import compute_loss_and_metrics

        logits = torch.tensor([[[[3.0, 0.0, 0.0, 0.0]]]], dtype=torch.float32)
        teacher_logits = torch.tensor([[[[0.0, 3.0, 0.0, 0.0]]]], dtype=torch.float32)
        targets = torch.tensor([[[0]]], dtype=torch.long)

        without_kl, metrics_without = compute_loss_and_metrics(logits, targets)
        with_kl, metrics_with = compute_loss_and_metrics(
            logits,
            targets,
            teacher_logits=teacher_logits,
            kl_weight=0.5,
            kl_temperature=1.0,
        )

        self.assertGreater(metrics_with["kl_loss"], 0.0)
        self.assertAlmostEqual(metrics_without["kl_loss"], 0.0)
        self.assertGreater(float(with_kl), float(without_kl))

    def test_loss_and_metrics_adds_end_token_loss_at_first_padding_step(self):
        from Script.stage1.train_real_text_gpt import compute_loss_and_metrics

        logits = torch.zeros((1, 3, 2, 514), dtype=torch.float32)
        targets = torch.tensor([[[7, 8], [513, 513], [513, 513]]], dtype=torch.long)
        logits[:, :, :, 512] = 10.0

        loss, metrics = compute_loss_and_metrics(
            logits,
            targets,
            end_token_weight=0.25,
            end_token_id=512,
        )

        self.assertEqual(metrics["end_tokens"], 2)
        self.assertLess(metrics["end_loss"], 0.1)
        self.assertGreater(float(loss), metrics["ce_loss"])

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

    def test_prepare_autoregressive_inputs_uses_only_previous_latents(self):
        from Script.stage1.train_real_text_gpt import prepare_autoregressive_inputs

        latents = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
        indices = torch.arange(1 * 5 * 4, dtype=torch.long).reshape(1, 5, 4)

        context_latents, targets = prepare_autoregressive_inputs(latents, indices)

        self.assertEqual(context_latents.shape, (1, 4, 2))
        self.assertEqual(targets.shape, (1, 5, 4))
        self.assertTrue(torch.equal(context_latents, latents[:, :-1, :]))
        self.assertTrue(torch.equal(targets, indices))

    def test_reconstruct_latents_from_rvq_indices_matches_gpt_sampling_space(self):
        from Script.stage1.train_real_text_gpt import reconstruct_latents_from_rvq_indices

        embeddings = [
            torch.tensor(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
            torch.tensor(
                [
                    [0.0, 10.0],
                    [0.0, 20.0],
                    [0.0, 0.0],
                ]
            ),
        ]
        indices = torch.tensor(
            [
                [
                    [0, 0],
                    [1, 1],
                    [513, 513],
                ]
            ],
            dtype=torch.long,
        )

        latents = reconstruct_latents_from_rvq_indices(indices, embeddings, pad_index=513)

        expected = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [0.0, 0.0]]])
        self.assertTrue(torch.equal(latents, expected))

    def test_select_rvq_logits_uses_first_four_depth_slots(self):
        from Script.stage1.train_real_text_gpt import select_rvq_logits_for_targets

        logits = torch.arange(1 * 3 * 5 * 7, dtype=torch.float32).reshape(1, 3, 5, 7)
        targets = torch.zeros((1, 3, 4), dtype=torch.long)

        selected = select_rvq_logits_for_targets(logits, targets)

        self.assertEqual(selected.shape, (1, 3, 4, 7))
        self.assertTrue(torch.equal(selected, logits[:, :, :4, :]))

    def test_configure_trainable_scope_can_freeze_temporal_text_encoder(self):
        import torch.nn as nn

        from Script.stage1.train_real_text_gpt import configure_trainable_scope

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.trans_temporal = nn.Linear(2, 2)
                self.trans_base = nn.Linear(2, 2)
                self.trans_head = nn.Linear(2, 2)
                self.linear = nn.Linear(2, 2)

        model = FakeModel()
        count = configure_trainable_scope(model, "base_head")

        self.assertGreater(count, 0)
        self.assertFalse(any(param.requires_grad for param in model.trans_temporal.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.trans_base.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.trans_head.parameters()))
        self.assertFalse(any(param.requires_grad for param in model.linear.parameters()))

    def test_validate_output_dir_rejects_nonempty_clean_run(self):
        from Script.stage1.train_real_text_gpt import validate_output_dir_for_training

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "train_log.jsonl").write_text("old\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                validate_output_dir_for_training(output_dir, append_log=False)

            validate_output_dir_for_training(output_dir, append_log=True)

    def test_training_run_lock_rejects_concurrent_writer(self):
        from Script.stage1.train_real_text_gpt import training_run_lock

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with training_run_lock(output_dir, metadata={"test": True}):
                self.assertTrue((output_dir / ".train.lock").exists())
                with self.assertRaises(RuntimeError):
                    with training_run_lock(output_dir):
                        pass
            self.assertFalse((output_dir / ".train.lock").exists())


if __name__ == "__main__":
    unittest.main()
