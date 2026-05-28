import unittest
from pathlib import Path

import torch

from Script.stage1.train_text_gpt import build_text_gpt_model, gpt_config


class Stage1GPTTests(unittest.TestCase):
    def test_stage1_text_gpt_helpers_exist_and_build_model(self):
        cfg = gpt_config()
        model = build_text_gpt_model(cfg, device="cpu")
        self.assertIsNotNone(model)
        self.assertEqual(model.get_block_size(), cfg.block_size)

    def test_model_can_be_loaded_from_text_gpt_checkpoint(self):
        cfg = gpt_config()
        model = build_text_gpt_model(cfg, device="cpu")
        state = torch.load(Path(__file__).resolve().parents[1] / "text_generation_GPT.pth", map_location="cpu")
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        self.assertLess(len(unexpected), 50)

    def test_forward_drops_extra_condition_frame_when_present(self):
        cfg = gpt_config()
        model = build_text_gpt_model(cfg, device="cpu")
        latent = torch.zeros(1, 8, 768)
        indices = torch.zeros(1, 8, 4, dtype=torch.long)
        clip_feature = torch.zeros(1, 512)
        bert_feature = torch.zeros(1, 64, 1024)
        bert_mask = torch.ones(1, 64, dtype=torch.bool)

        logits, proj = model(latent, indices, clip_feature, bert_feature, bert_mask)
        self.assertEqual(logits.shape, (1, 8, 5, 513))
        self.assertEqual(proj.shape, (1, 8, 768))

    def test_sample_accepts_tensor_pre_latent_for_rolling_context(self):
        cfg = gpt_config()
        model = build_text_gpt_model(cfg, device="cpu")
        pre_latent = torch.zeros(1, 2, 768)
        clip_feature = torch.zeros(1, 512)
        bert_feature = torch.zeros(1, 8, 1024)
        bert_mask = torch.zeros(1, 8, dtype=torch.bool)

        latents, indices = model.sample(
            clip_feature,
            bert_feature,
            bert_mask,
            if_categorial=False,
            max_length=1,
            pre_latent=pre_latent,
        )

        self.assertEqual(latents.shape, pre_latent.shape)
        self.assertEqual(indices.shape[-1], 1)

    def test_sample_accepts_none_pre_latent(self):
        cfg = gpt_config()
        model = build_text_gpt_model(cfg, device="cpu")
        clip_feature = torch.zeros(1, 512)
        bert_feature = torch.zeros(1, 8, 1024)
        bert_mask = torch.zeros(1, 8, dtype=torch.bool)

        latents, indices = model.sample(
            clip_feature,
            bert_feature,
            bert_mask,
            if_categorial=False,
            max_length=2,
            pre_latent=None,
        )

        self.assertEqual(latents.shape, (1, 1, 768))
        self.assertEqual(indices.shape[-1], 1)


if __name__ == "__main__":
    unittest.main()
