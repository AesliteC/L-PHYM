import unittest


class Stage1RealGenerateTests(unittest.TestCase):
    def test_t5_and_hash_text_encoders_return_expected_tensor_shapes_with_stubs(self):
        import sys
        import types

        import torch

        from Script.stage1.generate_long_motion import encode_text_with_hash, encode_text_with_t5

        transformers = types.ModuleType("transformers")

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, model_name):
                return cls()

            def __call__(self, texts, return_tensors, padding, truncation, max_length):
                assert return_tensors == "pt"
                return {
                    "input_ids": torch.zeros((len(texts), max_length), dtype=torch.long),
                    "attention_mask": torch.ones((len(texts), max_length), dtype=torch.long),
                }

        class FakeEncoder:
            @classmethod
            def from_pretrained(cls, model_name):
                return cls()

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, **encoded):
                batch, length = encoded["input_ids"].shape
                return types.SimpleNamespace(last_hidden_state=torch.zeros((batch, length, 1024)))

        transformers.T5Tokenizer = FakeTokenizer
        transformers.T5EncoderModel = FakeEncoder
        old_transformers = sys.modules.get("transformers")
        sys.modules["transformers"] = transformers
        try:
            feature, mask = encode_text_with_t5("walk forward", "fake-t5", 12, "cpu")
        finally:
            if old_transformers is None:
                del sys.modules["transformers"]
            else:
                sys.modules["transformers"] = old_transformers

        self.assertEqual(feature.shape, (1, 12, 1024))
        self.assertEqual(mask.shape, (1, 12))
        self.assertEqual(mask.dtype, torch.bool)

        hash_feature, hash_mask = encode_text_with_hash("walk forward", "cpu")
        self.assertEqual(hash_feature.shape[-1], 1024)
        self.assertEqual(hash_mask.dtype, torch.bool)

    def test_rolling_generation_keeps_model_context_within_block_size(self):
        import torch

        from Script.stage1.generate_long_motion import sample_latents_rolling

        class FakeModel:
            block_size = 8

            def __init__(self):
                self.context_lengths = []

            def get_block_size(self):
                return self.block_size

            def sample(self, clip_feature, bert_feature, bert_mask, if_categorial, max_length, pre_latent):
                self.context_lengths.append(0 if pre_latent is None else int(pre_latent.shape[1]))
                context = 0 if pre_latent is None else int(pre_latent.shape[1])
                total = context + max_length - 1
                return torch.ones((1, total, 768), dtype=torch.float32), torch.zeros((max_length, 4), dtype=torch.long)

        model = FakeModel()
        latents = sample_latents_rolling(
            model=model,
            clip_feature=torch.zeros((1, 512)),
            bert_feature=torch.zeros((1, 16, 1024)),
            bert_mask=torch.zeros((1, 16), dtype=torch.bool),
            max_length=18,
            context_size=5,
            chunk_size=4,
            categorical=False,
        )

        self.assertEqual(latents.shape, (1, 18, 768))
        self.assertLessEqual(max(model.context_lengths), 5)
        self.assertEqual(model.context_lengths, [0, 3, 3, 3, 5])


if __name__ == "__main__":
    unittest.main()
