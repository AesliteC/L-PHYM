import unittest


class Stage1RealGenerateTests(unittest.TestCase):
    def test_direct_script_execution_prefers_own_repo_root(self):
        from pathlib import Path

        import Script.stage1.generate_long_motion as generate

        expected_root = Path(generate.__file__).resolve().parents[2]
        old_path = list(generate.sys.path)

        try:
            generate.sys.path[:] = ["/tmp/other_checkout"] + [
                path for path in old_path if Path(path or ".").resolve() != expected_root
            ]
            generate._ensure_own_repo_root_on_path(package="")
            self.assertEqual(Path(generate.sys.path[0]).resolve(), expected_root)
        finally:
            generate.sys.path[:] = old_path

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
                self.sampling_args = []

            def get_block_size(self):
                return self.block_size

            def sample(
                self,
                clip_feature,
                bert_feature,
                bert_mask,
                if_categorial,
                max_length,
                pre_latent,
                top_k=50,
                top_p=1.0,
                temperature=1.0,
            ):
                self.context_lengths.append(0 if pre_latent is None else int(pre_latent.shape[1]))
                self.sampling_args.append((top_k, top_p, temperature))
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
            top_k=7,
            top_p=0.9,
            temperature=0.8,
        )

        self.assertEqual(latents.shape, (1, 18, 768))
        self.assertLessEqual(max(model.context_lengths), 5)
        self.assertEqual(model.context_lengths, [0, 3, 3, 3, 5])
        self.assertEqual(model.sampling_args, [(7, 0.9, 0.8)] * 5)

    def test_segmented_generation_uses_local_text_per_segment(self):
        import torch

        import Script.stage1.generate_long_motion as generate

        class FakeModel:
            def get_block_size(self):
                return 8

        calls = []

        def fake_text_encoder(text, model_name, max_length, device):
            calls.append(text)
            value = float(len(calls))
            return torch.full((1, max_length, 1024), value), torch.zeros((1, max_length), dtype=torch.bool)

        def fake_with_prefix(
            model,
            clip_feature,
            bert_feature,
            bert_mask,
            max_length,
            prefix_latents,
            context_size,
            chunk_size,
            categorical,
            allow_early_stop,
            top_k=50,
            top_p=1.0,
            temperature=1.0,
            **kwargs,
        ):
            segment_value = bert_feature[0, 0, 0]
            if clip_feature is None:
                pass
            return torch.full((1, max_length, 768), segment_value)

        old_encode = generate.encode_text_with_t5
        old_with_prefix = generate.sample_latents_with_prefix
        generate.encode_text_with_t5 = fake_text_encoder
        generate.sample_latents_with_prefix = fake_with_prefix
        try:
            latents = generate.sample_latents_segmented(
                model=FakeModel(),
                clip_feature=torch.zeros((1, 512)),
                text_segments=["walk forward", "turn left"],
                text_encoder="t5",
                text_model="fake-t5",
                max_text_length=4,
                device="cpu",
                segment_length=3,
                context_size=5,
                chunk_size=2,
                categorical=False,
                allow_early_stop=False,
            )
        finally:
            generate.encode_text_with_t5 = old_encode
            generate.sample_latents_with_prefix = old_with_prefix

        self.assertEqual(calls, ["walk forward", "turn left"])
        self.assertEqual(latents.shape, (1, 6, 768))
        self.assertTrue(torch.equal(latents[:, :3, :], torch.ones((1, 3, 768))))
        self.assertTrue(torch.equal(latents[:, 3:, :], torch.full((1, 3, 768), 2.0)))

    def test_segmented_generation_carries_previous_segment_latents_as_context(self):
        import torch

        import Script.stage1.generate_long_motion as generate

        class FakeModel:
            def __init__(self):
                self.pre_context_lengths = []

            def get_block_size(self):
                return 8

            def sample(
                self,
                clip_feature,
                bert_feature,
                bert_mask,
                if_categorial,
                max_length,
                pre_latent,
                top_k=50,
                top_p=1.0,
                temperature=1.0,
            ):
                context = 0 if pre_latent is None else int(pre_latent.shape[1])
                self.pre_context_lengths.append(context)
                value = float(bert_feature[0, 0, 0])
                if context == 0:
                    return torch.full((1, max_length - 1, 768), value), torch.zeros((max_length, 4), dtype=torch.long)
                prefix = pre_latent
                generated = torch.full((1, max_length - 1, 768), value)
                return torch.cat([prefix, generated], dim=1), torch.zeros((max_length, 4), dtype=torch.long)

        calls = []

        def fake_text_encoder(text, model_name, max_length, device):
            calls.append(text)
            return torch.full((1, max_length, 1024), float(len(calls))), torch.zeros((1, max_length), dtype=torch.bool)

        old_encode = generate.encode_text_with_t5
        generate.encode_text_with_t5 = fake_text_encoder
        try:
            model = FakeModel()
            latents = generate.sample_latents_segmented(
                model=model,
                clip_feature=torch.zeros((1, 512)),
                text_segments=["walk forward", "turn left"],
                text_encoder="t5",
                text_model="fake-t5",
                max_text_length=4,
                device="cpu",
                segment_length=3,
                context_size=2,
                chunk_size=3,
                categorical=False,
                allow_early_stop=False,
            )
        finally:
            generate.encode_text_with_t5 = old_encode

        self.assertEqual(calls, ["walk forward", "turn left"])
        self.assertEqual(model.pre_context_lengths, [0, 2])
        self.assertEqual(latents.shape, (1, 6, 768))
        self.assertTrue(torch.equal(latents[:, :3, :], torch.ones((1, 3, 768))))
        self.assertTrue(torch.equal(latents[:, 3:, :], torch.full((1, 3, 768), 2.0)))

    def test_segment_lengths_parser_requires_one_length_per_segment(self):
        from Script.stage1.generate_long_motion import parse_segment_lengths

        self.assertEqual(parse_segment_lengths("2, 3,4", expected_count=3), [2, 3, 4])
        self.assertIsNone(parse_segment_lengths(None, expected_count=3))
        with self.assertRaises(ValueError):
            parse_segment_lengths("2,3", expected_count=3)
        with self.assertRaises(ValueError):
            parse_segment_lengths("2,0,3", expected_count=3)

    def test_segmented_generation_accepts_per_segment_lengths(self):
        import torch

        import Script.stage1.generate_long_motion as generate

        class FakeModel:
            def get_block_size(self):
                return 8

            def sample(
                self,
                clip_feature,
                bert_feature,
                bert_mask,
                if_categorial,
                max_length,
                pre_latent,
                top_k=50,
                top_p=1.0,
                temperature=1.0,
            ):
                context = 0 if pre_latent is None else int(pre_latent.shape[1])
                value = float(bert_feature[0, 0, 0])
                body = torch.full((1, max_length - 1, 768), value)
                if context == 0:
                    return body, torch.zeros((max_length, 4), dtype=torch.long)
                return torch.cat([pre_latent, body], dim=1), torch.zeros((max_length, 4), dtype=torch.long)

        calls = []

        def fake_text_encoder(text, model_name, max_length, device):
            calls.append(text)
            return torch.full((1, max_length, 1024), float(len(calls))), torch.zeros((1, max_length), dtype=torch.bool)

        old_encode = generate.encode_text_with_t5
        generate.encode_text_with_t5 = fake_text_encoder
        try:
            latents = generate.sample_latents_segmented(
                model=FakeModel(),
                clip_feature=torch.zeros((1, 512)),
                text_segments=["walk", "turn", "wave"],
                text_encoder="t5",
                text_model="fake-t5",
                max_text_length=4,
                device="cpu",
                segment_length=9,
                segment_lengths=[2, 3, 4],
                context_size=2,
                chunk_size=4,
                categorical=False,
                allow_early_stop=False,
            )
        finally:
            generate.encode_text_with_t5 = old_encode

        self.assertEqual(calls, ["walk", "turn", "wave"])
        self.assertEqual(latents.shape, (1, 9, 768))
        self.assertTrue(torch.equal(latents[:, :2, :], torch.ones((1, 2, 768))))
        self.assertTrue(torch.equal(latents[:, 2:5, :], torch.full((1, 3, 768), 2.0)))
        self.assertTrue(torch.equal(latents[:, 5:, :], torch.full((1, 4, 768), 3.0)))

    def test_segmented_generation_continues_after_segment_early_stop(self):
        import torch

        import Script.stage1.generate_long_motion as generate

        class FakeModel:
            def get_block_size(self):
                return 8

        calls = []

        def fake_text_encoder(text, model_name, max_length, device):
            calls.append(text)
            return torch.full((1, max_length, 1024), float(len(calls))), torch.zeros((1, max_length), dtype=torch.bool)

        def fake_with_prefix(
            model,
            clip_feature,
            bert_feature,
            bert_mask,
            max_length,
            prefix_latents,
            context_size,
            chunk_size,
            categorical,
            allow_early_stop,
            top_k=50,
            top_p=1.0,
            temperature=1.0,
            **kwargs,
        ):
            value = float(bert_feature[0, 0, 0])
            out_len = 1 if len(calls) == 1 else max_length
            return torch.full((1, out_len, 768), value)

        old_encode = generate.encode_text_with_t5
        old_with_prefix = generate.sample_latents_with_prefix
        generate.encode_text_with_t5 = fake_text_encoder
        generate.sample_latents_with_prefix = fake_with_prefix
        try:
            latents = generate.sample_latents_segmented(
                model=FakeModel(),
                clip_feature=torch.zeros((1, 512)),
                text_segments=["walk", "turn"],
                text_encoder="t5",
                text_model="fake-t5",
                max_text_length=4,
                device="cpu",
                segment_length=3,
                segment_lengths=[3, 2],
                context_size=2,
                chunk_size=2,
                categorical=False,
                allow_early_stop=True,
            )
        finally:
            generate.encode_text_with_t5 = old_encode
            generate.sample_latents_with_prefix = old_with_prefix

        self.assertEqual(calls, ["walk", "turn"])
        self.assertEqual(latents.shape, (1, 3, 768))
        self.assertTrue(torch.equal(latents[:, :1, :], torch.ones((1, 1, 768))))
        self.assertTrue(torch.equal(latents[:, 1:, :], torch.full((1, 2, 768), 2.0)))

    def test_segment_progress_conditioning_changes_clip_feature(self):
        import torch

        from Script.stage1.segment_conditioning import add_progress_to_clip_feature

        clip_feature = torch.zeros((2, 512), dtype=torch.float32)
        conditioned = add_progress_to_clip_feature(
            clip_feature,
            mode="scalar",
            segment_idx=torch.tensor([0, 2]),
            num_segments=torch.tensor([3, 3]),
            segment_progress=torch.tensor([0.0, 1.0]),
            prefix_lengths=torch.tensor([0, 5]),
            context_size=10,
        )

        self.assertEqual(conditioned.shape, (2, 512))
        self.assertGreater(float(conditioned.abs().sum()), 0.0)
        self.assertAlmostEqual(float(conditioned[0, 0]), 0.0)
        self.assertAlmostEqual(float(conditioned[1, 0]), 1.0)

    def test_segmented_generation_can_match_training_progress_prefix_scale(self):
        import torch

        import Script.stage1.generate_long_motion as generate

        class FakeModel:
            def get_block_size(self):
                return 52

        clip_features = []

        def fake_text_encoder(text, model_name, max_length, device):
            return torch.zeros((1, max_length, 1024)), torch.zeros((1, max_length), dtype=torch.bool)

        def fake_with_prefix(
            model,
            clip_feature,
            bert_feature,
            bert_mask,
            max_length,
            prefix_latents,
            context_size,
            chunk_size,
            categorical,
            allow_early_stop,
            top_k=50,
            top_p=1.0,
            temperature=1.0,
            **kwargs,
        ):
            clip_features.append(clip_feature.clone())
            return torch.ones((1, max_length, 768), dtype=torch.float32)

        old_encode = generate.encode_text_with_t5
        old_with_prefix = generate.sample_latents_with_prefix
        generate.encode_text_with_t5 = fake_text_encoder
        generate.sample_latents_with_prefix = fake_with_prefix
        try:
            generate.sample_latents_segmented(
                model=FakeModel(),
                clip_feature=torch.zeros((1, 512)),
                text_segments=["walk", "turn"],
                text_encoder="t5",
                text_model="fake-t5",
                max_text_length=4,
                device="cpu",
                segment_length=30,
                segment_lengths=[30, 30],
                context_size=30,
                chunk_size=20,
                categorical=False,
                allow_early_stop=False,
                progress_context_size=51,
                progress_prefix_cap=25,
            )
        finally:
            generate.encode_text_with_t5 = old_encode
            generate.sample_latents_with_prefix = old_with_prefix

        self.assertEqual(len(clip_features), 2)
        self.assertAlmostEqual(float(clip_features[0][0, 3]), 0.0)
        self.assertAlmostEqual(float(clip_features[1][0, 3]), 25.0 / 51.0, places=6)

    def test_auto_generation_mode_selects_segmented_for_joined_text(self):
        from Script.stage1.generate_long_motion import resolve_generation_mode

        self.assertEqual(resolve_generation_mode("auto", "walk then turn", " then "), "segmented")
        self.assertEqual(resolve_generation_mode("auto", "walk forward", " then "), "rolling")
        self.assertEqual(resolve_generation_mode("rolling", "walk then turn", " then "), "rolling")
        self.assertEqual(resolve_generation_mode("segmented", "walk forward", " then "), "segmented")

    def test_segment_lengths_resolver_uses_max_length_when_segment_length_is_omitted(self):
        from Script.stage1.generate_long_motion import resolve_segment_lengths

        self.assertEqual(
            resolve_segment_lengths(
                segment_lengths_arg=None,
                segment_length_arg=None,
                max_length=10,
                expected_count=3,
            ),
            [4, 3, 3],
        )
        self.assertEqual(
            resolve_segment_lengths(
                segment_lengths_arg=None,
                segment_length_arg=5,
                max_length=10,
                expected_count=3,
            ),
            [5, 5, 5],
        )
        self.assertEqual(
            resolve_segment_lengths(
                segment_lengths_arg="2,3,4",
                segment_length_arg=5,
                max_length=10,
                expected_count=3,
            ),
            [2, 3, 4],
        )
        with self.assertRaises(ValueError):
            resolve_segment_lengths(
                segment_lengths_arg=None,
                segment_length_arg=None,
                max_length=2,
                expected_count=3,
            )


if __name__ == "__main__":
    unittest.main()
