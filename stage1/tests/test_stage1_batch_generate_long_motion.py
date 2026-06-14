import unittest


class Stage1BatchGenerateLongMotionTests(unittest.TestCase):
    def test_collect_texts_uses_explicit_segments_for_segmented_prompts(self):
        from Script.stage1.batch_generate_long_motion import collect_texts_for_encoding
        from Script.stage1.run_text_gpt_comparison import PromptRecord

        prompts = [
            PromptRecord(
                "p0",
                "walk then wave",
                segments=("walk then pause", "wave"),
                segment_lengths=(25, 25),
            )
        ]

        texts = collect_texts_for_encoding(prompts, generation_mode="segmented", segment_joiner=" then ")

        self.assertEqual(texts, ["walk then pause", "wave"])

    def test_collect_texts_uses_whole_text_for_rolling_prompts(self):
        from Script.stage1.batch_generate_long_motion import collect_texts_for_encoding
        from Script.stage1.run_text_gpt_comparison import PromptRecord

        prompts = [PromptRecord("p0", "walk then wave", segments=("walk", "wave"))]

        texts = collect_texts_for_encoding(prompts, generation_mode="rolling", segment_joiner=" then ")

        self.assertEqual(texts, ["walk then wave"])


if __name__ == "__main__":
    unittest.main()
