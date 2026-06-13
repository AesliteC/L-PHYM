import tempfile
import unittest
from pathlib import Path

import torch


class Stage1CachePromptExportTests(unittest.TestCase):
    def test_scale_segment_lengths_preserves_total_and_positive_values(self):
        from Script.stage1.export_cache_prompt_tsv import scale_segment_lengths

        self.assertEqual(scale_segment_lengths([22, 32, 23], total_length=75), (22, 31, 22))
        self.assertEqual(scale_segment_lengths([50, 49, 44, 24], total_length=75), (22, 22, 20, 11))
        self.assertEqual(sum(scale_segment_lengths([1, 1, 8], total_length=7)), 7)
        self.assertTrue(all(item > 0 for item in scale_segment_lengths([1, 1, 8], total_length=7)))

    def test_prompts_from_cache_deduplicates_segment_windows(self):
        from Script.stage1.export_cache_prompt_tsv import prompts_from_cache

        cache = {
            "sequence_ids": ["0000_train_000001", "0000_train_000001", "0000_train_000001"],
            "captions": ["walk forward", "turn then step", "turn then step"],
            "segment_idxs": torch.tensor([0, 1, 1]),
            "num_segments": torch.tensor([2, 2, 2]),
            "segment_ranges": [(0, 20), (20, 50), (20, 50)],
        }

        prompts, rows = prompts_from_cache(cache, total_length=25)

        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].name, "train_000001")
        self.assertEqual(prompts[0].segments, ("walk forward", "turn then step"))
        self.assertEqual(prompts[0].segment_lengths, (10, 15))
        self.assertEqual(rows[0]["raw_lengths"], [20, 30])

    def test_cli_writes_four_column_prompt_tsv(self):
        from Script.stage1.export_cache_prompt_tsv import main
        from Script.stage1.run_text_gpt_comparison import read_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cache_path = tmp / "cache.pt"
            output = tmp / "prompts.tsv"
            summary = tmp / "summary.json"
            torch.save(
                {
                    "sequence_ids": ["0000_train_000001", "0000_train_000001"],
                    "captions": ["walk forward", "turn then step"],
                    "segment_idxs": torch.tensor([0, 1]),
                    "num_segments": torch.tensor([2, 2]),
                    "segment_ranges": [(0, 20), (20, 50)],
                },
                cache_path,
            )

            main(
                [
                    "--cache",
                    str(cache_path),
                    "--output",
                    str(output),
                    "--summary",
                    str(summary),
                    "--total-length",
                    "25",
                ]
            )

            prompts = read_prompts(output)
            summary_exists = summary.exists()

        self.assertEqual(prompts[0].name, "train_000001")
        self.assertEqual(prompts[0].segments, ("walk forward", "turn then step"))
        self.assertEqual(prompts[0].segment_lengths, (10, 15))
        self.assertTrue(summary_exists)


if __name__ == "__main__":
    unittest.main()
