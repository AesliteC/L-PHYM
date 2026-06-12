import tempfile
import unittest
from pathlib import Path


class Stage1TextGPTComparisonTests(unittest.TestCase):
    def test_read_prompts_parses_tsv_and_skips_comments(self):
        from Script.stage1.run_text_gpt_comparison import read_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.tsv"
            path.write_text(
                "# comment\n"
                "\n"
                "walk_turn\ta person walks then turns\n"
                "jump\t a person jumps \n",
                encoding="utf-8",
            )

            prompts = read_prompts(path)

            self.assertEqual(
                prompts,
                [
                    ("walk_turn", "a person walks then turns"),
                    ("jump", "a person jumps"),
                ],
            )

    def test_read_prompts_rejects_non_tsv_line(self):
        from Script.stage1.run_text_gpt_comparison import read_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.tsv"
            path.write_text("missing_tab\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                read_prompts(path)


if __name__ == "__main__":
    unittest.main()
