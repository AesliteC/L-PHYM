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

    def test_comparison_forwards_progress_scale_arguments_to_generator(self):
        import json

        import Script.stage1.run_text_gpt_comparison as comparison

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            prompts = tmp / "prompts.tsv"
            prompts.write_text("walk_turn\twalk then turn\n", encoding="utf-8")
            bvh_dir = tmp / "bvh"
            video_dir = tmp / "video"
            commands = []

            def fake_run_command(command, log_path=None):
                commands.append(command)
                if log_path is not None:
                    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(log_path).write_text("ok\n", encoding="utf-8")
                if "--output-bvh" in command:
                    output = Path(command[command.index("--output-bvh") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(
                        "HIERARCHY\n"
                        "ROOT Hips\n"
                        "{\n"
                        "  OFFSET 0 0 0\n"
                        "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
                        "  End Site\n"
                        "  {\n"
                        "    OFFSET 0 1 0\n"
                        "  }\n"
                        "}\n"
                        "MOTION\n"
                        "Frames: 2\n"
                        "Frame Time: 0.008333\n"
                        "0 0 0 0 0 0\n"
                        "0 0 0 0 0 0\n",
                        encoding="utf-8",
                    )

            def fake_count_bvh_frames(path):
                return 2

            def fake_evaluate_bvh_files(inputs, sample_stride, lags, expected_min_frames):
                return {"rows": []}

            old_run_command = comparison.run_command
            old_count_bvh_frames = comparison.count_bvh_frames
            old_evaluate = comparison.evaluate_bvh_files
            comparison.run_command = fake_run_command
            comparison.count_bvh_frames = fake_count_bvh_frames
            comparison.evaluate_bvh_files = fake_evaluate_bvh_files
            try:
                comparison.main(
                    [
                        "--run-id",
                        "test_run",
                        "--prompts",
                        str(prompts),
                        "--finetuned-checkpoint",
                        "finetuned.pth",
                        "--bvh-dir",
                        str(bvh_dir),
                        "--video-dir",
                        str(video_dir),
                        "--skip-render",
                        "--progress-context-size",
                        "51",
                        "--progress-prefix-cap",
                        "25",
                    ]
                )
            finally:
                comparison.run_command = old_run_command
                comparison.count_bvh_frames = old_count_bvh_frames
                comparison.evaluate_bvh_files = old_evaluate

            generation_commands = [
                command
                for command in commands
                if any("generate_long_motion.py" in str(part) for part in command)
            ]
            self.assertEqual(len(generation_commands), 2)
            for command in generation_commands:
                self.assertIn("--progress-context-size", command)
                self.assertEqual(command[command.index("--progress-context-size") + 1], "51")
                self.assertIn("--progress-prefix-cap", command)
                self.assertEqual(command[command.index("--progress-prefix-cap") + 1], "25")
            summary = json.loads((video_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["sampling"]["progress_context_size"], 51)
            self.assertEqual(summary["sampling"]["progress_prefix_cap"], 25)


if __name__ == "__main__":
    unittest.main()
