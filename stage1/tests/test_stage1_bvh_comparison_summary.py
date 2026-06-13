import unittest


class Stage1BVHComparisonSummaryTests(unittest.TestCase):
    def test_summarize_rows_groups_models_and_prompt_deltas(self):
        from Script.stage1.summarize_bvh_comparison import summarize_rows

        rows = [
            {
                "label": "walk__baseline_top_p",
                "frames": 100,
                "duration_sec": 1.0,
                "pose_velocity_mean": 2.0,
                "pose_variance_mean": 3.0,
                "lag_20_repeat_fraction_0.995": 0.1,
            },
            {
                "label": "walk__finetuned_top_p",
                "frames": 150,
                "duration_sec": 1.5,
                "pose_velocity_mean": 4.0,
                "pose_variance_mean": 5.0,
                "lag_20_repeat_fraction_0.995": 0.2,
            },
            {
                "label": "turn__baseline_top_p",
                "frames": 200,
                "duration_sec": 2.0,
                "pose_velocity_mean": 6.0,
                "pose_variance_mean": 7.0,
                "lag_20_repeat_fraction_0.995": 0.3,
            },
            {
                "label": "turn__finetuned_top_p",
                "frames": 300,
                "duration_sec": 3.0,
                "pose_velocity_mean": 8.0,
                "pose_variance_mean": 9.0,
                "lag_20_repeat_fraction_0.995": 0.4,
            },
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary["model_summary"]["baseline_top_p"]["count"], 2)
        self.assertEqual(summary["model_summary"]["finetuned_top_p"]["count"], 2)
        self.assertAlmostEqual(summary["model_summary"]["baseline_top_p"]["avg_frames"], 150.0)
        self.assertAlmostEqual(summary["model_summary"]["finetuned_top_p"]["avg_frames"], 225.0)
        prompt_rows = summary["paired_comparison"]["prompts"]
        self.assertEqual(len(prompt_rows), 2)
        self.assertEqual(prompt_rows[0]["prompt"], "turn")
        self.assertAlmostEqual(prompt_rows[0]["delta_frames"], 100.0)
        self.assertEqual(prompt_rows[1]["prompt"], "walk")
        self.assertAlmostEqual(prompt_rows[1]["delta_duration_sec"], 0.5)

    def test_summary_to_markdown_contains_model_average_table(self):
        from Script.stage1.summarize_bvh_comparison import summarize_rows, summary_to_markdown

        summary = summarize_rows(
            [
                {"label": "walk__baseline_top_p", "frames": 100, "duration_sec": 1.0},
                {"label": "walk__finetuned_top_p", "frames": 120, "duration_sec": 1.2},
            ],
            metrics=["frames", "duration_sec"],
        )

        markdown = summary_to_markdown(summary)

        self.assertIn("BVH Comparison Summary", markdown)
        self.assertIn("baseline_top_p", markdown)
        self.assertIn("finetuned_top_p", markdown)
        self.assertIn("delta_frames", markdown)


if __name__ == "__main__":
    unittest.main()
