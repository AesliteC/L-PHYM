import json
import tempfile
import unittest
from pathlib import Path

import torch


class Stage1LLMTokenPlanningTests(unittest.TestCase):
    def test_direct_script_execution_prefers_own_repo_root(self):
        import Script.stage1.llm_token_planning as llm_token_planning

        expected_root = Path(llm_token_planning.__file__).resolve().parents[2]

        self.assertEqual(Path(llm_token_planning.sys.path[0]).resolve(), expected_root)

    def test_export_example_bank_trims_padding_and_target_mask(self):
        from Script.stage1.llm_token_planning import export_example_bank_from_cache, load_example_bank

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cache_path = tmp / "cache.pt"
            output_path = tmp / "bank.jsonl"
            torch.save(
                {
                    "indices": torch.tensor(
                        [
                            [
                                [1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 10, 11, 12],
                                [513, 513, 513, 513],
                            ]
                        ],
                        dtype=torch.long,
                    ),
                    "target_masks": torch.tensor([[True, False, True, False]], dtype=torch.bool),
                    "captions": ["a person walks forward"],
                    "sequence_ids": ["seq0"],
                    "window_ranges": [[0, 4]],
                },
                cache_path,
            )

            summary = export_example_bank_from_cache(
                cache_path,
                output_path,
                max_examples=10,
                max_tokens_per_example=4,
                min_tokens_per_example=2,
            )
            examples = load_example_bank(output_path)

            self.assertEqual(summary["examples_written"], 1)
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].indices, [[1, 2, 3, 4], [9, 10, 11, 12]])
            self.assertEqual(examples[0].caption, "a person walks forward")

    def test_retrieval_prefers_caption_overlap(self):
        from Script.stage1.llm_token_planning import MotionTokenExample, retrieve_examples

        examples = [
            MotionTokenExample("walk", "a person walks forward", [[1, 2, 3, 4]], "bank"),
            MotionTokenExample("kick", "a person kicks with the right foot", [[5, 6, 7, 8]], "bank"),
            MotionTokenExample("dance", "a person dances in place", [[9, 10, 11, 12]], "bank"),
        ]

        rows = retrieve_examples(examples, "kick with right foot", top_k=2, min_tokens=1)

        self.assertEqual(rows[0][0].example_id, "kick")
        self.assertGreater(rows[0][1], rows[1][1])

    def test_build_prompt_contains_schema_segments_and_examples(self):
        from Script.stage1.llm_token_planning import MotionTokenExample, build_llm_prompt

        example = MotionTokenExample("walk", "a person walks forward", [[1, 2, 3, 4], [5, 6, 7, 8]], "bank")
        prompt = build_llm_prompt(
            "a person walks forward then kicks",
            [("a person walks forward", [(example, 1.0)])],
            max_tokens_per_example=1,
        )

        self.assertIn('"tokens":[[d0,d1,d2,d3], ...]', prompt)
        self.assertIn("Segment 1: a person walks forward", prompt)
        self.assertIn("Caption: a person walks forward", prompt)
        self.assertIn("[[1,2,3,4]]", prompt)

    def test_validate_llm_tokens_extracts_json_and_repairs_range(self):
        from Script.stage1.llm_token_planning import parse_llm_tokens, validate_token_sequence

        raw = parse_llm_tokens('Here is JSON:\n{"tokens": [[1, 2, 3, 4], [600, -2, 5, 6]]}')
        tokens, validation = validate_token_sequence(raw, repair=True, min_length=2)

        self.assertEqual(tokens, [[1, 2, 3, 4], [511, 0, 5, 6]])
        self.assertTrue(validation["ok"])
        self.assertEqual(validation["clipped_values"], 2)

    def test_validate_llm_tokens_reports_repeated_tuple_runs(self):
        from Script.stage1.llm_token_planning import validate_token_sequence

        raw = [[1, 2, 3, 4]] * 4
        _tokens, validation = validate_token_sequence(raw, max_consecutive_repeat=3)

        self.assertFalse(validation["ok"])
        self.assertEqual(validation["repeat_violations"][0]["length"], 4)

    def test_validate_llm_tokens_can_trim_repeated_tuple_runs(self):
        from Script.stage1.llm_token_planning import validate_token_sequence

        raw = [[1, 2, 3, 4]] * 5 + [[5, 6, 7, 8]]
        tokens, validation = validate_token_sequence(
            raw,
            max_consecutive_repeat=3,
            trim_repeat_runs=True,
        )

        self.assertTrue(validation["ok"])
        self.assertEqual(tokens, [[1, 2, 3, 4]] * 3 + [[5, 6, 7, 8]])
        self.assertEqual(validation["repeat_repairs"], 2)

    def test_retrieval_only_plan_and_cli_outputs_tokens(self):
        from Script.stage1 import llm_token_planning

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bank = tmp / "bank.jsonl"
            bank.write_text(
                json.dumps(
                    {
                        "example_id": "walk",
                        "caption": "a person walks forward",
                        "indices": [[1, 2, 3, 4], [5, 6, 7, 8]],
                        "source": "unit",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "example_id": "kick",
                        "caption": "a person kicks with the right foot",
                        "indices": [[9, 10, 11, 12], [13, 14, 15, 16]],
                        "source": "unit",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tokens_path = tmp / "tokens.json"
            validation_path = tmp / "validation.json"

            llm_token_planning.main(
                [
                    "retrieval-plan",
                    "--bank",
                    str(bank),
                    "--text",
                    "a person walks forward then kicks with the right foot",
                    "--segment-token-count",
                    "3",
                    "--output-tokens",
                    str(tokens_path),
                    "--validation-json",
                    str(validation_path),
                ]
            )

            payload = json.loads(tokens_path.read_text(encoding="utf-8"))
            validation = json.loads(validation_path.read_text(encoding="utf-8"))

            self.assertEqual(len(payload["tokens"]), 6)
            self.assertEqual(payload["tokens"][:3], [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]])
            self.assertTrue(validation["ok"])


if __name__ == "__main__":
    unittest.main()
