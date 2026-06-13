import json
import tempfile
import unittest
from pathlib import Path


class Stage1ModelSuiteTests(unittest.TestCase):
    def test_default_prompt_writer_and_llm_response_map(self):
        from Script.stage1.run_stage1_model_suite import (
            DEFAULT_PROMPTS,
            load_llm_response_map,
            read_prompts,
            write_default_prompts,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            prompts = tmp / "prompts.tsv"
            write_default_prompts(prompts)
            self.assertEqual(read_prompts(prompts), list(DEFAULT_PROMPTS))

            response = tmp / "response.json"
            response.write_text('{"tokens":[]}', encoding="utf-8")
            mapping = tmp / "map.json"
            mapping.write_text(json.dumps({"walk_turn_wave": response.name}), encoding="utf-8")
            loaded = load_llm_response_map(str(mapping))

        self.assertEqual(loaded["walk_turn_wave"], response)

    def test_model_suite_skip_generation_collects_metrics_for_existing_bvhs(self):
        import Script.stage1.run_stage1_model_suite as suite

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            prompts = tmp / "prompts.tsv"
            prompts.write_text("walk_turn\twalk then turn\n", encoding="utf-8")
            motion_dataset = tmp / "simple_motion_data.h5"
            motion_dataset.write_text("placeholder", encoding="utf-8")
            bvh_dir = tmp / "bvh"
            bvh_dir.mkdir()
            for model in ("baseline_top_p", "finetuned_top_p"):
                (bvh_dir / f"walk_turn__{model}.bvh").write_text(
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
                    "Frames: 3\n"
                    "Frame Time: 0.008333\n"
                    "0 0 0 0 0 0\n"
                    "1 0 0 1 0 0\n"
                    "2 0 0 2 0 0\n",
                    encoding="utf-8",
                )

            suite.main(
                [
                    "--run-id",
                    "unit",
                    "--prompts",
                    str(prompts),
                    "--finetuned-checkpoint",
                    "finetuned.pth",
                    "--suite-dir",
                    str(tmp / "suite"),
                    "--bvh-dir",
                    str(bvh_dir),
                    "--motion-dataset",
                    str(motion_dataset),
                    "--skip-generation",
                    "--skip-backup",
                ]
            )

            summary = json.loads((tmp / "suite" / "suite_summary.json").read_text(encoding="utf-8"))
            metrics = json.loads((tmp / "suite" / "summary_metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["run_id"], "unit")
        self.assertEqual(len(summary["generated"]), 2)
        self.assertIn(str(motion_dataset), summary["generated"][0]["command"])
        self.assertEqual(summary["config"]["motion_dataset"], str(motion_dataset))
        self.assertIn("baseline_top_p", summary["model_averages"])
        self.assertEqual(len(metrics["rows"]), 2)

    def test_model_suite_wires_retrieval_backup_commands(self):
        import Script.stage1.run_stage1_model_suite as suite

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            prompts = tmp / "prompts.tsv"
            prompts.write_text("walk_turn\twalk then turn\n", encoding="utf-8")
            bank = tmp / "bank.jsonl"
            bank.write_text(
                json.dumps(
                    {
                        "example_id": "walk",
                        "caption": "walk",
                        "indices": [[1, 2, 3, 4], [5, 6, 7, 8]],
                        "source": "unit",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            motion_dataset = tmp / "simple_motion_data.h5"
            motion_dataset.write_text("placeholder", encoding="utf-8")
            bvh_dir = tmp / "bvh"
            bvh_dir.mkdir()
            (bvh_dir / "walk_turn__backup_retrieval.bvh").write_text(
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
                "1 0 0 1 0 0\n",
                encoding="utf-8",
            )
            commands = []

            def fake_run_command(command, log_path=None):
                commands.append(command)
                if log_path is not None:
                    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(log_path).write_text("ok\n", encoding="utf-8")

            old_run_command = suite.run_command
            suite.run_command = fake_run_command
            try:
                suite.main(
                    [
                        "--run-id",
                        "unit_backup",
                        "--prompts",
                        str(prompts),
                        "--finetuned-checkpoint",
                        "finetuned.pth",
                        "--suite-dir",
                        str(tmp / "suite"),
                        "--bvh-dir",
                        str(bvh_dir),
                        "--example-bank",
                        str(bank),
                        "--motion-dataset",
                        str(motion_dataset),
                        "--skip-gpt",
                        "--skip-generation",
                    ]
                )
            finally:
                suite.run_command = old_run_command

            summary = json.loads((tmp / "suite" / "suite_summary.json").read_text(encoding="utf-8"))

        flattened = [" ".join(command) for command in commands]
        self.assertTrue(any("build-prompt" in command for command in flattened))
        self.assertTrue(any("retrieval-plan" in command for command in flattened))
        self.assertTrue(any("--trim-repeat-runs" in command for command in flattened))
        self.assertEqual(summary["generated"][0]["model"], "backup_retrieval")
        self.assertIn(str(motion_dataset), summary["generated"][0]["command"])
        self.assertIn("backup_retrieval", summary["model_averages"])
        self.assertEqual(summary["config"]["motion_dataset"], str(motion_dataset))

    def test_skip_backup_does_not_export_bank(self):
        import Script.stage1.run_stage1_model_suite as suite

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            prompts = tmp / "prompts.tsv"
            prompts.write_text("walk_turn\twalk then turn\n", encoding="utf-8")
            commands = []

            def fake_run_command(command, log_path=None):
                commands.append(command)

            old_run_command = suite.run_command
            suite.run_command = fake_run_command
            try:
                suite.main(
                    [
                        "--run-id",
                        "unit_skip_backup",
                        "--prompts",
                        str(prompts),
                        "--finetuned-checkpoint",
                        "finetuned.pth",
                        "--suite-dir",
                        str(tmp / "suite"),
                        "--bvh-dir",
                        str(tmp / "bvh"),
                        "--backup-cache",
                        "cache.pt",
                        "--skip-gpt",
                        "--skip-backup",
                    ]
                )
            finally:
                suite.run_command = old_run_command

        self.assertEqual(commands, [])

    def test_model_suite_wires_llm_response_without_retrieval_bank(self):
        import Script.stage1.run_stage1_model_suite as suite

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            prompts = tmp / "prompts.tsv"
            prompts.write_text("walk_turn\twalk then turn\n", encoding="utf-8")
            response = tmp / "response.json"
            response.write_text('{"tokens": [[1, 2, 3, 4], [5, 6, 7, 8]]}', encoding="utf-8")
            mapping = tmp / "responses.json"
            mapping.write_text(json.dumps({"walk_turn": str(response)}), encoding="utf-8")
            bvh_dir = tmp / "bvh"
            bvh_dir.mkdir()
            (bvh_dir / "walk_turn__backup_llm.bvh").write_text(
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
                "1 0 0 1 0 0\n",
                encoding="utf-8",
            )
            commands = []

            def fake_run_command(command, log_path=None):
                commands.append(command)
                if log_path is not None:
                    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(log_path).write_text("ok\n", encoding="utf-8")

            old_run_command = suite.run_command
            suite.run_command = fake_run_command
            try:
                suite.main(
                    [
                        "--run-id",
                        "unit_llm",
                        "--prompts",
                        str(prompts),
                        "--finetuned-checkpoint",
                        "finetuned.pth",
                        "--suite-dir",
                        str(tmp / "suite"),
                        "--bvh-dir",
                        str(bvh_dir),
                        "--llm-response-map",
                        str(mapping),
                        "--skip-gpt",
                        "--skip-generation",
                    ]
                )
            finally:
                suite.run_command = old_run_command

            summary = json.loads((tmp / "suite" / "suite_summary.json").read_text(encoding="utf-8"))

        flattened = [" ".join(command) for command in commands]
        self.assertTrue(any("validate" in command for command in flattened))
        self.assertEqual(summary["generated"][0]["model"], "backup_llm")
        self.assertIn("backup_llm", summary["model_averages"])


if __name__ == "__main__":
    unittest.main()
