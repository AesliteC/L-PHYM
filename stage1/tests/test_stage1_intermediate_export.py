from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class Stage1IntermediateExportTests(unittest.TestCase):
    def test_reshape_sample_indices_trims_extra_gpt_step(self):
        from Script.stage1.intermediate_motion_format import reshape_sample_indices

        raw = np.arange(20, dtype=np.int64).reshape(20, 1)
        indices = reshape_sample_indices(raw, latent_length=4, rvq_depth=4)

        self.assertEqual(indices.shape, (4, 4))
        np.testing.assert_array_equal(indices[0], np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(indices[-1], np.array([12, 13, 14, 15]))

    def test_write_and_validate_intermediate_npz(self):
        from Script.stage1.intermediate_motion_format import (
            FORMAT_VERSION,
            load_metadata,
            validate_intermediate_npz,
            write_format_markdown,
            write_intermediate_npz,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.npz"
            doc_path = Path(tmpdir) / "FORMAT.md"
            latent = np.ones((3, 768), dtype=np.float32)
            dynamic = np.ones((12, 256), dtype=np.float32) * 2.0
            indices = np.arange(12, dtype=np.int64).reshape(3, 4)
            write_intermediate_npz(
                path,
                motion_latent=latent,
                dynamic_control=dynamic,
                rvq_indices=indices,
                metadata={
                    "sample_id": "unit_test",
                    "prompt": "walk forward",
                    "checkpoint": "text_generation_GPT.pth",
                },
            )

            summary = validate_intermediate_npz(path)
            metadata = load_metadata(path)

            self.assertEqual(summary["motion_latent_shape"], [3, 768])
            self.assertEqual(summary["dynamic_control_shape"], [12, 256])
            self.assertEqual(summary["rvq_indices_shape"], [3, 4])
            self.assertEqual(metadata["format_version"], FORMAT_VERSION)
            self.assertEqual(metadata["prompt"], "walk forward")
            self.assertEqual(metadata["latent_dim"], 768)

            with np.load(path, allow_pickle=False) as data:
                self.assertIn("metadata_json", data.files)
                self.assertNotIn("poses", data.files)
                self.assertNotIn("trans", data.files)

            write_format_markdown(doc_path)
            text = doc_path.read_text(encoding="utf-8")
            self.assertIn("motion_latent", text)
            self.assertIn("dynamic_control", text)
            self.assertIn(FORMAT_VERSION, text)

    def test_validation_rejects_wrong_latent_dim(self):
        from Script.stage1.intermediate_motion_format import write_intermediate_npz

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.npz"
            with self.assertRaises(ValueError):
                write_intermediate_npz(
                    path,
                    motion_latent=np.zeros((2, 767), dtype=np.float32),
                    dynamic_control=np.zeros((8, 256), dtype=np.float32),
                    rvq_indices=np.zeros((2, 4), dtype=np.int64),
                    metadata={"sample_id": "bad", "prompt": "bad"},
                )

    def test_exporter_skip_model_writes_package_manifest_and_format_doc(self):
        from Script.stage1.export_baseline_intermediate import main
        from Script.stage1.intermediate_motion_format import validate_intermediate_npz

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "package"
            main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--skip-model",
                    "--prompt",
                    "Walk forward.",
                    "--prompt",
                    "Turn left and wave.",
                    "--max-length",
                    "3",
                ]
            )

            manifest = output_dir / "manifest.jsonl"
            format_doc = output_dir / "MOCONVQ_INTERMEDIATE_FORMAT.md"
            samples = sorted((output_dir / "samples").glob("*.npz"))

            self.assertTrue(manifest.exists())
            self.assertTrue(format_doc.exists())
            self.assertEqual(len(samples), 2)
            self.assertIn("motion_latent", format_doc.read_text(encoding="utf-8"))

            rows = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            summary = validate_intermediate_npz(samples[0])
            self.assertEqual(summary["motion_latent_shape"], [3, 768])
            self.assertEqual(summary["dynamic_control_shape"], [12, 256])


if __name__ == "__main__":
    unittest.main()
