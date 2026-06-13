import unittest
from pathlib import Path

from Script.stage1.humanml3d import build_long_horizon_manifest, load_humanml3d_catalog


<<<<<<< HEAD
class HumanML3DStage1Tests(unittest.TestCase):
    def test_catalog_reads_canonical_split_counts(self):
        root = Path(__file__).resolve().parents[2] / "HumanML3D" / "HumanML3D"
=======
def _humanml_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "HumanML3D" / "HumanML3D"
        if (candidate / "all.txt").exists():
            return candidate
    fallback = Path("/home/chenjie/cc/robotics/HumanML3D/HumanML3D")
    if (fallback / "all.txt").exists():
        return fallback
    raise FileNotFoundError("HumanML3D/HumanML3D/all.txt was not found")


class HumanML3DStage1Tests(unittest.TestCase):
    def test_catalog_reads_canonical_split_counts(self):
        root = _humanml_root()
>>>>>>> origin/main
        catalog = load_humanml3d_catalog(root)

        self.assertEqual(len(catalog.all_ids), 29228)
        self.assertEqual(len(catalog.split_ids["train"]), 23384)
        self.assertEqual(len(catalog.split_ids["val"]), 1460)
        self.assertEqual(len(catalog.split_ids["test"]), 4384)
        self.assertEqual(len(catalog.split_ids["train_val"]), 24844)

        sample = catalog.by_id["000001"]
        self.assertEqual(sample.sample_id, "000001")
        self.assertEqual(sample.joints_path.name, "000001.npy")
        self.assertEqual(sample.vecs_path.name, "000001.npy")
        self.assertEqual(sample.text_path.name, "000001.txt")
        self.assertIsNotNone(sample.source_path)

    def test_build_long_horizon_manifest_is_deterministic_and_structured(self):
<<<<<<< HEAD
        root = Path(__file__).resolve().parents[2] / "HumanML3D" / "HumanML3D"
=======
        root = _humanml_root()
>>>>>>> origin/main
        catalog = load_humanml3d_catalog(root)
        manifest = build_long_horizon_manifest(
            catalog,
            split="train",
            num_sequences=2,
            min_clips=2,
            max_clips=2,
            seed=7,
        )

        self.assertEqual(len(manifest), 2)
        first = manifest[0]
        self.assertEqual(first["split"], "train")
        self.assertEqual(len(first["sample_ids"]), 2)
        self.assertEqual(len(first["clip_captions"]), 2)
        self.assertEqual(len(first["frame_lengths"]), 2)
        self.assertIn("then", first["caption"])

    def test_manifest_can_be_written_to_jsonl(self):
        from Script.stage1.humanml3d import write_manifest_jsonl
        import tempfile
        import json

<<<<<<< HEAD
        root = Path(__file__).resolve().parents[2] / "HumanML3D" / "HumanML3D"
=======
        root = _humanml_root()
>>>>>>> origin/main
        catalog = load_humanml3d_catalog(root)
        manifest = build_long_horizon_manifest(
            catalog,
            split="train",
            num_sequences=1,
            min_clips=2,
            max_clips=2,
            seed=11,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "manifest.jsonl"
            write_manifest_jsonl(manifest, out)
            rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sequence_id"], manifest[0]["sequence_id"])


if __name__ == "__main__":
    unittest.main()
