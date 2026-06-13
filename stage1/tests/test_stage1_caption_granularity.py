import tempfile
import unittest
from pathlib import Path

import numpy as np


def _write_caption_fixture(base: Path) -> Path:
    root = base / "HumanML3D"
    for dirname in ("new_joints", "new_joint_vecs", "texts"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    ids = ["000001", "000002"]
    for split in ("all", "train", "val", "test", "train_val"):
        (root / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    for sample_id in ids:
        np.save(root / "new_joints" / f"{sample_id}.npy", np.zeros((4, 22, 3), dtype=np.float32))
        np.save(root / "new_joint_vecs" / f"{sample_id}.npy", np.zeros((4, 263), dtype=np.float32))
    (root / "texts" / "000001.txt").write_text(
        "a person walks then turns#a/DET person/NOUN walk/VERB then/ADV turn/VERB#0.0#0.0\n"
        "a person walks#a/DET person/NOUN walk/VERB#0.0#0.0\n",
        encoding="utf-8",
    )
    (root / "texts" / "000002.txt").write_text(
        "a person jumps while waving#a/DET person/NOUN jump/VERB while/SCONJ wave/VERB#0.0#0.0\n",
        encoding="utf-8",
    )
    return root


class Stage1CaptionGranularityTests(unittest.TestCase):
    def test_summarize_caption_granularity_reports_atomic_keep_rates(self):
        from Script.stage1.diagnose_humanml3d_caption_granularity import summarize_caption_granularity

        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_caption_fixture(Path(tmpdir))
            summary = summarize_caption_granularity(root, splits=("train",), examples_per_bucket=2)

        train = summary["splits"]["train"]
        self.assertEqual(train["samples"], 2)
        self.assertEqual(train["mode_counts"]["none"], 2)
        self.assertEqual(train["mode_counts"]["prefer_atomic"], 2)
        self.assertEqual(train["mode_counts"]["atomic"], 1)
        self.assertEqual(train["first_caption_non_atomic"], 2)
        self.assertEqual(train["prefer_atomic_non_atomic"], 1)
        self.assertEqual(len(train["examples"]["rejected_atomic"]), 1)


if __name__ == "__main__":
    unittest.main()
