import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path


def _row(label: str, accepted: bool = True):
    return {
        "label": label,
        "path": f"exports/{label}.bvh",
        "caption": f"caption {label}",
        "accepted": accepted,
        "reject_reasons": [] if accepted else ["depth0_unique<16"],
    }


class Stage1BVHQualitySplitTests(unittest.TestCase):
    def test_split_quality_summary_uses_only_accepted_rows_by_default(self):
        from Script.stage1.split_bvh_quality_summary import split_quality_summary

        payload = {
            "rows": [_row("a"), _row("b"), _row("c"), _row("d"), _row("bad", accepted=False)],
        }

        train, val = split_quality_summary(payload, seed=3, val_fraction=0.25, min_val=1)

        labels = {row["label"] for row in train + val}
        self.assertEqual(labels, {"a", "b", "c", "d"})
        self.assertEqual(len(val), 1)
        self.assertEqual(len(train), 3)
        self.assertEqual(train, sorted(train, key=lambda row: row["label"]))
        self.assertEqual(val, sorted(val, key=lambda row: row["label"]))

    def test_split_quality_summary_is_deterministic(self):
        from Script.stage1.split_bvh_quality_summary import split_quality_summary

        payload = {"rows": [_row(str(idx)) for idx in range(10)]}

        first = split_quality_summary(payload, seed=7, val_fraction=0.3)
        second = split_quality_summary(payload, seed=7, val_fraction=0.3)
        other = split_quality_summary(payload, seed=8, val_fraction=0.3)

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_cli_writes_train_and_val_quality_summaries(self):
        from Script.stage1 import split_bvh_quality_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "quality.json"
            train_output = tmp / "train_quality.json"
            val_output = tmp / "val_quality.json"
            source.write_text(
                json.dumps(
                    {
                        "metric_notes": {"scope": "test"},
                        "thresholds": {"min_frames": 120},
                        "counts": {"total": 6, "accepted": 5, "rejected": 1},
                        "rows": [_row(str(idx)) for idx in range(5)] + [_row("bad", accepted=False)],
                    }
                ),
                encoding="utf-8",
            )

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                split_bvh_quality_summary.main(
                    [
                        "--quality-summary",
                        str(source),
                        "--train-output",
                        str(train_output),
                        "--val-output",
                        str(val_output),
                        "--seed",
                        "11",
                        "--val-fraction",
                        "0.4",
                        "--quiet",
                    ]
                )
            compact = json.loads(stream.getvalue())
            train_payload = json.loads(train_output.read_text(encoding="utf-8"))
            val_payload = json.loads(val_output.read_text(encoding="utf-8"))

        self.assertEqual(compact["train_counts"]["accepted"], 3)
        self.assertEqual(compact["val_counts"]["accepted"], 2)
        self.assertEqual(train_payload["source_counts"], {"total": 6, "accepted": 5, "rejected": 1})
        self.assertEqual(train_payload["split"], "train")
        self.assertEqual(val_payload["split"], "val")
        self.assertEqual({row["accepted"] for row in train_payload["rows"] + val_payload["rows"]}, {True})


if __name__ == "__main__":
    unittest.main()
