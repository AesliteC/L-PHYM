import json
import tempfile
import unittest
from pathlib import Path


class Stage1PlotTrainCurvesTests(unittest.TestCase):
    def test_load_train_log_rejects_duplicate_epoch_by_default(self):
        from Script.stage1.plot_train_curves import load_train_log

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_log.jsonl"
            row = {"epoch": 0, "train": {"loss": 1.0}, "val": {"loss": 1.2}}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_train_log(path)

            rows = load_train_log(path, allow_duplicate_epochs=True)
            self.assertEqual(len(rows), 2)

    def test_summarize_curve_reports_best_and_last_metrics(self):
        from Script.stage1.plot_train_curves import summarize_curve

        rows = [
            {
                "epoch": 0,
                "train": {"loss": 2.0, "ce_loss": 1.9, "token_accuracy": 0.1},
                "val": {"loss": 1.5, "ce_loss": 1.4, "token_accuracy": 0.2},
            },
            {
                "epoch": 1,
                "train": {"loss": 1.0, "ce_loss": 0.9, "token_accuracy": 0.3},
                "val": {"loss": 1.2, "ce_loss": 1.1, "token_accuracy": 0.4},
            },
        ]

        summary = summarize_curve(rows)

        self.assertEqual(summary["best_val_epoch"], 1)
        self.assertAlmostEqual(summary["best_val_loss"], 1.2)
        self.assertAlmostEqual(summary["last_train_token_accuracy"], 0.3)
        self.assertAlmostEqual(summary["last_val_token_accuracy"], 0.4)


if __name__ == "__main__":
    unittest.main()
