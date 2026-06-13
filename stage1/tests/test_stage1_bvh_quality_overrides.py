import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path


def _row(label: str, accepted: bool):
    return {
        "label": label,
        "path": f"exports/{label}.bvh",
        "caption": f"caption {label}",
        "accepted": accepted,
        "reject_reasons": [] if accepted else ["p99_abs_z>8"],
    }


class Stage1BVHQualityOverrideTests(unittest.TestCase):
    def test_apply_quality_overrides_can_force_include_and_exclude(self):
        from Script.stage1.apply_bvh_quality_overrides import apply_quality_overrides

        payload = {
            "counts": {"total": 3, "accepted": 1, "rejected": 2},
            "accepted_paths": ["exports/good.bvh"],
            "rejected_labels": ["bad", "maybe"],
            "rows": [_row("good", True), _row("bad", False), _row("maybe", False)],
        }

        out = apply_quality_overrides(
            payload,
            include={"maybe": "mp4_audit_walk_ok"},
            exclude={"good": "mp4_audit_floor_motion"},
        )

        by_label = {row["label"]: row for row in out["rows"]}
        self.assertEqual(out["counts"], {"total": 3, "accepted": 1, "rejected": 2})
        self.assertFalse(by_label["good"]["accepted"])
        self.assertIn("manual_exclude:mp4_audit_floor_motion", by_label["good"]["reject_reasons"])
        self.assertTrue(by_label["maybe"]["accepted"])
        self.assertEqual(by_label["maybe"]["reject_reasons"], [])
        self.assertEqual(out["accepted_paths"], ["exports/maybe.bvh"])
        self.assertIn("good", out["rejected_labels"])
        self.assertEqual(by_label["good"]["manual_overrides"][0]["action"], "exclude")
        self.assertEqual(by_label["maybe"]["manual_overrides"][0]["action"], "include")

    def test_apply_quality_overrides_rejects_unknown_labels(self):
        from Script.stage1.apply_bvh_quality_overrides import apply_quality_overrides

        with self.assertRaises(ValueError):
            apply_quality_overrides({"rows": [_row("good", True)]}, include={"missing": "not_found"})

    def test_cli_writes_override_summary(self):
        from Script.stage1 import apply_bvh_quality_overrides

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "quality.json"
            output = tmp / "quality_v2.json"
            source.write_text(
                json.dumps(
                    {
                        "counts": {"total": 2, "accepted": 1, "rejected": 1},
                        "rows": [_row("good", True), _row("maybe", False)],
                    }
                ),
                encoding="utf-8",
            )

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                apply_bvh_quality_overrides.main(
                    [
                        "--quality-summary",
                        str(source),
                        "--include",
                        "maybe:mp4_audit_ok",
                        "--exclude",
                        "good:mp4_audit_bad",
                        "--output-json",
                        str(output),
                        "--quiet",
                    ]
                )
            compact = json.loads(stream.getvalue())
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(compact["counts"], {"total": 2, "accepted": 1, "rejected": 1})
        self.assertEqual(payload["manual_override_summary"]["include"], {"maybe": "mp4_audit_ok"})
        self.assertEqual(payload["manual_override_summary"]["exclude"], {"good": "mp4_audit_bad"})
        self.assertEqual(payload["source_summary"], str(source))


if __name__ == "__main__":
    unittest.main()
