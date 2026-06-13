from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import random


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_count(total: int, val_fraction: float, min_val: int, max_val: int | None) -> int:
    if total <= 1 or val_fraction <= 0.0:
        return 0
    count = int(round(total * val_fraction))
    count = max(min_val, count)
    if max_val is not None:
        count = min(max_val, count)
    return min(max(count, 0), total - 1)


def _subset_payload(
    source: dict[str, object],
    rows: list[dict[str, object]],
    *,
    split: str,
    seed: int,
    val_fraction: float,
    source_path: Path,
) -> dict[str, object]:
    return {
        "metric_notes": source.get("metric_notes", {}),
        "thresholds": source.get("thresholds", {}),
        "source_summary": str(source_path),
        "split": split,
        "split_config": {
            "seed": seed,
            "val_fraction": val_fraction,
        },
        "source_counts": source.get("counts", {}),
        "counts": {
            "total": len(rows),
            "accepted": sum(1 for row in rows if row.get("accepted")),
            "rejected": sum(1 for row in rows if not row.get("accepted")),
        },
        "accepted_paths": [str(row["path"]) for row in rows if row.get("accepted")],
        "rejected_labels": [str(row.get("label", "")) for row in rows if not row.get("accepted")],
        "rows": rows,
    }


def split_quality_summary(
    payload: dict[str, object],
    *,
    seed: int = 13,
    val_fraction: float = 0.2,
    min_val: int = 1,
    max_val: int | None = None,
    accepted_only: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = [dict(row) for row in payload.get("rows", [])]  # type: ignore[arg-type]
    if accepted_only:
        rows = [row for row in rows if row.get("accepted")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    val_count = _split_count(len(rows), val_fraction=val_fraction, min_val=min_val, max_val=max_val)
    val_rows = sorted(rows[:val_count], key=lambda row: str(row.get("label", "")))
    train_rows = sorted(rows[val_count:], key=lambda row: str(row.get("label", "")))
    return train_rows, val_rows


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-summary", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-val", type=int, default=1)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Print only split counts and output paths.")
    args = parser.parse_args(argv)

    source_path = Path(args.quality_summary)
    source = _load_json(source_path)
    train_rows, val_rows = split_quality_summary(
        source,
        seed=args.seed,
        val_fraction=args.val_fraction,
        min_val=args.min_val,
        max_val=args.max_val,
        accepted_only=not args.include_rejected,
    )
    train_payload = _subset_payload(
        source,
        train_rows,
        split="train",
        seed=args.seed,
        val_fraction=args.val_fraction,
        source_path=source_path,
    )
    val_payload = _subset_payload(
        source,
        val_rows,
        split="val",
        seed=args.seed,
        val_fraction=args.val_fraction,
        source_path=source_path,
    )

    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.write_text(json.dumps(train_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    val_output.write_text(json.dumps(val_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    compact = {
        "train_output": str(train_output),
        "val_output": str(val_output),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "accepted_only": not args.include_rejected,
        "train_counts": train_payload["counts"],
        "val_counts": val_payload["counts"],
    }
    print(json.dumps(compact if args.quiet else {"train": train_payload, "val": val_payload}, indent=2))


if __name__ == "__main__":
    main()
