from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json

from Script.stage1.humanml3d import load_humanml3d_catalog
from Script.stage1.synthesize_long_humanml3d import (
    CAPTION_FILTER_MODE_CHOICES,
    _choose_caption,
    caption_complexity_score,
    caption_word_count,
    is_atomic_caption,
)


def summarize_caption_granularity(
    humanml_root: Path,
    splits: Iterable[str] = ("train", "val", "test"),
    max_caption_words: int = 0,
    examples_per_bucket: int = 5,
) -> dict[str, object]:
    catalog = load_humanml3d_catalog(humanml_root)
    split_rows: dict[str, object] = {}
    for split in splits:
        if split not in catalog.split_ids:
            raise ValueError(f"unknown split: {split}")
        ids = list(catalog.split_ids[split])
        mode_counts = {mode: 0 for mode in CAPTION_FILTER_MODE_CHOICES}
        first_caption_non_atomic = 0
        best_caption_non_atomic = 0
        total_caption_count = 0
        word_counts = []
        complexity_scores = []
        examples = {
            "first_non_atomic": [],
            "accepted_atomic": [],
            "rejected_atomic": [],
        }
        for sample_id in ids:
            sample = catalog.by_id[sample_id]
            total_caption_count += len(sample.captions)
            first_caption = _choose_caption(sample.captions, sample_id, filter_mode="none")
            best_caption = _choose_caption(
                sample.captions,
                sample_id,
                filter_mode="prefer_atomic",
                max_caption_words=max_caption_words,
            )
            atomic_caption = _choose_caption(
                sample.captions,
                sample_id,
                filter_mode="atomic",
                max_caption_words=max_caption_words,
            )
            for mode in CAPTION_FILTER_MODE_CHOICES:
                if (
                    _choose_caption(
                        sample.captions,
                        sample_id,
                        filter_mode=mode,
                        max_caption_words=max_caption_words,
                    )
                    is not None
                ):
                    mode_counts[mode] += 1
            if first_caption is not None:
                word_counts.append(caption_word_count(first_caption))
                complexity_scores.append(caption_complexity_score(first_caption, max_caption_words))
                first_atomic = is_atomic_caption(first_caption, max_caption_words=max_caption_words)
                first_caption_non_atomic += int(not first_atomic)
                if not first_atomic and len(examples["first_non_atomic"]) < examples_per_bucket:
                    examples["first_non_atomic"].append({"sample_id": sample_id, "caption": first_caption})
            if best_caption is not None:
                best_caption_non_atomic += int(not is_atomic_caption(best_caption, max_caption_words=max_caption_words))
            if atomic_caption is not None and len(examples["accepted_atomic"]) < examples_per_bucket:
                examples["accepted_atomic"].append({"sample_id": sample_id, "caption": atomic_caption})
            if atomic_caption is None and len(examples["rejected_atomic"]) < examples_per_bucket:
                examples["rejected_atomic"].append({"sample_id": sample_id, "first_caption": first_caption})

        total = max(len(ids), 1)
        split_rows[split] = {
            "samples": len(ids),
            "captions": total_caption_count,
            "avg_captions_per_sample": float(total_caption_count / total),
            "mode_counts": mode_counts,
            "mode_keep_rate": {mode: float(count / total) for mode, count in mode_counts.items()},
            "first_caption_non_atomic": first_caption_non_atomic,
            "first_caption_non_atomic_rate": float(first_caption_non_atomic / total),
            "prefer_atomic_non_atomic": best_caption_non_atomic,
            "prefer_atomic_non_atomic_rate": float(best_caption_non_atomic / total),
            "first_caption_avg_words": float(sum(word_counts) / max(len(word_counts), 1)),
            "first_caption_avg_complexity": float(sum(complexity_scores) / max(len(complexity_scores), 1)),
            "examples": examples,
        }
    return {
        "humanml_root": str(humanml_root),
        "max_caption_words": max_caption_words,
        "splits": split_rows,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml-root", default="../HumanML3D/HumanML3D")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-caption-words", type=int, default=0)
    parser.add_argument("--examples-per-bucket", type=int, default=5)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    summary = summarize_caption_granularity(
        Path(args.humanml_root),
        splits=[item.strip() for item in args.splits.split(",") if item.strip()],
        max_caption_words=args.max_caption_words,
        examples_per_bucket=args.examples_per_bucket,
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
