from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable
import argparse
import json
import sys

if __package__ in {None, ""}:
    repo_root = str(Path(__file__).resolve().parents[2])
    if not sys.path or sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)

import torch

from Script.stage1.run_text_gpt_comparison import PromptRecord, format_prompt_tsv_line


def short_sequence_name(sequence_id: str) -> str:
    parts = sequence_id.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return sequence_id


def _int_value(value: object) -> int:
    if hasattr(value, "item"):
        return int(value.item())  # type: ignore[union-attr]
    return int(value)


def scale_segment_lengths(raw_lengths: Iterable[int], total_length: int) -> tuple[int, ...]:
    raw = [int(item) for item in raw_lengths]
    if not raw:
        raise ValueError("raw segment lengths must not be empty")
    if any(item < 1 for item in raw):
        raise ValueError(f"raw segment lengths must be positive, got {raw}")
    if total_length < len(raw):
        raise ValueError(f"total_length {total_length} is too short for {len(raw)} segments")
    raw_total = float(sum(raw))
    exact = [float(item) * float(total_length) / raw_total for item in raw]
    scaled = [max(1, int(item)) for item in exact]
    while sum(scaled) > total_length:
        candidates = [idx for idx, value in enumerate(scaled) if value > 1]
        if not candidates:
            break
        idx = min(candidates, key=lambda item: exact[item] - int(exact[item]))
        scaled[idx] -= 1
    remainder_order = sorted(
        range(len(raw)),
        key=lambda item: (exact[item] - int(exact[item]), raw[item]),
        reverse=True,
    )
    order_idx = 0
    while sum(scaled) < total_length:
        idx = remainder_order[order_idx % len(remainder_order)]
        scaled[idx] += 1
        order_idx += 1
    return tuple(scaled)


def _validate_cache(cache: dict[str, object]) -> None:
    required = ("sequence_ids", "captions", "segment_idxs", "num_segments", "segment_ranges")
    missing = [key for key in required if key not in cache]
    if missing:
        raise ValueError(f"cache is missing segment prompt metadata: {missing}")
    expected = len(cache["sequence_ids"])  # type: ignore[arg-type]
    for key in required[1:]:
        if len(cache[key]) != expected:  # type: ignore[arg-type]
            raise ValueError(f"cache field {key!r} length does not match sequence_ids")


def prompts_from_cache(
    cache: dict[str, object],
    *,
    total_length: int,
    joiner: str = " then ",
    name_mode: str = "short",
    limit: int | None = None,
) -> tuple[list[PromptRecord], list[dict[str, object]]]:
    if name_mode not in {"short", "sequence_id"}:
        raise ValueError(f"unknown name_mode: {name_mode}")
    _validate_cache(cache)

    groups: OrderedDict[str, dict[str, object]] = OrderedDict()
    sequence_ids = cache["sequence_ids"]  # type: ignore[assignment]
    captions = cache["captions"]  # type: ignore[assignment]
    segment_idxs = cache["segment_idxs"]  # type: ignore[assignment]
    num_segments_values = cache["num_segments"]  # type: ignore[assignment]
    segment_ranges = cache["segment_ranges"]  # type: ignore[assignment]

    for row_idx, raw_sequence_id in enumerate(sequence_ids):
        sequence_id = str(raw_sequence_id)
        segment_idx = _int_value(segment_idxs[row_idx])
        num_segments = _int_value(num_segments_values[row_idx])
        if segment_idx < 0 or segment_idx >= num_segments:
            raise ValueError(f"invalid segment_idx={segment_idx} for {sequence_id}")
        group = groups.setdefault(
            sequence_id,
            {
                "num_segments": num_segments,
                "segments": [None] * num_segments,
                "ranges": [None] * num_segments,
            },
        )
        if int(group["num_segments"]) != num_segments:
            raise ValueError(f"inconsistent num_segments for {sequence_id}")
        caption = str(captions[row_idx]).strip()
        if not caption:
            raise ValueError(f"empty caption for {sequence_id} segment {segment_idx}")
        segment_range = tuple(int(item) for item in segment_ranges[row_idx])
        if len(segment_range) != 2 or segment_range[1] <= segment_range[0]:
            raise ValueError(f"invalid segment range for {sequence_id} segment {segment_idx}: {segment_range}")

        segments = group["segments"]  # type: ignore[assignment]
        ranges = group["ranges"]  # type: ignore[assignment]
        existing_caption = segments[segment_idx]
        existing_range = ranges[segment_idx]
        if existing_caption is not None and existing_caption != caption:
            raise ValueError(f"inconsistent caption for {sequence_id} segment {segment_idx}")
        if existing_range is not None and existing_range != segment_range:
            raise ValueError(f"inconsistent range for {sequence_id} segment {segment_idx}")
        segments[segment_idx] = caption
        ranges[segment_idx] = segment_range

    prompts: list[PromptRecord] = []
    rows: list[dict[str, object]] = []
    for sequence_id, group in groups.items():
        segments = group["segments"]  # type: ignore[assignment]
        ranges = group["ranges"]  # type: ignore[assignment]
        if any(item is None for item in segments) or any(item is None for item in ranges):
            raise ValueError(f"incomplete segment metadata for {sequence_id}")
        clean_segments = tuple(str(item) for item in segments)
        clean_ranges = tuple(tuple(int(value) for value in item) for item in ranges)
        raw_lengths = tuple(end - start for start, end in clean_ranges)
        scaled_lengths = scale_segment_lengths(raw_lengths, total_length=total_length)
        name = short_sequence_name(sequence_id) if name_mode == "short" else sequence_id
        long_text = joiner.join(clean_segments)
        prompts.append(PromptRecord(name, long_text, clean_segments, scaled_lengths))
        rows.append(
            {
                "sequence_id": sequence_id,
                "name": name,
                "segments": list(clean_segments),
                "segment_ranges": [list(item) for item in clean_ranges],
                "raw_lengths": list(raw_lengths),
                "scaled_lengths": list(scaled_lengths),
                "text": long_text,
            }
        )
        if limit is not None and len(prompts) >= limit:
            break
    return prompts, rows


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="Segment-prefix GPT cache .pt file.")
    parser.add_argument("--output", required=True, help="Output prompt TSV path.")
    parser.add_argument("--summary", default="", help="Optional JSON summary path.")
    parser.add_argument("--total-length", type=int, default=75)
    parser.add_argument("--joiner", default=" then ")
    parser.add_argument("--name-mode", choices=("short", "sequence_id"), default="short")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    cache = torch.load(args.cache, map_location="cpu")
    prompts, rows = prompts_from_cache(
        cache,
        total_length=args.total_length,
        joiner=args.joiner,
        name_mode=args.name_mode,
        limit=args.limit,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(format_prompt_tsv_line(prompt) for prompt in prompts), encoding="utf-8")

    payload = {
        "cache": str(args.cache),
        "output": str(output),
        "total_length": int(args.total_length),
        "joiner": args.joiner,
        "name_mode": args.name_mode,
        "limit": args.limit,
        "num_prompts": len(prompts),
        "rows": rows,
    }
    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("output", "num_prompts", "total_length")}, indent=2))


if __name__ == "__main__":
    main()
