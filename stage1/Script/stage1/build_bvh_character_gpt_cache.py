from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import argparse
import json
import sys


def _ensure_own_repo_root_on_path(package: str | None = __package__) -> None:
    if package not in {None, ""}:
        return
    repo_root = str(Path(__file__).resolve().parents[2])
    if not sys.path or sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)


_ensure_own_repo_root_on_path()

import numpy as np
import torch

from Script.stage1.build_native_moconvq_gpt_cache import summarize_cache
from Script.stage1.diagnose_bvh_character_retarget import extract_bvh_with_moconvq_character
from Script.stage1.real_moconvq_cache import (
    CACHE_SAMPLE_MODE_CHOICES,
    DEFAULT_TEXT_GPT_BLOCK_SIZE,
    build_loaded_moconvq_agent,
    build_t5_text_encoder,
    encode_observation_with_agent,
    make_clip_aligned_windows,
    make_segment_prefix_windows,
    make_windows,
    select_window_caption,
)


@dataclass(frozen=True)
class BVHSpec:
    path: Path
    caption: str
    sample_ids: tuple[str, ...] = ()
    clip_captions: tuple[str, ...] = ()
    clip_boundaries: tuple[tuple[int, int], ...] = ()
    transition_forced: tuple[bool, ...] = ()


def _as_bvh_spec(value: BVHSpec | tuple[Path, str]) -> BVHSpec:
    if isinstance(value, BVHSpec):
        return value
    path, caption = value
    return BVHSpec(path=Path(path), caption=str(caption))


def parse_bvh_specs(values: list[str]) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("BVH specs must be formatted as '<path.bvh>=<caption>'")
        path, caption = value.split("=", 1)
        bvh_path = Path(path.strip())
        caption = caption.strip()
        if not str(bvh_path) or not caption:
            raise ValueError(f"invalid BVH spec: {value!r}")
        specs.append((bvh_path, caption))
    return specs


def _load_export_rows(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not str(path):
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, object]] = {}
    for raw_row in payload.get("exports", []):
        row = dict(raw_row)
        keys = {
            str(row.get("sample_id", "")),
            str(row.get("sequence_id", "")),
            str(Path(str(row.get("output_bvh", ""))).stem) if row.get("output_bvh") else "",
            str(Path(str(row.get("output_bvh", "")))) if row.get("output_bvh") else "",
        }
        for key in keys:
            if key:
                rows[key] = row
    return rows


def _row_to_bvh_spec(row: dict[str, object], export_rows: dict[str, dict[str, object]] | None = None) -> BVHSpec:
    export_rows = export_rows or {}
    path = Path(str(row["path"]))
    label = str(row.get("label") or path.stem)
    export_row = export_rows.get(label) or export_rows.get(str(path)) or {}
    caption = str(row.get("caption") or export_row.get("caption") or path.stem).strip() or path.stem
    sample_ids = row.get("sample_ids") or export_row.get("sample_ids") or []
    clip_captions = row.get("clip_captions") or export_row.get("clip_captions") or []
    clip_boundaries = row.get("clip_boundaries") or export_row.get("clip_boundaries") or []
    transition_forced = row.get("transition_forced") or export_row.get("transition_forced") or []
    return BVHSpec(
        path=path,
        caption=caption,
        sample_ids=tuple(str(item) for item in sample_ids),
        clip_captions=tuple(str(item) for item in clip_captions),
        clip_boundaries=tuple((int(start), int(end)) for start, end in clip_boundaries),
        transition_forced=tuple(bool(item) for item in transition_forced),
    )


def bvh_specs_from_quality_summary(
    path: Path,
    accepted_only: bool = True,
    export_summary: Path | None = None,
) -> list[BVHSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    export_rows = _load_export_rows(export_summary)
    specs: list[BVHSpec] = []
    for row in payload.get("rows", []):
        if accepted_only and not bool(row.get("accepted")):
            continue
        specs.append(_row_to_bvh_spec(dict(row), export_rows=export_rows))
    if not specs:
        raise ValueError(f"quality summary produced no BVH specs: {path}")
    return specs


def specs_from_quality_summary(path: Path, accepted_only: bool = True) -> list[tuple[Path, str]]:
    return [(spec.path, spec.caption) for spec in bvh_specs_from_quality_summary(path, accepted_only=accepted_only)]


def _write_observation_h5(path: Path, rows: list[dict[str, object]]) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for row in rows:
            key = str(row["key"])
            observation = np.asarray(row["observation"], dtype=np.float32)
            spec = _as_bvh_spec(row["spec"])  # type: ignore[arg-type]
            group = handle.create_group(key)
            group.create_dataset("observation", data=np.asarray(observation, dtype=np.float32), compression="gzip")
            group.attrs["caption"] = spec.caption
            group.attrs["sample_ids_json"] = json.dumps(list(spec.sample_ids), ensure_ascii=False)
            group.attrs["clip_captions_json"] = json.dumps(list(spec.clip_captions), ensure_ascii=False)
            group.attrs["clip_boundaries_json"] = json.dumps([list(item) for item in spec.clip_boundaries])
            group.attrs["transition_forced_json"] = json.dumps(list(spec.transition_forced))


def _latent_clip_boundaries(
    clip_boundaries: tuple[tuple[int, int], ...],
    *,
    latent_length: int,
    observation_length: int,
) -> list[tuple[int, int]]:
    if observation_length <= 0 or not clip_boundaries:
        return []
    scale = float(latent_length) / float(observation_length)
    boundaries: list[tuple[int, int]] = []
    last_end = 0
    for idx, (start, end) in enumerate(clip_boundaries):
        latent_start = int(round(float(start) * scale))
        latent_end = int(round(float(end) * scale))
        if idx == 0:
            latent_start = 0
        latent_start = max(last_end, min(latent_start, latent_length))
        latent_end = max(latent_start, min(latent_end, latent_length))
        if latent_end > latent_start:
            boundaries.append((latent_start, latent_end))
            last_end = latent_end
    if boundaries:
        start, _ = boundaries[-1]
        boundaries[-1] = (start, latent_length)
    return boundaries


def _empty_cache(window_size: int, rvq_depth: int) -> dict[str, object]:
    return {
        "latents": torch.empty((0, window_size, 768), dtype=torch.float32),
        "indices": torch.empty((0, window_size, rvq_depth), dtype=torch.long),
        "text_features": torch.empty((0, 0, 1024), dtype=torch.float32),
        "text_masks": torch.empty((0, 0), dtype=torch.bool),
        "target_masks": torch.empty((0, window_size), dtype=torch.bool),
        "end_masks": torch.empty((0, window_size), dtype=torch.bool),
    }


def build_cache_from_bvh_observations(
    rows: list[dict[str, object]],
    agent,
    text_encoder,
    window_size: int,
    window_stride: int,
    rvq_depth: int = 4,
    pad_index: int = 513,
    caption_mode: str = "window",
    caption_joiner: str = " then ",
    window_policy: str = "clip",
    sample_mode: str = "segment_prefix",
    prefix_size: int = 25,
    text_model: str | None = None,
    max_text_length: int | None = None,
) -> dict[str, object]:
    if caption_mode not in {"sequence", "window"}:
        raise ValueError(f"unknown caption_mode: {caption_mode}")
    if window_policy not in {"sequence", "clip"}:
        raise ValueError(f"unknown window_policy: {window_policy}")
    if sample_mode not in CACHE_SAMPLE_MODE_CHOICES:
        raise ValueError(f"unknown sample_mode: {sample_mode}")
    max_motion_tokens = DEFAULT_TEXT_GPT_BLOCK_SIZE - 1
    if window_size > max_motion_tokens:
        raise ValueError(
            f"window_size {window_size} exceeds GPT motion context {max_motion_tokens}; "
            f"block_size {DEFAULT_TEXT_GPT_BLOCK_SIZE} reserves one condition token"
        )

    latents_all: list[np.ndarray] = []
    indices_all: list[np.ndarray] = []
    text_features_all: list[np.ndarray] = []
    text_masks_all: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    end_masks: list[np.ndarray] = []
    captions: list[str] = []
    sequence_ids: list[str] = []
    window_ranges: list[tuple[int, int]] = []
    target_ranges: list[tuple[int, int]] = []
    prefix_ranges: list[tuple[int, int]] = []
    segment_ranges: list[tuple[int, int]] = []
    segment_idxs: list[int] = []
    num_segments_all: list[int] = []
    segment_progresses: list[float] = []
    prefix_lengths: list[int] = []
    sample_ids_all: list[list[str]] = []
    source_observation_shapes: dict[str, list[int]] = {}
    encoded_text_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for row in rows:
        key = str(row["key"])
        spec = _as_bvh_spec(row["spec"])  # type: ignore[arg-type]
        observation = np.asarray(row["observation"], dtype=np.float32)
        source_observation_shapes[key] = list(observation.shape)
        latent, index = encode_observation_with_agent(agent, observation, rvq_depth=rvq_depth)
        latent_boundaries = _latent_clip_boundaries(
            spec.clip_boundaries,
            latent_length=len(latent),
            observation_length=len(observation),
        )
        if sample_mode == "segment_prefix" and latent_boundaries:
            sample_windows = make_segment_prefix_windows(
                latent,
                index,
                window_size=window_size,
                window_stride=window_stride,
                clip_boundaries=latent_boundaries,
                prefix_size=prefix_size,
                pad_index=pad_index,
            )
        else:
            if window_policy == "clip" and latent_boundaries:
                windows = make_clip_aligned_windows(
                    latent,
                    index,
                    window_size=window_size,
                    window_stride=window_stride,
                    clip_boundaries=latent_boundaries,
                    pad_index=pad_index,
                )
            else:
                windows = make_windows(
                    latent,
                    index,
                    window_size=window_size,
                    window_stride=window_stride,
                    pad_index=pad_index,
                )
            sample_windows = []
            for latent_window, index_window, window_range in windows:
                valid_time = index_window[:, 0] != pad_index
                valid_count = int(valid_time.sum())
                end_mask = np.zeros((window_size,), dtype=bool)
                if valid_count < window_size:
                    end_mask[valid_count] = True
                sample_windows.append(
                    {
                        "latent": latent_window,
                        "indices": index_window,
                        "target_mask": valid_time.astype(bool),
                        "end_mask": end_mask,
                        "window_range": window_range,
                        "target_range": window_range,
                        "prefix_range": (window_range[0], window_range[0]),
                        "segment_idx": 0,
                        "num_segments": max(len(latent_boundaries), 1),
                        "segment_range": window_range,
                        "prefix_length": 0,
                    }
                )

        for sample_window in sample_windows:
            caption_window_range = (
                sample_window["target_range"] if sample_mode == "segment_prefix" else sample_window["window_range"]
            )
            window_caption = spec.caption
            if caption_mode == "window":
                window_caption = select_window_caption(
                    full_caption=spec.caption,
                    clip_captions=list(spec.clip_captions),
                    clip_boundaries=[list(item) for item in spec.clip_boundaries],
                    window_range=caption_window_range,  # type: ignore[arg-type]
                    latent_length=len(latent),
                    observation_length=len(observation),
                    joiner=caption_joiner,
                )
            if window_caption not in encoded_text_cache:
                encoded_text_cache[window_caption] = text_encoder([window_caption])
            text_feature, text_mask = encoded_text_cache[window_caption]
            latents_all.append(np.asarray(sample_window["latent"], dtype=np.float32))
            indices_all.append(np.asarray(sample_window["indices"], dtype=np.int64))
            target_masks.append(np.asarray(sample_window["target_mask"], dtype=bool))
            end_masks.append(np.asarray(sample_window["end_mask"], dtype=bool))
            text_features_all.append(text_feature[0])
            text_masks_all.append(text_mask[0])
            captions.append(window_caption)
            sequence_ids.append(key)
            window_ranges.append(tuple(sample_window["window_range"]))  # type: ignore[arg-type]
            target_ranges.append(tuple(sample_window["target_range"]))  # type: ignore[arg-type]
            prefix_ranges.append(tuple(sample_window["prefix_range"]))  # type: ignore[arg-type]
            segment_ranges.append(tuple(sample_window["segment_range"]))  # type: ignore[arg-type]
            segment_idx = int(sample_window["segment_idx"])
            num_segments = int(sample_window["num_segments"])
            segment_idxs.append(segment_idx)
            num_segments_all.append(num_segments)
            segment_progresses.append(float(segment_idx / max(num_segments - 1, 1)) if num_segments > 1 else 0.0)
            prefix_lengths.append(int(sample_window["prefix_length"]))
            sample_ids_all.append(list(spec.sample_ids) if spec.sample_ids else [key])

    cache = _empty_cache(window_size=window_size, rvq_depth=rvq_depth)
    if latents_all:
        cache.update(
            {
                "latents": torch.from_numpy(np.stack(latents_all, axis=0)),
                "indices": torch.from_numpy(np.stack(indices_all, axis=0)),
                "text_features": torch.from_numpy(np.stack(text_features_all, axis=0)),
                "text_masks": torch.from_numpy(np.stack(text_masks_all, axis=0)),
                "target_masks": torch.from_numpy(np.stack(target_masks, axis=0)),
                "end_masks": torch.from_numpy(np.stack(end_masks, axis=0)),
            }
        )
    cache.update(
        {
            "captions": captions,
            "sequence_ids": sequence_ids,
            "window_ranges": window_ranges,
            "target_ranges": target_ranges,
            "prefix_ranges": prefix_ranges,
            "segment_ranges": segment_ranges,
            "segment_idxs": torch.as_tensor(segment_idxs, dtype=torch.long),
            "num_segments": torch.as_tensor(num_segments_all, dtype=torch.long),
            "segment_progress": torch.as_tensor(segment_progresses, dtype=torch.float32),
            "prefix_lengths": torch.as_tensor(prefix_lengths, dtype=torch.long),
            "sample_ids": sample_ids_all,
            "filtered_sequences": [],
            "observation_quality": [],
            "config": {
                "source": "bvh_character_moconvq_observation",
                "source_observation_shapes": source_observation_shapes,
                "window_size": window_size,
                "window_stride": window_stride,
                "rvq_depth": rvq_depth,
                "pad_index": pad_index,
                "caption_mode": caption_mode,
                "caption_joiner": caption_joiner,
                "window_policy": window_policy,
                "sample_mode": sample_mode,
                "prefix_size": prefix_size,
                "text_model": text_model,
                "max_text_length": max_text_length,
            },
        }
    )
    return cache


def build_bvh_character_cache(
    bvh_specs: list[BVHSpec | tuple[Path, str]],
    agent,
    text_encoder,
    output_observation_h5: Path,
    window_size: int,
    window_stride: int,
    rvq_depth: int = 4,
    fps: int = 20,
    flip: bool = False,
    text_model: str | None = None,
    max_text_length: int | None = None,
    caption_mode: str = "window",
    caption_joiner: str = " then ",
    window_policy: str = "clip",
    sample_mode: str = "segment_prefix",
    prefix_size: int = 25,
) -> dict[str, object]:
    if not bvh_specs:
        raise ValueError("at least one BVH spec is required")
    specs = [_as_bvh_spec(spec) for spec in bvh_specs]
    rows: list[dict[str, object]] = []
    for idx, spec in enumerate(specs):
        bvh_path = spec.path
        if not bvh_path.exists():
            raise FileNotFoundError(str(bvh_path))
        motion = extract_bvh_with_moconvq_character([bvh_path], agent=agent, fps=fps, flip=flip)
        key = f"{idx:04d}_{bvh_path.stem}"
        rows.append({"key": key, "observation": motion["observation"], "spec": spec})
    _write_observation_h5(output_observation_h5, rows)
    cache = build_cache_from_bvh_observations(
        rows=rows,
        agent=agent,
        text_encoder=text_encoder,
        window_size=window_size,
        window_stride=window_stride,
        rvq_depth=rvq_depth,
        text_model=text_model,
        max_text_length=max_text_length,
        caption_mode=caption_mode,
        caption_joiner=caption_joiner,
        window_policy=window_policy,
        sample_mode=sample_mode,
        prefix_size=prefix_size,
    )
    cache["config"]["bvh_specs"] = [  # type: ignore[index]
        {
            "path": str(spec.path),
            "caption": spec.caption,
            "sample_ids": list(spec.sample_ids),
            "clip_captions": list(spec.clip_captions),
            "clip_boundaries": [list(item) for item in spec.clip_boundaries],
            "transition_forced": list(spec.transition_forced),
        }
        for spec in specs
    ]
    cache["config"]["fps"] = fps  # type: ignore[index]
    cache["config"]["flip"] = flip  # type: ignore[index]
    cache["config"]["intermediate_observation_h5"] = str(output_observation_h5)  # type: ignore[index]
    return cache


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh",
        action="append",
        default=[],
        help="BVH spec formatted as '<path.bvh>=<caption>'.",
    )
    parser.add_argument(
        "--motion",
        action="append",
        default=[],
        help="Alias for --bvh; accepted for consistency with native cache specs.",
    )
    parser.add_argument(
        "--quality-summary",
        default="",
        help="Optional summarize_bvh_retarget_quality.py JSON; accepted rows are converted to BVH specs.",
    )
    parser.add_argument(
        "--export-summary",
        default="",
        help="Optional export_long_humanml3d_to_bvh.py JSON; supplements quality rows with clip captions/boundaries.",
    )
    parser.add_argument(
        "--include-rejected-quality",
        action="store_true",
        help="Use all quality-summary rows instead of accepted rows only.",
    )
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--motion-dataset", default="")
    parser.add_argument("--text-model", default="t5-large")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--window-stride", type=int, default=25)
    parser.add_argument("--rvq-depth", type=int, default=4)
    parser.add_argument("--caption-mode", choices=("sequence", "window"), default="window")
    parser.add_argument("--caption-joiner", default=" then ")
    parser.add_argument("--window-policy", choices=("sequence", "clip"), default="clip")
    parser.add_argument("--sample-mode", choices=CACHE_SAMPLE_MODE_CHOICES, default="segment_prefix")
    parser.add_argument("--prefix-size", type=int, default=25)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observation-h5", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--quiet", action="store_true", help="Print only summary path and core cache counts.")
    args = parser.parse_args(argv)

    raw_specs = args.bvh + args.motion
    bvh_specs: list[BVHSpec | tuple[Path, str]] = parse_bvh_specs(raw_specs)
    if args.quality_summary:
        bvh_specs.extend(
            bvh_specs_from_quality_summary(
                Path(args.quality_summary),
                accepted_only=not args.include_rejected_quality,
                export_summary=Path(args.export_summary) if args.export_summary else None,
            )
        )
    if not bvh_specs:
        raise SystemExit("provide at least one --bvh '<path.bvh>=<caption>' or --quality-summary")

    import MoConVQCore.Utils.pytorch_utils as ptu

    agent = build_loaded_moconvq_agent(
        gpu=args.gpu,
        base_data=Path(args.base_data),
        motion_dataset=Path(args.motion_dataset) if args.motion_dataset else None,
    )
    text_encoder = build_t5_text_encoder(args.text_model, device=str(ptu.device), max_length=args.max_text_length)
    cache = build_bvh_character_cache(
        bvh_specs=bvh_specs,
        agent=agent,
        text_encoder=text_encoder,
        output_observation_h5=Path(args.observation_h5),
        window_size=args.window_size,
        window_stride=args.window_stride,
        rvq_depth=args.rvq_depth,
        fps=args.fps,
        flip=args.flip,
        text_model=args.text_model,
        max_text_length=args.max_text_length,
        caption_mode=args.caption_mode,
        caption_joiner=args.caption_joiner,
        window_policy=args.window_policy,
        sample_mode=args.sample_mode,
        prefix_size=args.prefix_size,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    summary = summarize_cache(cache)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.quiet:
        print(
            json.dumps(
                {
                    "output": str(output),
                    "summary": args.summary,
                    "windows": summary["windows"],
                    "valid_tokens": summary["valid_tokens"],
                    "unique_sequences": summary["unique_sequences"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
