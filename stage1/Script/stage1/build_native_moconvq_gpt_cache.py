from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
import argparse
import json

import h5py
import numpy as np
import torch

from Script.stage1.real_moconvq_cache import (
    DEFAULT_TEXT_GPT_BLOCK_SIZE,
    build_loaded_moconvq_agent,
    build_t5_text_encoder,
    encode_observation_with_agent,
    make_windows,
)


def parse_motion_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                "motion specs must be formatted as '<h5_group_or_observation_key>=<caption>', "
                f"got {value!r}"
            )
        key, caption = value.split("=", 1)
        key = key.strip().strip("/")
        caption = caption.strip()
        if not key or not caption:
            raise ValueError(f"invalid motion spec: {value!r}")
        specs.append((key, caption))
    return specs


def _resolve_observation_dataset(handle: h5py.File, key: str) -> np.ndarray:
    if key in handle and isinstance(handle[key], h5py.Dataset):
        observation_key = key
    else:
        observation_key = f"{key.rstrip('/')}/observation"
    if observation_key not in handle:
        raise KeyError(f"observation dataset not found for key {key!r}: tried {observation_key!r}")
    observation = np.asarray(handle[observation_key], dtype=np.float32)
    if observation.ndim != 2 or observation.shape[-1] != 323:
        raise ValueError(f"expected observation shape (T, 323), got {observation.shape} for {observation_key}")
    return observation


def _empty_text_tensors(window_size: int, rvq_depth: int) -> dict[str, object]:
    return {
        "latents": torch.empty((0, window_size, 768), dtype=torch.float32),
        "indices": torch.empty((0, window_size, rvq_depth), dtype=torch.long),
        "text_features": torch.empty((0, 0, 1024), dtype=torch.float32),
        "text_masks": torch.empty((0, 0), dtype=torch.bool),
        "target_masks": torch.empty((0, window_size), dtype=torch.bool),
        "end_masks": torch.empty((0, window_size), dtype=torch.bool),
    }


def build_native_cache_from_h5(
    native_h5_path: Path,
    motion_specs: list[tuple[str, str]],
    agent,
    text_encoder: Callable[[list[str]], tuple[np.ndarray, np.ndarray]],
    window_size: int,
    window_stride: int,
    rvq_depth: int = 4,
    pad_index: int = 513,
    include_tail: bool = True,
    text_model: str | None = None,
    max_text_length: int | None = None,
) -> dict[str, object]:
    max_motion_tokens = DEFAULT_TEXT_GPT_BLOCK_SIZE - 1
    if window_size > max_motion_tokens:
        raise ValueError(
            f"window_size {window_size} exceeds GPT motion context {max_motion_tokens}; "
            f"block_size {DEFAULT_TEXT_GPT_BLOCK_SIZE} reserves one condition token"
        )
    if not motion_specs:
        raise ValueError("at least one motion spec is required")

    latents_all: list[np.ndarray] = []
    indices_all: list[np.ndarray] = []
    text_features_all: list[np.ndarray] = []
    text_masks_all: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    end_masks: list[np.ndarray] = []
    captions: list[str] = []
    sequence_ids: list[str] = []
    window_ranges: list[tuple[int, int]] = []
    sample_ids: list[list[str]] = []
    source_observation_shapes: dict[str, list[int]] = {}
    encoded_text_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    with h5py.File(native_h5_path, "r") as handle:
        for key, caption in motion_specs:
            observation = _resolve_observation_dataset(handle, key)
            source_observation_shapes[key] = list(observation.shape)
            latent, index = encode_observation_with_agent(agent, observation, rvq_depth=rvq_depth)
            windows = make_windows(
                latent,
                index,
                window_size=window_size,
                window_stride=window_stride,
                pad_index=pad_index,
                include_tail=include_tail,
            )
            if caption not in encoded_text_cache:
                encoded_text_cache[caption] = text_encoder([caption])
            text_feature, text_mask = encoded_text_cache[caption]
            for latent_window, index_window, window_range in windows:
                valid_time = index_window[:, 0] != pad_index
                target_mask = valid_time.astype(bool)
                end_mask = np.zeros((window_size,), dtype=bool)
                valid_count = int(valid_time.sum())
                if valid_count < window_size:
                    end_mask[valid_count] = True
                latents_all.append(latent_window)
                indices_all.append(index_window)
                text_features_all.append(text_feature[0])
                text_masks_all.append(text_mask[0])
                target_masks.append(target_mask)
                end_masks.append(end_mask)
                captions.append(caption)
                sequence_ids.append(key)
                window_ranges.append(tuple(window_range))
                sample_ids.append([key])

    cache = _empty_text_tensors(window_size=window_size, rvq_depth=rvq_depth)
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
            "target_ranges": window_ranges,
            "prefix_ranges": [(start, start) for start, _ in window_ranges],
            "segment_ranges": window_ranges,
            "segment_idxs": torch.zeros((len(window_ranges),), dtype=torch.long),
            "num_segments": torch.ones((len(window_ranges),), dtype=torch.long),
            "segment_progress": torch.zeros((len(window_ranges),), dtype=torch.float32),
            "prefix_lengths": torch.zeros((len(window_ranges),), dtype=torch.long),
            "sample_ids": sample_ids,
            "filtered_sequences": [],
            "observation_quality": [],
            "config": {
                "source": "native_moconvq_observation_h5",
                "native_h5": str(native_h5_path),
                "motion_specs": [{"key": key, "caption": caption} for key, caption in motion_specs],
                "source_observation_shapes": source_observation_shapes,
                "window_size": window_size,
                "window_stride": window_stride,
                "rvq_depth": rvq_depth,
                "pad_index": pad_index,
                "include_tail": include_tail,
                "text_model": text_model,
                "max_text_length": max_text_length,
            },
        }
    )
    return cache


def summarize_cache(cache: dict[str, object]) -> dict[str, object]:
    indices = torch.as_tensor(cache["indices"], dtype=torch.long)
    valid = indices != 513
    return {
        "windows": int(torch.as_tensor(cache["latents"]).shape[0]),
        "latents_shape": list(torch.as_tensor(cache["latents"]).shape),
        "indices_shape": list(indices.shape),
        "text_features_shape": list(torch.as_tensor(cache["text_features"]).shape),
        "valid_tokens": int(valid.sum().item()),
        "index_min": int(indices[valid].min().item()) if bool(valid.any()) else -1,
        "index_max": int(indices[valid].max().item()) if bool(valid.any()) else -1,
        "unique_sequences": len(set(str(item) for item in cache["sequence_ids"])),  # type: ignore[index]
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-h5", default="simple_motion_data.h5")
    parser.add_argument(
        "--motion",
        action="append",
        default=[],
        help="Motion spec formatted as '<group_or_observation_key>=<caption>'. "
        "Example: walk1_subject5='a person walks forward'.",
    )
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--text-model", default="t5-large")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--window-stride", type=int, default=25)
    parser.add_argument("--rvq-depth", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args(argv)

    motion_specs = parse_motion_specs(args.motion or ["walk1_subject5=a person walks forward"])
    import MoConVQCore.Utils.pytorch_utils as ptu

    agent = build_loaded_moconvq_agent(gpu=args.gpu, base_data=Path(args.base_data))
    text_encoder = build_t5_text_encoder(args.text_model, device=str(ptu.device), max_length=args.max_text_length)
    cache = build_native_cache_from_h5(
        native_h5_path=Path(args.native_h5),
        motion_specs=motion_specs,
        agent=agent,
        text_encoder=text_encoder,
        window_size=args.window_size,
        window_stride=args.window_stride,
        rvq_depth=args.rvq_depth,
        text_model=args.text_model,
        max_text_length=args.max_text_length,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    summary = summarize_cache(cache)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
