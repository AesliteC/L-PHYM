from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import csv
import json
import random

import numpy as np


def _read_nonempty_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _parse_caption_file(path: Path) -> List[Dict[str, str]]:
    captions: List[Dict[str, str]] = []
    for line in _read_nonempty_lines(path):
        parts = line.split("#")
        if len(parts) < 4:
            continue
        captions.append(
            {
                "raw": parts[0],
                "processed": parts[1],
                "start": parts[2],
                "end": parts[3],
            }
        )
    return captions


@dataclass(frozen=True)
class HumanML3DSample:
    sample_id: str
    text_path: Path
    joints_path: Path
    vecs_path: Path
    source_path: Path | None
    start_frame: int | None
    end_frame: int | None
    captions: List[Dict[str, str]]


@dataclass(frozen=True)
class HumanML3DCatalog:
    root: Path
    all_ids: List[str]
    split_ids: Dict[str, List[str]]
    by_id: Dict[str, HumanML3DSample]


def load_humanml3d_catalog(root: Path) -> HumanML3DCatalog:
    root = root.resolve()
    if (root / "HumanML3D").is_dir():
        root = root / "HumanML3D"

    all_ids = _read_nonempty_lines(root / "all.txt")
    split_ids = {
        split: _read_nonempty_lines(root / f"{split}.txt")
        for split in ("train", "val", "test", "train_val")
    }

    source_rows: Dict[str, Dict[str, str]] = {}
    index_csv = root.parent / "index.csv"
    if index_csv.exists():
        with index_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                source_rows[Path(row["new_name"]).stem] = row

    by_id: Dict[str, HumanML3DSample] = {}
    for sample_id in all_ids:
        text_path = root / "texts" / f"{sample_id}.txt"
        joints_path = root / "new_joints" / f"{sample_id}.npy"
        vecs_path = root / "new_joint_vecs" / f"{sample_id}.npy"
        source = source_rows.get(sample_id)
        by_id[sample_id] = HumanML3DSample(
            sample_id=sample_id,
            text_path=text_path,
            joints_path=joints_path,
            vecs_path=vecs_path,
            source_path=(root.parent / source["source_path"].lstrip("./")) if source else None,
            start_frame=int(source["start_frame"]) if source else None,
            end_frame=int(source["end_frame"]) if source else None,
            captions=_parse_caption_file(text_path),
        )

    return HumanML3DCatalog(root=root, all_ids=all_ids, split_ids=split_ids, by_id=by_id)


def build_long_horizon_manifest(
    catalog: HumanML3DCatalog,
    split: str,
    num_sequences: int,
    min_clips: int,
    max_clips: int,
    seed: int = 0,
) -> List[Dict[str, object]]:
    if split not in catalog.split_ids:
        raise ValueError(f"unknown split: {split}")
    if min_clips < 1 or max_clips < min_clips:
        raise ValueError("invalid clip bounds")

    rng = random.Random(seed)
    ids = list(catalog.split_ids[split])
    manifest: List[Dict[str, object]] = []

    for seq_idx in range(num_sequences):
        clip_count = rng.randint(min_clips, max_clips)
        sample_ids = [ids[(seq_idx + offset) % len(ids)] for offset in range(clip_count)]
        clip_captions = []
        frame_lengths = []
        source_paths = []
        start_frames = []
        end_frames = []
        selected_caption_texts = []

        for sample_id in sample_ids:
            sample = catalog.by_id[sample_id]
            caption = sample.captions[0] if sample.captions else {"raw": sample_id, "processed": sample_id, "start": "0.0", "end": "0.0"}
            clip_captions.append(caption["raw"])
            selected_caption_texts.append(caption["raw"])
            frame_lengths.append(int(np.load(sample.vecs_path, mmap_mode="r").shape[0]))
            source_paths.append(str(sample.source_path) if sample.source_path is not None else None)
            start_frames.append(sample.start_frame)
            end_frames.append(sample.end_frame)

        caption = " then ".join(selected_caption_texts)
        manifest.append(
            {
                "sequence_id": f"{split}_{seq_idx:06d}",
                "split": split,
                "sample_ids": sample_ids,
                "caption": caption,
                "clip_captions": clip_captions,
                "frame_lengths": frame_lengths,
                "source_paths": source_paths,
                "start_frames": start_frames,
                "end_frames": end_frames,
            }
        )

    return manifest


def write_manifest_jsonl(manifest: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
