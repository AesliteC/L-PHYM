from __future__ import annotations

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

import h5py
import numpy as np

from Script.stage1.export_humanml3d_to_bvh import (
    DEFAULT_TEMPLATE_BVH,
    ROTATION_SOURCE_CHOICES,
    _template_hierarchy_lines,
    humanml3d_sample_to_bvh_motion,
)
from Script.stage1.real_moconvq_cache import load_manifest


def _read_sequence_arrays(group: h5py.Group, sequence_id: str) -> tuple[np.ndarray, np.ndarray]:
    if "joints_22" not in group:
        raise ValueError(f"sequence {sequence_id} has no joints_22 dataset")
    joints = np.asarray(group["joints_22"], dtype=np.float32)
    if "joint_vecs_263" in group:
        joint_vecs = np.asarray(group["joint_vecs_263"], dtype=np.float32)
    else:
        joint_vecs = np.zeros((len(joints), 263), dtype=np.float32)
    length = min(len(joints), len(joint_vecs))
    if length < 2:
        raise ValueError(f"sequence {sequence_id} is too short for BVH export: {length} frames")
    return joints[:length], joint_vecs[:length]


def write_long_humanml3d_bvh(
    *,
    sequence_id: str,
    group: h5py.Group,
    manifest_row: dict[str, object] | None,
    output_bvh: Path,
    template_bvh: Path = DEFAULT_TEMPLATE_BVH,
    output_fps: float = 20.0,
    rotation_source: str = "joints_ik",
    unwrap_euler: bool = True,
) -> dict[str, object]:
    joints, joint_vecs = _read_sequence_arrays(group, sequence_id)
    motion = humanml3d_sample_to_bvh_motion(
        joints,
        joint_vecs,
        template_bvh=template_bvh,
        rotation_source=rotation_source,
        unwrap_euler=unwrap_euler,
    )
    hierarchy = _template_hierarchy_lines(template_bvh)
    output_bvh.parent.mkdir(parents=True, exist_ok=True)
    with output_bvh.open("w", encoding="utf-8") as handle:
        for line in hierarchy:
            handle.write(line)
            handle.write("\n")
        handle.write("MOTION\n")
        handle.write(f"Frames: {motion.shape[0]}\n")
        handle.write(f"Frame Time:   {1.0 / float(output_fps):0.6f}\n")
        for row in motion:
            handle.write(" ".join(f"{value: .6f}" for value in row))
            handle.write("\n")

    manifest_row = manifest_row or {}
    caption = str(manifest_row.get("caption") or group.attrs.get("caption") or sequence_id)
    sample_ids = manifest_row.get("sample_ids")
    clip_captions = manifest_row.get("clip_captions")
    clip_boundaries = manifest_row.get("clip_boundaries")
    transition_scores = manifest_row.get("transition_scores")
    transition_forced = manifest_row.get("transition_forced")
    return {
        "sample_id": sequence_id,
        "sequence_id": sequence_id,
        "output_bvh": str(output_bvh),
        "template_bvh": str(template_bvh),
        "frames": int(motion.shape[0]),
        "channels": int(motion.shape[1]),
        "frame_time": float(1.0 / float(output_fps)),
        "caption": caption,
        "sample_ids": sample_ids if isinstance(sample_ids, list) else [],
        "clip_captions": clip_captions if isinstance(clip_captions, list) else [],
        "clip_boundaries": clip_boundaries if isinstance(clip_boundaries, list) else [],
        "transition_scores": transition_scores if isinstance(transition_scores, list) else [],
        "transition_forced": transition_forced if isinstance(transition_forced, list) else [],
        "rotation_source": rotation_source,
        "unwrap_euler": bool(unwrap_euler),
    }


def export_long_humanml3d_to_bvh(
    *,
    long_h5: Path,
    manifest_path: Path,
    output_dir: Path,
    sequence_ids: list[str] | None = None,
    limit: int | None = None,
    template_bvh: Path = DEFAULT_TEMPLATE_BVH,
    output_fps: float = 20.0,
    rotation_source: str = "joints_ik",
    unwrap_euler: bool = True,
) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    exports: list[dict[str, object]] = []
    with h5py.File(long_h5, "r") as h5:
        selected = list(sequence_ids) if sequence_ids else list(h5.keys())
        if limit is not None:
            if limit < 1:
                raise ValueError("--limit must be positive")
            selected = selected[:limit]
        for sequence_id in selected:
            if sequence_id not in h5:
                raise KeyError(f"sequence {sequence_id!r} not found in {long_h5}")
            exports.append(
                write_long_humanml3d_bvh(
                    sequence_id=sequence_id,
                    group=h5[sequence_id],
                    manifest_row=manifest.get(sequence_id),
                    output_bvh=output_dir / f"{sequence_id}.bvh",
                    template_bvh=template_bvh,
                    output_fps=output_fps,
                    rotation_source=rotation_source,
                    unwrap_euler=unwrap_euler,
                )
            )
    return {
        "config": {
            "long_h5": str(long_h5),
            "manifest": str(manifest_path),
            "output_dir": str(output_dir),
            "template_bvh": str(template_bvh),
            "fps": float(output_fps),
            "rotation_source": rotation_source,
            "unwrap_euler": bool(unwrap_euler),
            "sequence_ids": sequence_ids or [],
            "limit": limit,
        },
        "exports": exports,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-h5", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sequence-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--template-bvh", default=str(DEFAULT_TEMPLATE_BVH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--rotation-source", choices=ROTATION_SOURCE_CHOICES, default="joints_ik")
    parser.add_argument("--no-unwrap-euler", action="store_true")
    parser.add_argument("--summary", default="")
    parser.add_argument("--quiet", action="store_true", help="Print only a compact run summary to stdout.")
    args = parser.parse_args(argv)

    payload = export_long_humanml3d_to_bvh(
        long_h5=Path(args.long_h5),
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        sequence_ids=list(args.sequence_id) or None,
        limit=args.limit,
        template_bvh=Path(args.template_bvh),
        output_fps=args.fps,
        rotation_source=args.rotation_source,
        unwrap_euler=not args.no_unwrap_euler,
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(text, encoding="utf-8")
    if args.quiet:
        print(
            json.dumps(
                {
                    "summary": args.summary,
                    "exports": len(payload["exports"]),
                    "output_dir": str(args.output_dir),
                    "rotation_source": args.rotation_source,
                },
                indent=2,
            )
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
