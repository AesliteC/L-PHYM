from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import sys

import numpy as np

if __package__ in {None, ""}:
    repo_root = str(Path(__file__).resolve().parents[2])
    if not sys.path or sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)

from Script.stage1.bvh_to_humanml3d_features import (
    APPROXIMATION_NOTE,
    convert_bvh_to_humanml3d_features,
    resolve_humanml_data_root,
)
from Script.stage1.export_humanml3d_to_bvh import (
    DEFAULT_TEMPLATE_BVH,
    ROTATION_SOURCE_CHOICES,
    select_humanml3d_sample_ids,
    write_humanml3d_bvh,
)


def _load_mean_std(humanml_root: Path) -> tuple[np.ndarray, np.ndarray]:
    mean = np.load(humanml_root / "Mean.npy").astype(np.float32)
    std = np.load(humanml_root / "Std.npy").astype(np.float32)
    if mean.shape != (263,) or std.shape != (263,):
        raise ValueError(f"expected Mean/Std shape (263,), got {mean.shape} and {std.shape}")
    return mean, np.maximum(std, 1e-8)


def _difference_stats(diff: np.ndarray, prefix: str) -> dict[str, float]:
    if diff.size == 0:
        return {
            f"{prefix}_mae": 0.0,
            f"{prefix}_rmse": 0.0,
            f"{prefix}_p95_abs": 0.0,
            f"{prefix}_max_abs": 0.0,
        }
    abs_diff = np.abs(diff)
    return {
        f"{prefix}_mae": float(np.mean(abs_diff)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(diff * diff))),
        f"{prefix}_p95_abs": float(np.percentile(abs_diff, 95)),
        f"{prefix}_max_abs": float(np.max(abs_diff)),
    }


def compare_feature_arrays(
    original: np.ndarray,
    roundtrip: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, object]:
    frames = min(len(original), len(roundtrip))
    if original.ndim != 2 or original.shape[1] != 263:
        raise ValueError(f"expected original shape (T, 263), got {original.shape}")
    if roundtrip.ndim != 2 or roundtrip.shape[1] != 263:
        raise ValueError(f"expected roundtrip shape (T, 263), got {roundtrip.shape}")
    original_cmp = original[:frames].astype(np.float32)
    roundtrip_cmp = roundtrip[:frames].astype(np.float32)
    diff = roundtrip_cmp - original_cmp
    z_diff = (roundtrip_cmp - mean) / std - (original_cmp - mean) / std
    stats: dict[str, object] = {
        "original_feature_frames": int(len(original)),
        "roundtrip_feature_frames": int(len(roundtrip)),
        "compared_feature_frames": int(frames),
    }
    stats.update(_difference_stats(diff, "feature"))
    stats.update(_difference_stats(z_diff, "feature_z"))
    return stats


def compare_joint_arrays(original: np.ndarray, roundtrip: np.ndarray) -> dict[str, object]:
    if original.ndim != 3 or original.shape[1:] != (22, 3):
        raise ValueError(f"expected original joints shape (T, 22, 3), got {original.shape}")
    if roundtrip.ndim != 3 or roundtrip.shape[1:] != (22, 3):
        raise ValueError(f"expected roundtrip joints shape (T, 22, 3), got {roundtrip.shape}")
    frames = min(len(original), len(roundtrip))
    original_cmp = original[:frames].astype(np.float32)
    roundtrip_cmp = roundtrip[:frames].astype(np.float32)
    direct = np.linalg.norm(roundtrip_cmp - original_cmp, axis=-1)
    original_local = original_cmp - original_cmp[:, :1]
    roundtrip_local = roundtrip_cmp - roundtrip_cmp[:, :1]
    local = np.linalg.norm(roundtrip_local - original_local, axis=-1)
    root = np.linalg.norm(roundtrip_cmp[:, 0] - original_cmp[:, 0], axis=-1)
    return {
        "original_joint_frames": int(len(original)),
        "roundtrip_joint_frames": int(len(roundtrip)),
        "compared_joint_frames": int(frames),
        "joint_mpjpe_mean": float(np.mean(direct)) if direct.size else 0.0,
        "joint_mpjpe_p95": float(np.percentile(direct, 95)) if direct.size else 0.0,
        "local_joint_mpjpe_mean": float(np.mean(local)) if local.size else 0.0,
        "local_joint_mpjpe_p95": float(np.percentile(local, 95)) if local.size else 0.0,
        "root_position_error_mean": float(np.mean(root)) if root.size else 0.0,
        "root_position_error_p95": float(np.percentile(root, 95)) if root.size else 0.0,
    }


def calibrate_sample(
    sample_id: str,
    humanml_root: Path,
    output_dir: Path,
    template_bvh: Path = DEFAULT_TEMPLATE_BVH,
    rotation_source: str = "joints_ik",
    target_fps: float = 20.0,
    feet_threshold: float = 0.002,
    example_id: str = "000021",
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> dict[str, object]:
    if rotation_source not in ROTATION_SOURCE_CHOICES:
        raise ValueError(f"unknown rotation_source {rotation_source!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bvh_path = output_dir / "bvh" / f"{sample_id}.bvh"
    roundtrip_vec_path = output_dir / "roundtrip_new_joint_vecs" / f"{sample_id}.npy"
    roundtrip_joints_path = output_dir / "roundtrip_new_joints" / f"{sample_id}.npy"

    export_summary = write_humanml3d_bvh(
        sample_id=sample_id,
        humanml_root=humanml_root,
        output_bvh=bvh_path,
        template_bvh=template_bvh,
        output_fps=target_fps,
        rotation_source=rotation_source,
    )
    convert_summary = convert_bvh_to_humanml3d_features(
        bvh_path,
        humanml_data_root=humanml_root,
        output_vecs=roundtrip_vec_path,
        output_joints=roundtrip_joints_path,
        target_fps=target_fps,
        feet_threshold=feet_threshold,
        example_id=example_id,
    )

    original_vecs = np.load(humanml_root / "new_joint_vecs" / f"{sample_id}.npy").astype(np.float32)
    original_joints = np.load(humanml_root / "new_joints" / f"{sample_id}.npy").astype(np.float32)
    roundtrip_vecs = np.load(roundtrip_vec_path).astype(np.float32)
    roundtrip_joints = np.load(roundtrip_joints_path).astype(np.float32)
    if mean is None or std is None:
        mean, std = _load_mean_std(humanml_root)

    row: dict[str, object] = {
        "sample_id": sample_id,
        "bvh": str(bvh_path),
        "roundtrip_vecs": str(roundtrip_vec_path),
        "roundtrip_joints": str(roundtrip_joints_path),
        "caption": export_summary.get("caption", ""),
    }
    row.update(compare_feature_arrays(original_vecs, roundtrip_vecs, mean=mean, std=std))
    row.update(compare_joint_arrays(original_joints, roundtrip_joints))
    row["export"] = export_summary
    row["convert"] = convert_summary
    return row


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    numeric_keys = (
        "feature_mae",
        "feature_rmse",
        "feature_p95_abs",
        "feature_z_mae",
        "feature_z_rmse",
        "feature_z_p95_abs",
        "joint_mpjpe_mean",
        "joint_mpjpe_p95",
        "local_joint_mpjpe_mean",
        "local_joint_mpjpe_p95",
        "root_position_error_mean",
        "root_position_error_p95",
    )
    summary: dict[str, float] = {"samples": float(len(rows))}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[f"avg_{key}"] = float(np.mean(values))
            summary[f"max_{key}"] = float(np.max(values))
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml-root", default="/home/chenjie/cc/robotics/HumanML3D")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--split", default="", help="Optionally select samples from a HumanML3D split.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--template-bvh", default=str(DEFAULT_TEMPLATE_BVH))
    parser.add_argument("--rotation-source", choices=ROTATION_SOURCE_CHOICES, default="joints_ik")
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--feet-threshold", type=float, default=0.002)
    parser.add_argument("--example-id", default="000021")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args(argv)

    humanml_root = resolve_humanml_data_root(Path(args.humanml_root))
    sample_ids = select_humanml3d_sample_ids(
        humanml_root=humanml_root,
        sample_ids=args.sample_id,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    if not sample_ids:
        raise SystemExit("provide at least one --sample-id or --split")
    mean, std = _load_mean_std(humanml_root)
    output_dir = Path(args.output_dir)
    rows = [
        calibrate_sample(
            sample_id=sample_id,
            humanml_root=humanml_root,
            output_dir=output_dir,
            template_bvh=Path(args.template_bvh),
            rotation_source=args.rotation_source,
            target_fps=args.target_fps,
            feet_threshold=args.feet_threshold,
            example_id=args.example_id,
            mean=mean,
            std=std,
        )
        for sample_id in sample_ids
    ]
    payload = {
        "config": {
            "humanml_root": str(humanml_root),
            "sample_ids": sample_ids,
            "template_bvh": args.template_bvh,
            "rotation_source": args.rotation_source,
            "target_fps": float(args.target_fps),
            "feet_threshold": float(args.feet_threshold),
            "example_id": args.example_id,
            "output_dir": str(output_dir),
        },
        "approximation_note": APPROXIMATION_NOTE,
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    summary_path = Path(args.summary) if args.summary else output_dir / "calibration_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
