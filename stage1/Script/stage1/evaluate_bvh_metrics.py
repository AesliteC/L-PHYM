from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import glob
import json

import numpy as np


def load_bvh_motion(path: Path) -> tuple[np.ndarray, float]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    motion_idx = None
    for idx, line in enumerate(lines):
        if line.strip().upper() == "MOTION":
            motion_idx = idx
            break
    if motion_idx is None:
        raise ValueError(f"BVH file has no MOTION section: {path}")

    frames = None
    frame_time = None
    data_start = None
    for idx in range(motion_idx + 1, len(lines)):
        stripped = lines[idx].strip()
        lower = stripped.lower()
        if lower.startswith("frames:"):
            frames = int(stripped.split(":", 1)[1].strip())
        elif lower.startswith("frame time:"):
            frame_time = float(stripped.split(":", 1)[1].strip())
            data_start = idx + 1
            break
    if frames is None or frame_time is None or data_start is None:
        raise ValueError(f"BVH file has incomplete MOTION header: {path}")

    rows: list[list[float]] = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped:
            rows.append([float(value) for value in stripped.split()])
    motion = np.asarray(rows, dtype=np.float64)
    if motion.shape[0] < frames:
        raise ValueError(f"BVH frame count mismatch in {path}: header={frames}, rows={motion.shape[0]}")
    if motion.shape[0] > frames:
        motion = motion[:frames]
    if motion.ndim != 2 or motion.shape[1] < 6:
        raise ValueError(f"BVH motion data has unexpected shape in {path}: {motion.shape}")
    return motion, frame_time


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    valid = denom > 1e-8
    cosine = np.ones(a.shape[0], dtype=np.float64)
    cosine[valid] = np.sum(a[valid] * b[valid], axis=-1) / denom[valid]
    return np.clip(cosine, -1.0, 1.0)


def compute_bvh_metrics(
    path: Path,
    sample_stride: int = 6,
    lags: tuple[int, ...] = (5, 10, 20, 30),
    expected_min_frames: int | None = None,
) -> dict[str, object]:
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    motion, frame_time = load_bvh_motion(path)
    sampled = motion[::sample_stride]
    root = sampled[:, :3]
    pose = sampled[:, 3:]

    root_delta = np.diff(root, axis=0)
    pose_delta = np.diff(pose, axis=0)
    root_step = np.linalg.norm(root_delta, axis=-1) if len(root_delta) else np.asarray([], dtype=np.float64)
    pose_step = np.linalg.norm(pose_delta, axis=-1) if len(pose_delta) else np.asarray([], dtype=np.float64)

    root_path = float(np.sum(root_step))
    root_displacement = float(np.linalg.norm(root[-1] - root[0])) if len(root) else 0.0
    row: dict[str, object] = {
        "file": str(path),
        "label": path.stem,
        "frames": int(motion.shape[0]),
        "channels": int(motion.shape[1]),
        "frame_time": float(frame_time),
        "fps": float(1.0 / frame_time) if frame_time > 0 else 0.0,
        "duration_sec": float(motion.shape[0] * frame_time),
        "sample_stride": int(sample_stride),
        "sampled_frames": int(sampled.shape[0]),
        "root_path_length": root_path,
        "root_displacement": root_displacement,
        "root_path_to_displacement_ratio": float(root_path / max(root_displacement, 1e-8)),
        "root_step_mean": float(np.mean(root_step)) if root_step.size else 0.0,
        "root_step_std": float(np.std(root_step)) if root_step.size else 0.0,
        "pose_velocity_mean": float(np.mean(pose_step)) if pose_step.size else 0.0,
        "pose_velocity_std": float(np.std(pose_step)) if pose_step.size else 0.0,
        "pose_variance_mean": float(np.mean(np.var(pose, axis=0))) if len(pose) else 0.0,
    }
    if expected_min_frames is not None:
        row["expected_min_frames"] = int(expected_min_frames)
        row["early_stop"] = bool(motion.shape[0] < expected_min_frames)

    centered_pose = pose - np.mean(pose, axis=0, keepdims=True) if len(pose) else pose
    for lag in lags:
        if lag < 1:
            continue
        if centered_pose.shape[0] <= lag:
            row[f"lag_{lag}_mean_cosine"] = None
            row[f"lag_{lag}_repeat_fraction_0.995"] = None
            continue
        cosine = _safe_cosine(centered_pose[:-lag], centered_pose[lag:])
        row[f"lag_{lag}_mean_cosine"] = float(np.mean(cosine))
        row[f"lag_{lag}_repeat_fraction_0.995"] = float(np.mean(cosine > 0.995))
    return row


def collect_bvh_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.bvh")))
        else:
            matches = [Path(item) for item in sorted(glob.glob(raw))]
            files.extend(matches if matches else [path])
    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def evaluate_bvh_files(
    paths: list[str],
    sample_stride: int = 6,
    lags: tuple[int, ...] = (5, 10, 20, 30),
    expected_min_frames: int | None = None,
) -> dict[str, object]:
    files = collect_bvh_files(paths)
    rows = [
        compute_bvh_metrics(
            path,
            sample_stride=sample_stride,
            lags=lags,
            expected_min_frames=expected_min_frames,
        )
        for path in files
    ]
    return {
        "metric_notes": {
            "scope": "Stage1 engineering diagnostics for generated BVH files; not a replacement for paper-level FID/R-precision.",
            "paper_metrics": "MoConVQ reports Text2Motion FID and R-precision on HumanML3D using a pretrained motion feature extractor.",
            "lag_cosine": "Higher means poses at the selected temporal lag are more similar; useful as a rough repetition proxy.",
            "lag_repeat_fraction_0.995": "Fraction of sampled frame pairs whose centered pose cosine is above 0.995 at that lag.",
            "pose_velocity_mean": "Mean L2 change of BVH non-root-position channels between sampled frames; does not measure semantic correctness.",
            "early_stop": "True when frames < expected_min_frames, if expected_min_frames is provided.",
        },
        "config": {
            "inputs": paths,
            "sample_stride": sample_stride,
            "lags": list(lags),
            "expected_min_frames": expected_min_frames,
        },
        "rows": rows,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="BVH files, directories, or glob patterns")
    parser.add_argument("--sample-stride", type=int, default=6)
    parser.add_argument("--lags", default="5,10,20,30")
    parser.add_argument("--expected-min-frames", type=int, default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--quiet", action="store_true", help="Print only a compact run summary to stdout.")
    args = parser.parse_args(argv)

    lags = tuple(int(item) for item in args.lags.split(",") if item.strip())
    summary = evaluate_bvh_files(
        args.paths,
        sample_stride=args.sample_stride,
        lags=lags,
        expected_min_frames=args.expected_min_frames,
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if args.quiet:
        rows = summary.get("rows", [])
        print(
            json.dumps(
                {
                    "output": args.output,
                    "rows": len(rows),
                    "early_stop": sum(1 for row in rows if row.get("early_stop")),
                },
                indent=2,
            )
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
