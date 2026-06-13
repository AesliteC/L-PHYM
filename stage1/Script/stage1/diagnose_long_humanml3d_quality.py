from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import math

import h5py
import numpy as np

from Script.stage1.real_moconvq_cache import load_manifest
from Script.stage1.synthesize_long_humanml3d import FOOT_IDS, estimate_facing_yaw


def _angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def transition_boundary_metrics(joints: np.ndarray, boundary: int) -> dict[str, float]:
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"expected joints shape (T, 22, 3), got {joints.shape}")
    if boundary <= 1 or boundary + 1 >= len(joints):
        raise ValueError(f"boundary {boundary} does not have enough neighbor frames for sequence length {len(joints)}")

    prev_last = joints[boundary - 1]
    prev_prev = joints[boundary - 2]
    next_first = joints[boundary]
    next_second = joints[boundary + 1]
    prev_root_vel = prev_last[0] - prev_prev[0]
    next_root_vel = next_second[0] - next_first[0]
    prev_foot_vel = prev_last[list(FOOT_IDS)] - prev_prev[list(FOOT_IDS)]
    next_foot_vel = next_second[list(FOOT_IDS)] - next_first[list(FOOT_IDS)]
    return {
        "root_gap": float(np.linalg.norm(prev_last[0] - next_first[0])),
        "root_velocity_gap": float(np.linalg.norm(prev_root_vel - next_root_vel)),
        "yaw_gap": abs(_angle_diff(estimate_facing_yaw(prev_last), estimate_facing_yaw(next_first))),
        "foot_height_gap": float(np.mean(np.abs(prev_last[list(FOOT_IDS), 1] - next_first[list(FOOT_IDS), 1]))),
        "foot_velocity_gap": float(np.mean(np.linalg.norm(prev_foot_vel - next_foot_vel, axis=-1))),
    }


def diagnose_long_humanml3d_quality(
    long_h5_path: Path,
    manifest_path: Path,
    output_json: Path,
    transition_jsonl: Path | None = None,
    root_gap_warn: float = 0.20,
    root_velocity_warn: float = 0.12,
    yaw_warn_rad: float = 0.75,
    foot_velocity_warn: float = 0.12,
    fps: int = 20,
) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    transition_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    root_gaps: list[float] = []
    root_velocity_gaps: list[float] = []
    yaw_gaps: list[float] = []
    foot_velocity_gaps: list[float] = []
    foot_height_gaps: list[float] = []
    forced_transitions = 0
    duplicate_sequences = 0
    repeated_caption_sequences = 0
    bad_transition_count = 0
    frame_counts: list[int] = []
    clip_counts: list[int] = []

    with h5py.File(long_h5_path, "r") as h5:
        for sequence_id in h5.keys():
            row = manifest.get(sequence_id, {})
            group = h5[sequence_id]
            joints = group["joints_22"][:]
            boundaries = row.get("clip_boundaries")
            if not boundaries and "clip_boundaries" in group:
                boundaries = group["clip_boundaries"][:].tolist()
            boundaries = [(int(start), int(end)) for start, end in (boundaries or [])]
            sample_ids = [str(item) for item in row.get("sample_ids", [])]
            clip_captions = [str(item) for item in row.get("clip_captions", [])]
            transition_scores = [float(item) for item in row.get("transition_scores", [])]
            transition_forced = [bool(item) for item in row.get("transition_forced", [])]

            frame_counts.append(int(len(joints)))
            clip_counts.append(int(len(boundaries)))
            duplicate_sequences += int(bool(sample_ids) and len(set(sample_ids)) < len(sample_ids))
            repeated_caption_sequences += int(bool(clip_captions) and len(set(clip_captions)) < len(clip_captions))

            sequence_bad = 0
            for transition_idx in range(max(len(boundaries) - 1, 0)):
                boundary = boundaries[transition_idx][1]
                try:
                    metrics = transition_boundary_metrics(joints, boundary)
                except ValueError as exc:
                    metrics = {
                        "root_gap": float("inf"),
                        "root_velocity_gap": float("inf"),
                        "yaw_gap": float("inf"),
                        "foot_height_gap": float("inf"),
                        "foot_velocity_gap": float("inf"),
                        "error": str(exc),
                    }
                root_gaps.append(float(metrics["root_gap"]))
                root_velocity_gaps.append(float(metrics["root_velocity_gap"]))
                yaw_gaps.append(float(metrics["yaw_gap"]))
                foot_height_gaps.append(float(metrics["foot_height_gap"]))
                foot_velocity_gaps.append(float(metrics["foot_velocity_gap"]))
                forced = bool(transition_forced[transition_idx]) if transition_idx < len(transition_forced) else False
                forced_transitions += int(forced)
                bad = (
                    forced
                    or float(metrics["root_gap"]) > root_gap_warn
                    or float(metrics["root_velocity_gap"]) > root_velocity_warn
                    or float(metrics["yaw_gap"]) > yaw_warn_rad
                    or float(metrics["foot_velocity_gap"]) > foot_velocity_warn
                )
                bad_transition_count += int(bad)
                sequence_bad += int(bad)
                transition_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "transition_idx": transition_idx,
                        "boundary": boundary,
                        "sample_pair": sample_ids[transition_idx : transition_idx + 2],
                        "caption_pair": clip_captions[transition_idx : transition_idx + 2],
                        "recorded_transition_score": (
                            transition_scores[transition_idx] if transition_idx < len(transition_scores) else None
                        ),
                        "forced": forced,
                        "bad": bad,
                        **metrics,
                    }
                )
            sequence_rows.append(
                {
                    "sequence_id": sequence_id,
                    "frames": int(len(joints)),
                    "duration_sec": float(len(joints) / max(fps, 1)),
                    "clips": int(len(boundaries)),
                    "bad_transitions": sequence_bad,
                    "duplicate_sample_ids": bool(sample_ids) and len(set(sample_ids)) < len(sample_ids),
                    "repeated_clip_captions": bool(clip_captions) and len(set(clip_captions)) < len(clip_captions),
                }
            )

    transitions = len(transition_rows)
    summary = {
        "long_h5": str(long_h5_path),
        "manifest": str(manifest_path),
        "sequences": len(sequence_rows),
        "transitions": transitions,
        "avg_clips": float(np.mean(clip_counts)) if clip_counts else 0.0,
        "avg_frames": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "avg_duration_sec": float(np.mean(frame_counts) / max(fps, 1)) if frame_counts else 0.0,
        "forced_transitions": forced_transitions,
        "bad_transition_count": bad_transition_count,
        "bad_transition_rate": float(bad_transition_count / max(transitions, 1)),
        "duplicate_sequences": duplicate_sequences,
        "repeated_caption_sequences": repeated_caption_sequences,
        "thresholds": {
            "root_gap_warn": root_gap_warn,
            "root_velocity_warn": root_velocity_warn,
            "yaw_warn_rad": yaw_warn_rad,
            "foot_velocity_warn": foot_velocity_warn,
        },
        "metrics": {
            "root_gap": {
                "mean": float(np.mean(root_gaps)) if root_gaps else 0.0,
                "p95": _percentile(root_gaps, 95),
                "max": float(np.max(root_gaps)) if root_gaps else 0.0,
            },
            "root_velocity_gap": {
                "mean": float(np.mean(root_velocity_gaps)) if root_velocity_gaps else 0.0,
                "p95": _percentile(root_velocity_gaps, 95),
                "max": float(np.max(root_velocity_gaps)) if root_velocity_gaps else 0.0,
            },
            "yaw_gap": {
                "mean": float(np.mean(yaw_gaps)) if yaw_gaps else 0.0,
                "p95": _percentile(yaw_gaps, 95),
                "max": float(np.max(yaw_gaps)) if yaw_gaps else 0.0,
            },
            "foot_height_gap": {
                "mean": float(np.mean(foot_height_gaps)) if foot_height_gaps else 0.0,
                "p95": _percentile(foot_height_gaps, 95),
                "max": float(np.max(foot_height_gaps)) if foot_height_gaps else 0.0,
            },
            "foot_velocity_gap": {
                "mean": float(np.mean(foot_velocity_gaps)) if foot_velocity_gaps else 0.0,
                "p95": _percentile(foot_velocity_gaps, 95),
                "max": float(np.max(foot_velocity_gaps)) if foot_velocity_gaps else 0.0,
            },
        },
        "sequence_examples": sequence_rows[:20],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if transition_jsonl is not None:
        transition_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with transition_jsonl.open("w", encoding="utf-8") as f:
            for transition in transition_rows:
                f.write(json.dumps(transition, ensure_ascii=False))
                f.write("\n")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-h5", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--transition-jsonl", default=None)
    parser.add_argument("--root-gap-warn", type=float, default=0.20)
    parser.add_argument("--root-velocity-warn", type=float, default=0.12)
    parser.add_argument("--yaw-warn-rad", type=float, default=0.75)
    parser.add_argument("--foot-velocity-warn", type=float, default=0.12)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args(argv)
    summary = diagnose_long_humanml3d_quality(
        long_h5_path=Path(args.long_h5),
        manifest_path=Path(args.manifest),
        output_json=Path(args.output_json),
        transition_jsonl=Path(args.transition_jsonl) if args.transition_jsonl else None,
        root_gap_warn=args.root_gap_warn,
        root_velocity_warn=args.root_velocity_warn,
        yaw_warn_rad=args.yaw_warn_rad,
        foot_velocity_warn=args.foot_velocity_warn,
        fps=args.fps,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
