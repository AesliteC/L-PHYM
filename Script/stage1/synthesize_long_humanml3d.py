from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import math
import random

import h5py
import numpy as np

from Script.stage1.humanml3d import HumanML3DCatalog, load_humanml3d_catalog


# HumanML3D face_joint_indx is [right hip, left hip, right shoulder, left shoulder].
HIP_IDS = (2, 1)
SHOULDER_IDS = (17, 16)
FOOT_IDS = (8, 11, 7, 10)
UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def estimate_facing_yaw(joints_frame: np.ndarray) -> float:
    right_axis = 0.5 * (
        joints_frame[HIP_IDS[0]] - joints_frame[HIP_IDS[1]]
        + joints_frame[SHOULDER_IDS[0]] - joints_frame[SHOULDER_IDS[1]]
    )
    right_axis[1] = 0.0
    right_axis = _normalize(right_axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    forward = np.cross(right_axis, UP)
    forward[1] = 0.0
    forward = _normalize(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return float(math.atan2(float(forward[0]), float(forward[2])))


def _angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def transition_score(prev_joints: np.ndarray, next_joints: np.ndarray) -> dict[str, float]:
    if len(prev_joints) < 2 or len(next_joints) < 2:
        raise ValueError("transition scoring needs at least two frames per clip")
    prev_root = prev_joints[-1, 0]
    next_root = next_joints[0, 0]
    prev_vel = prev_joints[-1, 0] - prev_joints[-2, 0]
    next_vel = next_joints[1, 0] - next_joints[0, 0]
    yaw_delta = abs(_angle_diff(estimate_facing_yaw(prev_joints[-1]), estimate_facing_yaw(next_joints[0])))
    prev_foot = prev_joints[-1, FOOT_IDS, :]
    next_foot = next_joints[0, FOOT_IDS, :]
    prev_foot_vel = prev_joints[-1, FOOT_IDS, :] - prev_joints[-2, FOOT_IDS, :]
    next_foot_vel = next_joints[1, FOOT_IDS, :] - next_joints[0, FOOT_IDS, :]

    parts = {
        "root_position": float(np.linalg.norm(prev_root - next_root)),
        "root_velocity": float(np.linalg.norm(prev_vel - next_vel)),
        "yaw": float(yaw_delta / math.pi),
        "foot_height": float(np.mean(np.abs(prev_foot[:, 1] - next_foot[:, 1]))),
        "foot_velocity": float(np.mean(np.linalg.norm(prev_foot_vel - next_foot_vel, axis=-1))),
    }
    parts["total"] = (
        parts["root_position"]
        + 0.5 * parts["root_velocity"]
        + parts["yaw"]
        + parts["foot_height"]
        + 0.5 * parts["foot_velocity"]
    )
    return parts


def align_clip_to_previous(prev_joints: np.ndarray, next_joints: np.ndarray, blend_frames: int) -> np.ndarray:
    target_root = prev_joints[-1, 0].astype(np.float32)
    source_root = next_joints[0, 0].astype(np.float32)
    yaw_delta = _angle_diff(estimate_facing_yaw(prev_joints[-1]), estimate_facing_yaw(next_joints[0]))
    rot = _rotation_y(yaw_delta)
    aligned = (next_joints - source_root) @ rot.T + target_root

    blend_count = min(max(int(blend_frames), 0), len(aligned))
    if blend_count > 0:
        original_roots = aligned[:blend_count, 0].copy()
        for idx in range(blend_count):
            alpha = float(idx + 1) / float(blend_count + 1)
            desired_root = (1.0 - alpha) * target_root + alpha * original_roots[idx]
            aligned[idx] += desired_root - aligned[idx, 0]
    return aligned.astype(np.float32)


def _choose_caption(captions: list[dict[str, str]], fallback: str) -> str:
    if not captions:
        return fallback
    return str(captions[0].get("raw") or captions[0].get("processed") or fallback)


def _load_clip(catalog: HumanML3DCatalog, sample_id: str) -> tuple[np.ndarray, np.ndarray, str]:
    sample = catalog.by_id[sample_id]
    joints = np.load(sample.joints_path).astype(np.float32)
    vecs = np.load(sample.vecs_path).astype(np.float32)
    caption = _choose_caption(sample.captions, sample_id)
    return joints, vecs, caption


def _sample_candidates(rng: random.Random, ids: list[str], pool_size: int) -> list[str]:
    count = min(max(pool_size, 1), len(ids))
    return rng.sample(ids, count) if count < len(ids) else list(ids)


def _is_valid_clip(catalog: HumanML3DCatalog, sample_id: str) -> bool:
    sample = catalog.by_id[sample_id]
    joints = np.load(sample.joints_path, mmap_mode="r")
    vecs = np.load(sample.vecs_path, mmap_mode="r")
    return (
        joints.ndim == 3
        and joints.shape[1:] == (22, 3)
        and joints.shape[0] >= 2
        and vecs.ndim == 2
        and vecs.shape[1] == 263
        and vecs.shape[0] >= 2
    )


def synthesize_dataset(
    humanml_root: Path,
    split: str,
    num_sequences: int,
    min_clips: int,
    max_clips: int,
    seed: int,
    candidate_pool: int,
    transition_max_score: float,
    blend_frames: int,
    caption_joiner: str,
    output_dir: Path,
) -> dict[str, object]:
    catalog = load_humanml3d_catalog(humanml_root)
    if split not in catalog.split_ids:
        raise ValueError(f"unknown split: {split}")
    if min_clips < 1 or max_clips < min_clips:
        raise ValueError("invalid clip count bounds")

    rng = random.Random(seed)
    split_ids = list(catalog.split_ids[split])
    ids = [sample_id for sample_id in split_ids if _is_valid_clip(catalog, sample_id)]
    filtered_invalid_clips = len(split_ids) - len(ids)
    if len(ids) < max_clips:
        raise ValueError(f"not enough valid clips in split {split}: {len(ids)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    h5_path = output_dir / "long_sequences.h5"
    summary_path = output_dir / "summary.json"

    rows: list[dict[str, object]] = []
    forced_count = 0
    frame_lengths: list[int] = []

    with h5py.File(h5_path, "w") as h5:
        for seq_idx in range(num_sequences):
            clip_count = rng.randint(min_clips, max_clips)
            sample_ids: list[str] = []
            clip_captions: list[str] = []
            transition_scores: list[float] = []
            transition_forced: list[bool] = []
            source_paths: list[str | None] = []
            start_frames: list[int | None] = []
            end_frames: list[int | None] = []
            clip_boundaries: list[tuple[int, int]] = []

            first_id = rng.choice(ids)
            first_joints, first_vecs, first_caption = _load_clip(catalog, first_id)
            merged_joints = [first_joints]
            merged_vecs = [first_vecs]
            sample_ids.append(first_id)
            clip_captions.append(first_caption)
            sample = catalog.by_id[first_id]
            source_paths.append(str(sample.source_path) if sample.source_path is not None else None)
            start_frames.append(sample.start_frame)
            end_frames.append(sample.end_frame)
            clip_boundaries.append((0, len(first_joints)))

            for _ in range(1, clip_count):
                prev = merged_joints[-1]
                best_id = None
                best_score = None
                best_parts = None
                for candidate_id in _sample_candidates(rng, ids, candidate_pool):
                    candidate_joints, _, _ = _load_clip(catalog, candidate_id)
                    parts = transition_score(prev, candidate_joints)
                    if best_score is None or parts["total"] < best_score:
                        best_id = candidate_id
                        best_score = parts["total"]
                        best_parts = parts

                if best_id is None or best_score is None or best_parts is None:
                    raise RuntimeError("failed to select transition candidate")

                candidate_joints, candidate_vecs, candidate_caption = _load_clip(catalog, best_id)
                aligned_joints = align_clip_to_previous(prev, candidate_joints, blend_frames)
                start = sum(len(part) for part in merged_joints)
                end = start + len(aligned_joints)
                forced = best_score > transition_max_score
                forced_count += int(forced)

                merged_joints.append(aligned_joints)
                merged_vecs.append(candidate_vecs)
                sample_ids.append(best_id)
                clip_captions.append(candidate_caption)
                transition_scores.append(float(best_score))
                transition_forced.append(bool(forced))
                sample = catalog.by_id[best_id]
                source_paths.append(str(sample.source_path) if sample.source_path is not None else None)
                start_frames.append(sample.start_frame)
                end_frames.append(sample.end_frame)
                clip_boundaries.append((start, end))

            sequence_id = f"{split}_{seq_idx:06d}"
            joints_22 = np.concatenate(merged_joints, axis=0).astype(np.float32)
            joint_vecs_263 = np.concatenate(merged_vecs, axis=0).astype(np.float32)
            frame_lengths.append(len(joints_22))
            caption = caption_joiner.join(clip_captions)
            row = {
                "sequence_id": sequence_id,
                "split": split,
                "sample_ids": sample_ids,
                "caption": caption,
                "clip_captions": clip_captions,
                "clip_boundaries": clip_boundaries,
                "transition_scores": transition_scores,
                "transition_forced": transition_forced,
                "source_paths": source_paths,
                "start_frames": start_frames,
                "end_frames": end_frames,
            }
            rows.append(row)

            group = h5.create_group(sequence_id)
            group.create_dataset("joints_22", data=joints_22, compression="gzip")
            group.create_dataset("joint_vecs_263", data=joint_vecs_263, compression="gzip")
            group.create_dataset("clip_boundaries", data=np.asarray(clip_boundaries, dtype=np.int32))
            group.create_dataset("transition_scores", data=np.asarray(transition_scores, dtype=np.float32))
            group.attrs["caption"] = caption
            group.attrs["sample_ids"] = ",".join(sample_ids)
            group.attrs["split"] = split

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    summary = {
        "num_sequences": len(rows),
        "avg_clips": float(np.mean([len(row["sample_ids"]) for row in rows])) if rows else 0.0,
        "avg_frames": float(np.mean(frame_lengths)) if frame_lengths else 0.0,
        "forced_transitions": forced_count,
        "failed_sequences": 0,
        "filtered_invalid_clips": filtered_invalid_clips,
        "config": {
            "humanml_root": str(humanml_root),
            "split": split,
            "num_sequences": num_sequences,
            "min_clips": min_clips,
            "max_clips": max_clips,
            "seed": seed,
            "candidate_pool": candidate_pool,
            "transition_max_score": transition_max_score,
            "blend_frames": blend_frames,
            "caption_joiner": caption_joiner,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml-root", default="../HumanML3D/HumanML3D")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--min-clips", type=int, default=2)
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-pool", type=int, default=256)
    parser.add_argument("--transition-max-score", type=float, default=0.35)
    parser.add_argument("--blend-frames", type=int, default=5)
    parser.add_argument("--caption-joiner", default=" then ")
    parser.add_argument("--output-dir", default="stage1_artifacts/long_humanml3d/train")
    args = parser.parse_args(argv)

    summary = synthesize_dataset(
        humanml_root=Path(args.humanml_root),
        split=args.split,
        num_sequences=args.num_sequences,
        min_clips=args.min_clips,
        max_clips=args.max_clips,
        seed=args.seed,
        candidate_pool=args.candidate_pool,
        transition_max_score=args.transition_max_score,
        blend_frames=args.blend_frames,
        caption_joiner=args.caption_joiner,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
