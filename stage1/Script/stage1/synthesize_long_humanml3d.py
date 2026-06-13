from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import math
import random
import re
import time

import h5py
import numpy as np

from Script.stage1.humanml3d import HumanML3DCatalog, load_humanml3d_catalog


# HumanML3D face_joint_indx is [right hip, left hip, right shoulder, left shoulder].
HIP_IDS = (2, 1)
SHOULDER_IDS = (17, 16)
FOOT_IDS = (8, 11, 7, 10)
UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
CAPTION_FILTER_MODE_CHOICES = ("none", "prefer_atomic", "atomic")
MULTI_ACTION_CAPTION_RE = re.compile(
    r"\b(and then|thens?|while|before|after|afterward|afterwards|followed by|subsequently|next)\b",
    re.IGNORECASE,
)


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
    next_joints = align_clip_to_previous(prev_joints, next_joints, blend_frames=0)
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


def _caption_text(caption: dict[str, str], fallback: str) -> str:
    text = str(caption.get("raw") or caption.get("processed") or fallback).strip()
    return text or fallback


def caption_word_count(caption: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", caption))


def caption_complexity_score(caption: str, max_caption_words: int = 0) -> int:
    """Heuristic score for whether a caption already describes multiple actions."""

    text = caption.strip()
    score = len(MULTI_ACTION_CAPTION_RE.findall(text))
    sentence_count = len([part for part in re.split(r"[.!?]+", text) if part.strip()])
    if sentence_count > 1:
        score += sentence_count - 1
    if max_caption_words > 0 and caption_word_count(text) > max_caption_words:
        score += caption_word_count(text) - max_caption_words
    return int(score)


def is_atomic_caption(caption: str, max_caption_words: int = 0) -> bool:
    return caption_complexity_score(caption, max_caption_words=max_caption_words) == 0


def _choose_caption(
    captions: list[dict[str, str]],
    fallback: str,
    filter_mode: str = "none",
    max_caption_words: int = 0,
) -> str | None:
    if filter_mode not in CAPTION_FILTER_MODE_CHOICES:
        raise ValueError(f"unknown caption filter mode: {filter_mode}")
    if not captions:
        return fallback if filter_mode != "atomic" or is_atomic_caption(fallback, max_caption_words) else None
    candidates = [_caption_text(caption, fallback) for caption in captions]
    if filter_mode == "none":
        return candidates[0]
    ranked = sorted(
        candidates,
        key=lambda text: (caption_complexity_score(text, max_caption_words), caption_word_count(text), text),
    )
    if filter_mode == "prefer_atomic":
        return ranked[0]
    for candidate in ranked:
        if is_atomic_caption(candidate, max_caption_words=max_caption_words):
            return candidate
    return None


def _load_clip(
    catalog: HumanML3DCatalog,
    sample_id: str,
    caption_filter_mode: str = "none",
    max_caption_words: int = 0,
) -> tuple[np.ndarray, np.ndarray, str]:
    sample = catalog.by_id[sample_id]
    joints = np.load(sample.joints_path).astype(np.float32)
    vecs = np.load(sample.vecs_path).astype(np.float32)
    caption = _choose_caption(
        sample.captions,
        sample_id,
        filter_mode=caption_filter_mode,
        max_caption_words=max_caption_words,
    )
    if caption is None:
        raise ValueError(f"sample {sample_id} has no caption accepted by mode {caption_filter_mode}")
    return joints, vecs, caption


def _sample_candidates(
    rng: random.Random,
    ids: list[str],
    pool_size: int,
    used_ids: set[str] | None = None,
    previous_id: str | None = None,
) -> list[str]:
    candidates = list(ids)
    if used_ids:
        unused = [sample_id for sample_id in candidates if sample_id not in used_ids]
        if unused:
            candidates = unused
    if previous_id is not None and len(candidates) > 1:
        without_previous = [sample_id for sample_id in candidates if sample_id != previous_id]
        if without_previous:
            candidates = without_previous
    count = min(max(pool_size, 1), len(candidates))
    return rng.sample(candidates, count) if count < len(candidates) else list(candidates)


def _has_valid_motion_data(catalog: HumanML3DCatalog, sample_id: str) -> bool:
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


def _has_accepted_caption(
    catalog: HumanML3DCatalog,
    sample_id: str,
    caption_filter_mode: str,
    max_caption_words: int,
) -> bool:
    sample = catalog.by_id[sample_id]
    return (
        _choose_caption(
            sample.captions,
            sample_id,
            filter_mode=caption_filter_mode,
            max_caption_words=max_caption_words,
        )
        is not None
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
    allow_forced_transitions: bool = False,
    max_sequence_attempts: int | None = None,
    drop_overlap_frames: int = 1,
    caption_filter_mode: str = "none",
    max_caption_words: int = 0,
) -> dict[str, object]:
    catalog = load_humanml3d_catalog(humanml_root)
    if split not in catalog.split_ids:
        raise ValueError(f"unknown split: {split}")
    if min_clips < 1 or max_clips < min_clips:
        raise ValueError("invalid clip count bounds")
    if drop_overlap_frames < 0:
        raise ValueError("drop_overlap_frames must be non-negative")
    if caption_filter_mode not in CAPTION_FILTER_MODE_CHOICES:
        raise ValueError(f"unknown caption filter mode: {caption_filter_mode}")
    if max_caption_words < 0:
        raise ValueError("max_caption_words must be non-negative")

    rng = random.Random(seed)
    split_ids = list(catalog.split_ids[split])
    motion_valid_ids = [sample_id for sample_id in split_ids if _has_valid_motion_data(catalog, sample_id)]
    ids = [
        sample_id
        for sample_id in motion_valid_ids
        if _has_accepted_caption(catalog, sample_id, caption_filter_mode, max_caption_words)
    ]
    filtered_invalid_clips = len(split_ids) - len(motion_valid_ids)
    filtered_caption_clips = len(motion_valid_ids) - len(ids)
    if len(ids) < max_clips:
        raise ValueError(
            f"not enough valid clips in split {split}: {len(ids)} after caption_filter_mode="
            f"{caption_filter_mode}; filtered_invalid={filtered_invalid_clips}, "
            f"filtered_caption={filtered_caption_clips}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    h5_path = output_dir / "long_sequences.h5"
    summary_path = output_dir / "summary.json"
    progress_path = output_dir / "synthesize_progress.jsonl"
    log_path = output_dir / "synthesize.log"

    rows: list[dict[str, object]] = []
    forced_count = 0
    failed_sequences = 0
    frame_lengths: list[int] = []
    skipped_scores: list[float] = []
    max_attempts = max_sequence_attempts or max(num_sequences * 20, num_sequences + 100)
    attempts = 0
    started = time.time()

    def write_event(event: str, **payload: object) -> None:
        record = {
            "event": event,
            "elapsed_sec": round(time.time() - started, 3),
            **payload,
        }
        with progress_path.open("a", encoding="utf-8") as progress_file:
            progress_file.write(json.dumps(record, ensure_ascii=False))
            progress_file.write("\n")
        message = f"[{event}] " + json.dumps(payload, ensure_ascii=False)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(message)
            log_file.write("\n")
        print(message, flush=True)

    progress_path.write_text("", encoding="utf-8")
    log_path.write_text("", encoding="utf-8")
    write_event(
        "start",
        split=split,
        num_sequences=num_sequences,
        min_clips=min_clips,
        max_clips=max_clips,
        seed=seed,
        candidate_pool=candidate_pool,
        transition_max_score=transition_max_score,
        drop_overlap_frames=drop_overlap_frames,
        allow_forced_transitions=allow_forced_transitions,
        max_attempts=max_attempts,
        caption_filter_mode=caption_filter_mode,
        max_caption_words=max_caption_words,
        motion_valid_clips=len(motion_valid_ids),
        caption_accepted_clips=len(ids),
        filtered_invalid_clips=filtered_invalid_clips,
        filtered_caption_clips=filtered_caption_clips,
    )

    with h5py.File(h5_path, "w") as h5:
        while len(rows) < num_sequences:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"failed to synthesize {num_sequences} sequences after {max_attempts} attempts; "
                    f"completed {len(rows)}, failed {failed_sequences}. Increase --candidate-pool, "
                    "relax --transition-max-score, reduce clip count, or pass --allow-forced-transitions."
                )
            clip_count = rng.randint(min_clips, max_clips)
            sample_ids: list[str] = []
            clip_captions: list[str] = []
            transition_scores: list[float] = []
            transition_forced: list[bool] = []
            source_paths: list[str | None] = []
            start_frames: list[int | None] = []
            end_frames: list[int | None] = []
            dropped_prefix_frames: list[int] = []
            clip_caption_complexity: list[int] = []
            clip_caption_is_atomic: list[bool] = []
            clip_boundaries: list[tuple[int, int]] = []

            first_id = rng.choice(ids)
            first_joints, first_vecs, first_caption = _load_clip(
                catalog,
                first_id,
                caption_filter_mode=caption_filter_mode,
                max_caption_words=max_caption_words,
            )
            merged_joints = [first_joints]
            merged_vecs = [first_vecs]
            sample_ids.append(first_id)
            clip_captions.append(first_caption)
            clip_caption_complexity.append(caption_complexity_score(first_caption, max_caption_words))
            clip_caption_is_atomic.append(is_atomic_caption(first_caption, max_caption_words))
            sample = catalog.by_id[first_id]
            source_paths.append(str(sample.source_path) if sample.source_path is not None else None)
            start_frames.append(sample.start_frame)
            end_frames.append(sample.end_frame)
            dropped_prefix_frames.append(0)
            clip_boundaries.append((0, len(first_joints)))

            sequence_failed = False
            for _ in range(1, clip_count):
                prev = merged_joints[-1]
                best_id = None
                best_score = None
                best_parts = None
                for candidate_id in _sample_candidates(
                    rng,
                    ids,
                    candidate_pool,
                    used_ids=set(sample_ids),
                    previous_id=sample_ids[-1],
                ):
                    candidate_joints, _, _ = _load_clip(
                        catalog,
                        candidate_id,
                        caption_filter_mode=caption_filter_mode,
                        max_caption_words=max_caption_words,
                    )
                    parts = transition_score(prev, candidate_joints)
                    if best_score is None or parts["total"] < best_score:
                        best_id = candidate_id
                        best_score = parts["total"]
                        best_parts = parts

                if best_id is None or best_score is None or best_parts is None:
                    raise RuntimeError("failed to select transition candidate")
                if best_score > transition_max_score and not allow_forced_transitions:
                    failed_sequences += 1
                    skipped_scores.append(float(best_score))
                    write_event(
                        "skip_sequence",
                        attempt=attempts,
                        completed=len(rows),
                        failed_sequences=failed_sequences,
                        clip_count=clip_count,
                        current_samples=sample_ids,
                        candidate_id=best_id,
                        best_score=float(best_score),
                        best_parts=best_parts,
                    )
                    sequence_failed = True
                    break

                candidate_joints, candidate_vecs, candidate_caption = _load_clip(
                    catalog,
                    best_id,
                    caption_filter_mode=caption_filter_mode,
                    max_caption_words=max_caption_words,
                )
                aligned_joints = align_clip_to_previous(prev, candidate_joints, blend_frames)
                drop_count = min(int(drop_overlap_frames), max(len(aligned_joints) - 2, 0))
                aligned_joints_to_store = aligned_joints[drop_count:]
                candidate_vecs_to_store = candidate_vecs[drop_count:]
                if len(aligned_joints_to_store) < 2 or len(candidate_vecs_to_store) < 2:
                    failed_sequences += 1
                    write_event(
                        "skip_sequence",
                        attempt=attempts,
                        completed=len(rows),
                        failed_sequences=failed_sequences,
                        clip_count=clip_count,
                        current_samples=sample_ids,
                        candidate_id=best_id,
                        best_score=float(best_score),
                        reason="clip too short after dropping overlap frames",
                        dropped_prefix_frames=drop_count,
                    )
                    sequence_failed = True
                    break
                start = sum(len(part) for part in merged_joints)
                end = start + len(aligned_joints_to_store)
                forced = best_score > transition_max_score
                forced_count += int(forced)

                merged_joints.append(aligned_joints_to_store)
                merged_vecs.append(candidate_vecs_to_store)
                sample_ids.append(best_id)
                clip_captions.append(candidate_caption)
                clip_caption_complexity.append(caption_complexity_score(candidate_caption, max_caption_words))
                clip_caption_is_atomic.append(is_atomic_caption(candidate_caption, max_caption_words))
                transition_scores.append(float(best_score))
                transition_forced.append(bool(forced))
                sample = catalog.by_id[best_id]
                source_paths.append(str(sample.source_path) if sample.source_path is not None else None)
                start_frames.append(sample.start_frame)
                end_frames.append(sample.end_frame)
                dropped_prefix_frames.append(drop_count)
                clip_boundaries.append((start, end))

            if sequence_failed:
                continue

            seq_idx = len(rows)
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
                "dropped_prefix_frames": dropped_prefix_frames,
                "clip_caption_complexity": clip_caption_complexity,
                "clip_caption_is_atomic": clip_caption_is_atomic,
            }
            rows.append(row)

            group = h5.create_group(sequence_id)
            group.create_dataset("joints_22", data=joints_22, compression="gzip")
            group.create_dataset("joint_vecs_263", data=joint_vecs_263, compression="gzip")
            group.create_dataset("clip_boundaries", data=np.asarray(clip_boundaries, dtype=np.int32))
            group.create_dataset("transition_scores", data=np.asarray(transition_scores, dtype=np.float32))
            group.create_dataset("dropped_prefix_frames", data=np.asarray(dropped_prefix_frames, dtype=np.int32))
            group.attrs["caption"] = caption
            group.attrs["sample_ids"] = ",".join(sample_ids)
            group.attrs["split"] = split
            if len(rows) <= 5 or len(rows) % 50 == 0 or len(rows) == num_sequences:
                write_event(
                    "sequence_written",
                    sequence_id=sequence_id,
                    completed=len(rows),
                    attempts=attempts,
                    failed_sequences=failed_sequences,
                    clip_count=len(sample_ids),
                    frame_count=len(joints_22),
                    max_transition_score=max(transition_scores) if transition_scores else 0.0,
                    sample_ids=sample_ids,
                )

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    summary = {
        "num_sequences": len(rows),
        "transitions": int(sum(len(row["transition_scores"]) for row in rows)),
        "avg_clips": float(np.mean([len(row["sample_ids"]) for row in rows])) if rows else 0.0,
        "avg_frames": float(np.mean(frame_lengths)) if frame_lengths else 0.0,
        "forced_transitions": forced_count,
        "duplicate_sequences": int(
            sum(1 for row in rows if len(set(row["sample_ids"])) < len(row["sample_ids"]))
        ),
        "failed_sequences": failed_sequences,
        "attempted_sequences": attempts,
        "skipped_transition_score_mean": float(np.mean(skipped_scores)) if skipped_scores else 0.0,
        "skipped_transition_score_max": float(np.max(skipped_scores)) if skipped_scores else 0.0,
        "filtered_invalid_clips": filtered_invalid_clips,
        "filtered_caption_clips": filtered_caption_clips,
        "caption_filter_mode": caption_filter_mode,
        "max_caption_words": max_caption_words,
        "non_atomic_clip_captions": int(
            sum(
                1
                for row in rows
                for is_atomic in row.get("clip_caption_is_atomic", [])
                if not is_atomic
            )
        ),
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
            "drop_overlap_frames": drop_overlap_frames,
            "caption_joiner": caption_joiner,
            "caption_filter_mode": caption_filter_mode,
            "max_caption_words": max_caption_words,
            "allow_forced_transitions": allow_forced_transitions,
            "max_sequence_attempts": max_sequence_attempts,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_event("summary", **summary)
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
    parser.add_argument("--drop-overlap-frames", type=int, default=1)
    parser.add_argument("--caption-joiner", default=" then ")
    parser.add_argument("--caption-filter-mode", choices=CAPTION_FILTER_MODE_CHOICES, default="none")
    parser.add_argument("--max-caption-words", type=int, default=0)
    parser.add_argument("--output-dir", default="stage1_artifacts/long_humanml3d/train")
    parser.add_argument("--allow-forced-transitions", action="store_true")
    parser.add_argument("--max-sequence-attempts", type=int, default=None)
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
        drop_overlap_frames=args.drop_overlap_frames,
        caption_joiner=args.caption_joiner,
        caption_filter_mode=args.caption_filter_mode,
        max_caption_words=args.max_caption_words,
        output_dir=Path(args.output_dir),
        allow_forced_transitions=args.allow_forced_transitions,
        max_sequence_attempts=args.max_sequence_attempts,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
