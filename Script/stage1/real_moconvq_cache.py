from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
import argparse
import json
import traceback

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation


MOCONVQ_BODY_NAMES = (
    "pelvis",
    "lowerBack",
    "torso",
    "rUpperLeg",
    "lUpperLeg",
    "rLowerLeg",
    "lLowerLeg",
    "rFoot",
    "lFoot",
    "rToes",
    "lToes",
    "head",
    "rClavicle",
    "lClavicle",
    "rUpperArm",
    "lUpperArm",
    "rLowerArm",
    "lLowerArm",
    "rHand",
    "lHand",
)

DEFAULT_TEXT_GPT_BLOCK_SIZE = 52

MOCONVQ_PARENT = np.asarray(
    [-1, 0, 1, 0, 0, 3, 4, 5, 6, 7, 8, 2, 2, 2, 12, 13, 14, 15, 16, 17],
    dtype=np.int64,
)

HUMANML3D_TO_MOCONVQ = np.asarray(
    [0, 3, 6, 2, 1, 5, 4, 8, 7, 11, 10, 15, 14, 13, 17, 16, 19, 18, 21, 20],
    dtype=np.int64,
)

UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _frame_axes(joints_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right = 0.5 * ((joints_frame[1] - joints_frame[2]) + (joints_frame[13] - joints_frame[14]))
    right[1] = 0.0
    right = _normalize(right, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    forward = np.cross(right, UP)
    forward[1] = 0.0
    forward = _normalize(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right = _normalize(np.cross(UP, forward), right)
    return right, UP.copy(), forward


def _quat_from_axes(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    mat = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float64)
    if np.linalg.det(mat) < 0:
        mat[:, 0] *= -1.0
    try:
        quat = Rotation.from_matrix(mat).as_quat()
    except ValueError:
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat.astype(np.float32)


def _bone_quat(body_id: int, positions: np.ndarray, frame_axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    right, _, forward = frame_axes
    children = np.where(MOCONVQ_PARENT == body_id)[0]
    if body_id in (0, 1, 2, 11, 12, 13):
        return _quat_from_axes(right, UP, forward)
    if len(children) > 0:
        direction = np.mean(positions[children] - positions[body_id], axis=0)
    else:
        parent = MOCONVQ_PARENT[body_id]
        direction = positions[body_id] - positions[parent]
    y_axis = _normalize(direction, UP)
    z_axis = forward - y_axis * float(np.dot(forward, y_axis))
    z_axis = _normalize(z_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    x_axis = _normalize(np.cross(y_axis, z_axis), right)
    z_axis = _normalize(np.cross(x_axis, y_axis), z_axis)
    return _quat_from_axes(x_axis, y_axis, z_axis)


def _ensure_quat_continuity(quats: np.ndarray) -> np.ndarray:
    fixed = quats.copy()
    for t in range(1, len(fixed)):
        dots = np.sum(fixed[t - 1] * fixed[t], axis=-1)
        fixed[t, dots < 0.0] *= -1.0
    return fixed


def _linear_velocity(values: np.ndarray, fps: int) -> np.ndarray:
    velocity = np.zeros_like(values, dtype=np.float32)
    if len(values) < 2:
        return velocity
    velocity[1:] = (values[1:] - values[:-1]) * float(fps)
    velocity[0] = velocity[1]
    return velocity


def _angular_velocity(quats: np.ndarray, fps: int) -> np.ndarray:
    avel = np.zeros(quats.shape[:-1] + (3,), dtype=np.float32)
    if len(quats) < 2:
        return avel
    for t in range(1, len(quats)):
        delta = Rotation.from_quat(quats[t].reshape(-1, 4)) * Rotation.from_quat(quats[t - 1].reshape(-1, 4)).inv()
        avel[t] = delta.as_rotvec().reshape(quats.shape[1], 3) * float(fps)
    avel[0] = avel[1]
    return avel.astype(np.float32)


def humanml3d_joints_to_moconvq_state(joints_22: np.ndarray, fps: int = 20) -> np.ndarray:
    if joints_22.ndim != 3 or joints_22.shape[1:] != (22, 3):
        raise ValueError(f"expected joints shape (T, 22, 3), got {joints_22.shape}")
    positions = joints_22[:, HUMANML3D_TO_MOCONVQ, :].astype(np.float32)
    quats = np.zeros((positions.shape[0], 20, 4), dtype=np.float32)
    for t in range(positions.shape[0]):
        axes = _frame_axes(joints_22[t].astype(np.float32))
        for body_id in range(20):
            quats[t, body_id] = _bone_quat(body_id, positions[t], axes)
    quats = _ensure_quat_continuity(quats)
    linear_vel = _linear_velocity(positions, fps=fps)
    angular_vel = _angular_velocity(quats, fps=fps)
    return np.concatenate([positions, quats, linear_vel, angular_vel], axis=-1).astype(np.float32)


def moconvq_state_to_observation(state: np.ndarray) -> np.ndarray:
    from MoConVQCore.Utils.motion_utils import state2ob

    with torch.no_grad():
        observation = state2ob(torch.as_tensor(state, dtype=torch.float32))
    observation_np = observation.detach().cpu().numpy().astype(np.float32)
    if observation_np.ndim == 1:
        observation_np = observation_np[None, :]
    if observation_np.shape[-1] != 323:
        raise ValueError(f"expected observation dim 323, got {observation_np.shape}")
    return observation_np


def load_manifest(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row["sequence_id"])] = row
    return rows


def _indices_to_time_depth(indexs: torch.Tensor | np.ndarray, rvq_depth: int) -> np.ndarray:
    arr = indexs.detach().cpu().numpy() if isinstance(indexs, torch.Tensor) else np.asarray(indexs)
    if arr.ndim == 3:
        arr = arr[:, 0, :].T
    elif arr.ndim == 2 and arr.shape[0] >= rvq_depth:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] < rvq_depth:
        raise ValueError(f"unexpected RVQ index shape: {arr.shape}")
    return arr[:, :rvq_depth].astype(np.int64)


def encode_observation_with_agent(agent, observation: np.ndarray, rvq_depth: int) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        info = agent.encode_seq_all(None, observation)
    latent = info["latent_vq"].detach().cpu().numpy().astype(np.float32)
    if latent.ndim == 3:
        latent = latent[0]
    indices = _indices_to_time_depth(info["indexs"], rvq_depth=rvq_depth)
    if latent.shape[0] != indices.shape[0]:
        length = min(latent.shape[0], indices.shape[0])
        latent = latent[:length]
        indices = indices[:length]
    if latent.shape[-1] != 768:
        raise ValueError(f"expected latent dim 768, got {latent.shape}")
    return latent, indices


def make_windows(
    latents: np.ndarray,
    indices: np.ndarray,
    window_size: int,
    window_stride: int,
    pad_index: int = 513,
    include_tail: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    if len(latents) != len(indices):
        raise ValueError("latent/index length mismatch")
    if window_size < 1 or window_stride < 1:
        raise ValueError("window size and stride must be positive")
    length = len(latents)
    if length == 0:
        return []
    starts = [0] if length <= window_size else list(range(0, length - window_size + 1, window_stride))
    if include_tail and length > window_size and starts[-1] != length - window_size:
        starts.append(length - window_size)
    windows = []
    for start in starts:
        end = min(start + window_size, length)
        latent_window = np.zeros((window_size, latents.shape[-1]), dtype=np.float32)
        index_window = np.full((window_size, indices.shape[-1]), pad_index, dtype=np.int64)
        valid = end - start
        latent_window[:valid] = latents[start:end]
        index_window[:valid] = indices[start:end]
        windows.append((latent_window, index_window, (start, end)))
    return windows


def make_clip_aligned_windows(
    latents: np.ndarray,
    indices: np.ndarray,
    window_size: int,
    window_stride: int,
    clip_boundaries: list[tuple[int, int]],
    pad_index: int = 513,
    include_tail: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    windows: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]] = []
    for clip_start, clip_end in clip_boundaries:
        if clip_end <= clip_start:
            continue
        local_windows = make_windows(
            latents[clip_start:clip_end],
            indices[clip_start:clip_end],
            window_size=window_size,
            window_stride=window_stride,
            pad_index=pad_index,
            include_tail=include_tail,
        )
        for latent_window, index_window, (local_start, local_end) in local_windows:
            windows.append((latent_window, index_window, (clip_start + local_start, clip_start + local_end)))
    return windows


def build_t5_text_encoder(
    model_name: str,
    device: str,
    max_length: int = 256,
) -> Callable[[list[str]], tuple[np.ndarray, np.ndarray]]:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    encoder = T5EncoderModel.from_pretrained(model_name).to(device)
    encoder.eval()

    def encode(captions: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encoded = tokenizer(
            captions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = encoder(**encoded)
        features = output.last_hidden_state.detach().cpu().numpy().astype(np.float32)
        masks = (~encoded["attention_mask"].bool()).detach().cpu().numpy().astype(bool)
        if features.shape[-1] != 1024:
            raise ValueError(f"expected T5 feature dim 1024, got {features.shape[-1]}")
        return features, masks

    return encode


def _manifest_clip_boundaries_to_latent(
    clip_boundaries: list[list[int]] | list[tuple[int, int]],
    latent_length: int,
    observation_length: int,
) -> list[tuple[int, int]]:
    if observation_length <= 0:
        return []
    scale = float(latent_length) / float(observation_length)
    latent_boundaries: list[tuple[int, int]] = []
    last_end = 0
    for idx, (start, end) in enumerate(clip_boundaries):
        latent_start = int(round(float(start) * scale))
        latent_end = int(round(float(end) * scale))
        if idx == 0:
            latent_start = 0
        latent_start = max(last_end, min(latent_start, latent_length))
        latent_end = max(latent_start, min(latent_end, latent_length))
        latent_boundaries.append((latent_start, latent_end))
        last_end = latent_end
    if latent_boundaries:
        start, _ = latent_boundaries[-1]
        latent_boundaries[-1] = (start, latent_length)
    return latent_boundaries


def select_window_caption(
    full_caption: str,
    clip_captions: list[str],
    clip_boundaries: list[list[int]] | list[tuple[int, int]],
    window_range: tuple[int, int],
    latent_length: int,
    observation_length: int,
    joiner: str = " then ",
) -> str:
    if not clip_captions or not clip_boundaries:
        return full_caption
    latent_boundaries = _manifest_clip_boundaries_to_latent(
        clip_boundaries,
        latent_length=latent_length,
        observation_length=observation_length,
    )
    if not latent_boundaries:
        return full_caption
    win_start, win_end = window_range
    selected = []
    for caption, (clip_start, clip_end) in zip(clip_captions, latent_boundaries):
        if clip_start < win_end and clip_end > win_start:
            selected.append(str(caption))
    return joiner.join(selected) if selected else full_caption


def _latent_clip_boundaries_from_row(
    row: dict[str, object],
    latent_length: int,
    observation_length: int,
) -> list[tuple[int, int]]:
    return _manifest_clip_boundaries_to_latent(
        row.get("clip_boundaries", []),
        latent_length=latent_length,
        observation_length=observation_length,
    )


def _filter_boundaries_around_forced_transitions(
    boundaries: list[tuple[int, int]],
    forced_transitions: list[object],
    margin: int,
) -> tuple[list[tuple[int, int]], bool]:
    if margin <= 0 or not forced_transitions:
        return boundaries, False
    filtered = list(boundaries)
    changed = False
    for transition_idx, forced in enumerate(forced_transitions):
        if not forced or transition_idx + 1 >= len(filtered):
            continue
        changed = True
        left_start, left_end = filtered[transition_idx]
        right_start, right_end = filtered[transition_idx + 1]
        filtered[transition_idx] = (left_start, max(left_start, left_end - margin))
        filtered[transition_idx + 1] = (min(right_end, right_start + margin), right_end)
    return [(start, end) for start, end in filtered if end > start], changed


def build_cache_from_long_h5(
    long_h5_path: Path,
    manifest_path: Path,
    agent,
    text_encoder: Callable[[list[str]], tuple[np.ndarray, np.ndarray]],
    window_size: int,
    window_stride: int,
    rvq_depth: int,
    fps: int = 20,
    caption_mode: str = "window",
    caption_joiner: str = " then ",
    window_policy: str = "sequence",
    forced_transition_margin: int = 0,
    text_model: str | None = None,
    max_text_length: int | None = None,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    if caption_mode not in {"sequence", "window"}:
        raise ValueError(f"unknown caption_mode: {caption_mode}")
    if window_policy not in {"sequence", "clip"}:
        raise ValueError(f"unknown window_policy: {window_policy}")
    max_motion_tokens = DEFAULT_TEXT_GPT_BLOCK_SIZE - 1
    if window_size > max_motion_tokens:
        raise ValueError(
            f"window_size {window_size} exceeds GPT motion context {max_motion_tokens}; "
            f"block_size {DEFAULT_TEXT_GPT_BLOCK_SIZE} reserves one condition token"
        )
    manifest = load_manifest(manifest_path)
    latents_all = []
    indices_all = []
    text_features_all = []
    text_masks_all = []
    captions = []
    sequence_ids = []
    window_ranges = []
    sample_ids_all = []
    failures: list[dict[str, str]] = []
    encoded_text_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    with h5py.File(long_h5_path, "r") as h5:
        for sequence_id in h5.keys():
            try:
                row = manifest.get(sequence_id, {})
                group = h5[sequence_id]
                caption = str(row.get("caption") or group.attrs.get("caption", ""))
                sample_ids = row.get("sample_ids")
                if sample_ids is None:
                    sample_ids = str(group.attrs.get("sample_ids", "")).split(",")
                joints = group["joints_22"][:]
                state = humanml3d_joints_to_moconvq_state(joints, fps=fps)
                observation = moconvq_state_to_observation(state)
                latent, index = encode_observation_with_agent(agent, observation, rvq_depth=rvq_depth)
                latent_clip_boundaries = _latent_clip_boundaries_from_row(
                    row,
                    latent_length=len(latent),
                    observation_length=len(observation),
                )
                latent_clip_boundaries, forced_boundaries_trimmed = _filter_boundaries_around_forced_transitions(
                    latent_clip_boundaries,
                    forced_transitions=list(row.get("transition_forced", [])),
                    margin=forced_transition_margin,
                )
                if window_policy == "clip" and latent_clip_boundaries:
                    windows = make_clip_aligned_windows(
                        latent,
                        index,
                        window_size=window_size,
                        window_stride=window_stride,
                        clip_boundaries=latent_clip_boundaries,
                        include_tail=not forced_boundaries_trimmed,
                    )
                else:
                    windows = make_windows(
                        latent,
                        index,
                        window_size=window_size,
                        window_stride=window_stride,
                    )
                for latent_window, index_window, window_range in windows:
                    window_caption = caption
                    if caption_mode == "window":
                        window_caption = select_window_caption(
                            full_caption=caption,
                            clip_captions=[str(x) for x in row.get("clip_captions", [])],
                            clip_boundaries=row.get("clip_boundaries", []),
                            window_range=window_range,
                            latent_length=len(latent),
                            observation_length=len(observation),
                            joiner=caption_joiner,
                        )
                    if window_caption not in encoded_text_cache:
                        encoded_text_cache[window_caption] = text_encoder([window_caption])
                    text_feature, text_mask = encoded_text_cache[window_caption]
                    latents_all.append(latent_window)
                    indices_all.append(index_window)
                    text_features_all.append(text_feature[0])
                    text_masks_all.append(text_mask[0])
                    captions.append(window_caption)
                    sequence_ids.append(sequence_id)
                    window_ranges.append(window_range)
                    sample_ids_all.append(list(sample_ids))
            except Exception as exc:  # noqa: BLE001 - failure log needs to preserve all conversion errors.
                failures.append(
                    {
                        "sequence_id": str(sequence_id),
                        "reason": str(exc),
                        "traceback_short": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                    }
                )

    cache = {
        "latents": torch.from_numpy(np.stack(latents_all, axis=0)) if latents_all else torch.empty((0, window_size, 768)),
        "indices": torch.from_numpy(np.stack(indices_all, axis=0)) if indices_all else torch.empty((0, window_size, rvq_depth), dtype=torch.long),
        "text_features": torch.from_numpy(np.stack(text_features_all, axis=0)) if text_features_all else torch.empty((0, 0, 1024)),
        "text_masks": torch.from_numpy(np.stack(text_masks_all, axis=0)) if text_masks_all else torch.empty((0, 0), dtype=torch.bool),
        "captions": captions,
        "sequence_ids": sequence_ids,
        "window_ranges": window_ranges,
        "sample_ids": sample_ids_all,
        "config": {
            "long_h5": str(long_h5_path),
            "manifest": str(manifest_path),
            "window_size": window_size,
            "window_stride": window_stride,
            "rvq_depth": rvq_depth,
            "fps": fps,
            "caption_mode": caption_mode,
            "caption_joiner": caption_joiner,
            "window_policy": window_policy,
            "forced_transition_margin": forced_transition_margin,
            "text_model": text_model,
            "max_text_length": max_text_length,
        },
    }
    return cache, failures


def build_loaded_moconvq_agent(gpu: int, base_data: Path):
    import argparse as _argparse

    import MoConVQCore.Utils.pytorch_utils as ptu
    from MoConVQCore.Env.vclode_track_env import VCLODETrackEnv
    from MoConVQCore.Model.MoConVQ import MoConVQ
    from MoConVQCore.Utils.misc import load_yaml
    from Script.tokenize_motion import flatten_dict

    parser = _argparse.ArgumentParser()
    parser.add_argument("--config_file", default="Data/Parameters/bigdata.yml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="stage1_real_cache")
    parser.add_argument("--load", default=False, action="store_true")
    parser.add_argument("--gpu", type=int, default=gpu)
    parser.add_argument("--cpu_b", type=int, default=0)
    parser.add_argument("--cpu_e", type=int, default=-1)
    parser.add_argument("--train_prior", default=False, action="store_true")
    parser = VCLODETrackEnv.add_specific_args(parser)
    parser = MoConVQ.add_specific_args(parser)
    args = vars(parser.parse_args(args=[]))
    args.update(flatten_dict(load_yaml(args["config_file"])))
    args["gpu"] = gpu
    ptu.init_gpu(True, gpu_id=gpu)
    env = VCLODETrackEnv(**args)
    agent = MoConVQ(323, 12, 57, 120, env, training=False, **args)
    agent.simple_load(str(base_data), strict=True)
    agent.eval()
    agent.posterior.limit = False
    return agent


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-h5", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--text-model", default="t5-large")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--window-stride", type=int, default=25)
    parser.add_argument("--rvq-depth", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--max-failure-rate", type=float, default=0.1)
    parser.add_argument("--caption-mode", choices=("sequence", "window"), default="window")
    parser.add_argument("--caption-joiner", default=" then ")
    parser.add_argument("--window-policy", choices=("sequence", "clip"), default="sequence")
    parser.add_argument("--forced-transition-margin", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failure-log", required=True)
    args = parser.parse_args(argv)

    import MoConVQCore.Utils.pytorch_utils as ptu

    agent = build_loaded_moconvq_agent(gpu=args.gpu, base_data=Path(args.base_data))
    text_encoder = build_t5_text_encoder(args.text_model, device=str(ptu.device), max_length=args.max_text_length)
    cache, failures = build_cache_from_long_h5(
        long_h5_path=Path(args.long_h5),
        manifest_path=Path(args.manifest),
        agent=agent,
        text_encoder=text_encoder,
        window_size=args.window_size,
        window_stride=args.window_stride,
        rvq_depth=args.rvq_depth,
        fps=args.fps,
        caption_mode=args.caption_mode,
        caption_joiner=args.caption_joiner,
        window_policy=args.window_policy,
        forced_transition_margin=args.forced_transition_margin,
        text_model=args.text_model,
        max_text_length=args.max_text_length,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    failure_log = Path(args.failure_log)
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    with failure_log.open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False))
            f.write("\n")

    total_sequences = len(set(cache["sequence_ids"])) + len(failures)
    failure_rate = len(failures) / max(total_sequences, 1)
    if cache["indices"].numel() > 0:
        idx_min = int(cache["indices"][cache["indices"] != 513].min()) if torch.any(cache["indices"] != 513) else 513
        idx_max = int(cache["indices"][cache["indices"] != 513].max()) if torch.any(cache["indices"] != 513) else 513
    else:
        idx_min = idx_max = -1
    print(
        json.dumps(
            {
                "windows": int(cache["latents"].shape[0]),
                "failed_sequences": len(failures),
                "failure_rate": failure_rate,
                "index_min": idx_min,
                "index_max": idx_max,
                "caption_mode": cache["config"]["caption_mode"],
                "window_policy": cache["config"]["window_policy"],
                "forced_transition_margin": cache["config"]["forced_transition_margin"],
            },
            indent=2,
        )
    )
    if failure_rate > args.max_failure_rate:
        raise SystemExit(f"failure rate {failure_rate:.3f} exceeded max {args.max_failure_rate:.3f}")


if __name__ == "__main__":
    main()
