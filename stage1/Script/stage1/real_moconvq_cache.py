from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
import argparse
from functools import lru_cache
import json
import sys
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
DEFAULT_MOCONVQ_WORLD_JSON = Path(__file__).resolve().parents[2] / "Data/Misc/world.json"
ROTATION_CALIBRATION_CHOICES = ("none", "rest")
ROTATION_SOURCE_CHOICES = ("heuristic", "humanml_vec6d")
CACHE_SAMPLE_MODE_CHOICES = ("window", "segment_prefix")

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


def _cont6d_to_matrix(cont6d: np.ndarray) -> np.ndarray:
    if cont6d.shape[-1] != 6:
        raise ValueError(f"expected cont6d last dim 6, got {cont6d.shape}")
    x_raw = cont6d[..., 0:3].astype(np.float64)
    y_raw = cont6d[..., 3:6].astype(np.float64)
    x_norm = np.linalg.norm(x_raw, axis=-1, keepdims=True)
    x = x_raw / np.maximum(x_norm, 1e-8)
    z = np.cross(x, y_raw)
    z_norm = np.linalg.norm(z, axis=-1, keepdims=True)
    z = z / np.maximum(z_norm, 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _humanml3d_root_yaw_matrices(joint_vecs_263: np.ndarray) -> np.ndarray:
    rot_vel = joint_vecs_263[:, 0].astype(np.float32)
    root_yaw_half = np.zeros_like(rot_vel, dtype=np.float32)
    if len(rot_vel) > 1:
        root_yaw_half[1:] = rot_vel[:-1]
    root_yaw_half = np.cumsum(root_yaw_half, axis=0)
    xyzw = np.zeros((len(joint_vecs_263), 4), dtype=np.float32)
    xyzw[:, 1] = np.sin(root_yaw_half)
    xyzw[:, 3] = np.cos(root_yaw_half)
    return Rotation.from_quat(xyzw).as_matrix().astype(np.float32)


def humanml3d_joint_vecs_to_global_quats_xyzw(joint_vecs_263: np.ndarray) -> np.ndarray:
    """Recover HumanML3D global joint rotations from the 263-d motion vector.

    HumanML3D's representation stores root yaw velocity plus 21 local joint
    rotations in continuous 6D form.  This helper reconstructs global 22-joint
    rotations using the official HumanML3D/T2M kinematic tree layout, but keeps
    the output in SciPy/MoConVQ quaternion order `(x, y, z, w)`.
    """

    if joint_vecs_263.ndim != 2 or joint_vecs_263.shape[1] != 263:
        raise ValueError(f"expected joint_vecs_263 shape (T, 263), got {joint_vecs_263.shape}")
    frames = joint_vecs_263.shape[0]
    local_cont6d = np.zeros((frames, 22, 6), dtype=np.float32)
    local_cont6d[:, 0, 0] = 1.0
    local_cont6d[:, 0, 4] = 1.0
    rot_start = 4 + (22 - 1) * 3
    rot_end = rot_start + (22 - 1) * 6
    local_cont6d[:, 1:, :] = joint_vecs_263[:, rot_start:rot_end].reshape(frames, 21, 6)
    local_mats = _cont6d_to_matrix(local_cont6d)
    root_mats = _humanml3d_root_yaw_matrices(joint_vecs_263)

    # HumanML3D/T2M kinematic tree from HumanML3D/paramUtil.py.
    chains = (
        (0, 2, 5, 8, 11),
        (0, 1, 4, 7, 10),
        (0, 3, 6, 9, 12, 15),
        (9, 14, 17, 19, 21),
        (9, 13, 16, 18, 20),
    )
    global_mats = np.zeros((frames, 22, 3, 3), dtype=np.float32)
    global_mats[:, 0] = root_mats
    for chain in chains:
        current = root_mats.copy()
        for joint_id in chain[1:]:
            current = np.matmul(current, local_mats[:, joint_id])
            global_mats[:, joint_id] = current

    quats = Rotation.from_matrix(global_mats.reshape(-1, 3, 3)).as_quat().reshape(frames, 22, 4)
    return _ensure_quat_continuity(quats.astype(np.float32))


@lru_cache(maxsize=8)
def _load_moconvq_world_rest_pose(world_json_path: str) -> tuple[np.ndarray, np.ndarray]:
    with Path(world_json_path).open("r", encoding="utf-8") as f:
        world = json.load(f)
    bodies = world["CharacterList"]["Characters"][0]["Bodies"][: len(MOCONVQ_BODY_NAMES)]
    names = [str(body["Name"]) for body in bodies]
    if names != list(MOCONVQ_BODY_NAMES):
        raise ValueError(
            "MoConVQ world body order does not match Stage1 mapping: "
            f"expected {list(MOCONVQ_BODY_NAMES)}, got {names}"
        )
    positions = np.asarray([body["Position"] for body in bodies], dtype=np.float32)
    quats = np.asarray([body["Quaternion"] for body in bodies], dtype=np.float32)
    return positions, quats


def moconvq_rest_rotation_reference(
    world_json_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return heuristic rest quats and target MoConVQ world rest quats.

    HumanML3D joint positions do not carry rigid-body local frame rotations.
    The heuristic retarget estimates each body quaternion from bone directions,
    but MoConVQ's simulator character uses the rigid-body frames defined in
    world.json.  The rest-pose reference lets us remove the static frame offset
    introduced by the heuristic axes.
    """

    world_path = Path(world_json_path) if world_json_path is not None else DEFAULT_MOCONVQ_WORLD_JSON
    rest_positions, target_rest_quats = _load_moconvq_world_rest_pose(str(world_path.resolve()))
    identity_axes = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        UP.copy(),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    heuristic_rest_quats = np.stack(
        [_bone_quat(body_id, rest_positions, identity_axes) for body_id in range(len(MOCONVQ_BODY_NAMES))],
        axis=0,
    ).astype(np.float32)
    return heuristic_rest_quats, target_rest_quats.copy()


def apply_moconvq_rotation_calibration(
    quats: np.ndarray,
    mode: str = "rest",
    world_json_path: Path | str | None = None,
) -> np.ndarray:
    if mode not in ROTATION_CALIBRATION_CHOICES:
        raise ValueError(f"unknown rotation calibration mode: {mode}")
    if mode == "none":
        return quats.astype(np.float32, copy=True)
    if quats.ndim != 3 or quats.shape[1:] != (len(MOCONVQ_BODY_NAMES), 4):
        raise ValueError(f"expected quaternion shape (T, 20, 4), got {quats.shape}")

    heuristic_rest, target_rest = moconvq_rest_rotation_reference(world_json_path)
    frames = quats.shape[0]
    current = Rotation.from_quat(quats.reshape(-1, 4))
    rest_inv = Rotation.from_quat(np.tile(heuristic_rest[None, :, :], (frames, 1, 1)).reshape(-1, 4)).inv()
    target = Rotation.from_quat(np.tile(target_rest[None, :, :], (frames, 1, 1)).reshape(-1, 4))
    calibrated = (current * rest_inv * target).as_quat().reshape(quats.shape).astype(np.float32)
    return calibrated


def apply_identity_source_rotation_calibration(
    quats: np.ndarray,
    mode: str = "rest",
    world_json_path: Path | str | None = None,
) -> np.ndarray:
    if mode not in ROTATION_CALIBRATION_CHOICES:
        raise ValueError(f"unknown rotation calibration mode: {mode}")
    if mode == "none":
        return quats.astype(np.float32, copy=True)
    if quats.ndim != 3 or quats.shape[1:] != (len(MOCONVQ_BODY_NAMES), 4):
        raise ValueError(f"expected quaternion shape (T, 20, 4), got {quats.shape}")
    world_path = Path(world_json_path) if world_json_path is not None else DEFAULT_MOCONVQ_WORLD_JSON
    _, target_rest = _load_moconvq_world_rest_pose(str(world_path.resolve()))
    frames = quats.shape[0]
    current = Rotation.from_quat(quats.reshape(-1, 4))
    target = Rotation.from_quat(np.tile(target_rest[None, :, :], (frames, 1, 1)).reshape(-1, 4))
    return (current * target).as_quat().reshape(quats.shape).astype(np.float32)


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


def humanml3d_joints_to_moconvq_state(
    joints_22: np.ndarray,
    joint_vecs_263: np.ndarray | None = None,
    fps: int = 20,
    rotation_source: str = "heuristic",
    rotation_calibration: str = "rest",
    world_json_path: Path | str | None = None,
) -> np.ndarray:
    if joints_22.ndim != 3 or joints_22.shape[1:] != (22, 3):
        raise ValueError(f"expected joints shape (T, 22, 3), got {joints_22.shape}")
    if rotation_source not in ROTATION_SOURCE_CHOICES:
        raise ValueError(f"unknown rotation_source: {rotation_source}")
    positions = joints_22[:, HUMANML3D_TO_MOCONVQ, :].astype(np.float32)
    if rotation_source == "humanml_vec6d":
        if joint_vecs_263 is None:
            raise ValueError("rotation_source='humanml_vec6d' requires joint_vecs_263")
        if len(joint_vecs_263) != len(joints_22):
            raise ValueError(
                "joints_22 and joint_vecs_263 length mismatch: "
                f"{len(joints_22)} != {len(joint_vecs_263)}"
            )
        humanml_quats = humanml3d_joint_vecs_to_global_quats_xyzw(joint_vecs_263)
        quats = humanml_quats[:, HUMANML3D_TO_MOCONVQ, :].astype(np.float32)
        quats = apply_identity_source_rotation_calibration(
            quats,
            mode=rotation_calibration,
            world_json_path=world_json_path,
        )
    else:
        quats = np.zeros((positions.shape[0], 20, 4), dtype=np.float32)
        for t in range(positions.shape[0]):
            axes = _frame_axes(joints_22[t].astype(np.float32))
            for body_id in range(20):
                quats[t, body_id] = _bone_quat(body_id, positions[t], axes)
        quats = _ensure_quat_continuity(quats)
        quats = apply_moconvq_rotation_calibration(
            quats,
            mode=rotation_calibration,
            world_json_path=world_json_path,
        )
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


def summarize_observation_abs_z(agent, observation: np.ndarray) -> dict[str, float]:
    """Summarize how far converted HumanML3D observations are from MoConVQ data.

    MoConVQ stores the observation normalization statistics used by the encoder.
    Large normalized outliers are a useful proxy for retarget failures before we
    spend time encoding and training on those sequences.
    """

    obs_mean = agent.obs_mean.detach().cpu().numpy().astype(np.float32)
    obs_std = agent.obs_std.detach().cpu().numpy().astype(np.float32)
    if obs_mean.shape != (observation.shape[-1],) or obs_std.shape != (observation.shape[-1],):
        raise ValueError(
            "agent observation statistics do not match converted observation: "
            f"mean={obs_mean.shape}, std={obs_std.shape}, observation={observation.shape}"
        )
    abs_z = np.abs((observation.astype(np.float32) - obs_mean) / (obs_std + 1e-8))
    return {
        "mean_abs_z": float(np.mean(abs_z)),
        "p95_abs_z": float(np.percentile(abs_z, 95)),
        "p99_abs_z": float(np.percentile(abs_z, 99)),
        "max_abs_z": float(np.max(abs_z)),
        "frac_gt_3": float(np.mean(abs_z > 3.0)),
        "frac_gt_5": float(np.mean(abs_z > 5.0)),
        "frac_gt_10": float(np.mean(abs_z > 10.0)),
    }


def observation_quality_rejection_reason(
    quality: dict[str, float],
    max_p99_abs_z: float | None = None,
    max_frac_gt_5: float | None = None,
    max_frac_gt_10: float | None = None,
) -> str | None:
    reasons = []
    if max_p99_abs_z is not None and quality["p99_abs_z"] > max_p99_abs_z:
        reasons.append(f"p99_abs_z={quality['p99_abs_z']:.4f}>{max_p99_abs_z:.4f}")
    if max_frac_gt_5 is not None and quality["frac_gt_5"] > max_frac_gt_5:
        reasons.append(f"frac_gt_5={quality['frac_gt_5']:.6f}>{max_frac_gt_5:.6f}")
    if max_frac_gt_10 is not None and quality["frac_gt_10"] > max_frac_gt_10:
        reasons.append(f"frac_gt_10={quality['frac_gt_10']:.6f}>{max_frac_gt_10:.6f}")
    return "; ".join(reasons) if reasons else None


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


def make_segment_prefix_windows(
    latents: np.ndarray,
    indices: np.ndarray,
    window_size: int,
    window_stride: int,
    clip_boundaries: list[tuple[int, int]],
    prefix_size: int,
    pad_index: int = 513,
    include_tail: bool = True,
) -> list[dict[str, object]]:
    """Build current-segment targets with previous motion as context.

    Each returned sample contains a motion prefix followed by target tokens from
    one semantic segment.  The prefix stays in the autoregressive context, but a
    separate target mask makes the loss ignore prefix tokens.
    """

    if len(latents) != len(indices):
        raise ValueError("latent/index length mismatch")
    if window_size < 1 or window_stride < 1:
        raise ValueError("window size and stride must be positive")
    if prefix_size < 0:
        raise ValueError("prefix_size must be non-negative")
    if not clip_boundaries:
        clip_boundaries = [(0, len(latents))]

    samples: list[dict[str, object]] = []
    prefix_size = min(int(prefix_size), window_size - 1) if window_size > 1 else 0
    target_capacity_without_prefix = max(window_size - prefix_size, 1)
    for segment_idx, (clip_start, clip_end) in enumerate(clip_boundaries):
        if clip_end <= clip_start:
            continue
        starts = [clip_start]
        segment_length = clip_end - clip_start
        if segment_length > target_capacity_without_prefix:
            starts = list(range(clip_start, clip_end, window_stride))
            if include_tail:
                tail_start = max(clip_start, clip_end - target_capacity_without_prefix)
                if starts[-1] != tail_start:
                    starts.append(tail_start)
            starts = sorted(set(starts))
        for target_start in starts:
            target_start = max(clip_start, min(target_start, clip_end))
            prefix_start = max(0, target_start - prefix_size)
            prefix_len = target_start - prefix_start
            target_capacity = window_size - prefix_len
            if target_capacity <= 0:
                continue
            target_end = min(clip_end, target_start + target_capacity)
            target_valid = target_end - target_start
            if target_valid <= 0:
                continue
            source_start = prefix_start
            source_end = target_end
            valid = source_end - source_start
            latent_window = np.zeros((window_size, latents.shape[-1]), dtype=np.float32)
            index_window = np.full((window_size, indices.shape[-1]), pad_index, dtype=np.int64)
            target_mask = np.zeros((window_size,), dtype=bool)
            end_mask = np.zeros((window_size,), dtype=bool)
            latent_window[:valid] = latents[source_start:source_end]
            index_window[:valid] = indices[source_start:source_end]
            target_mask[prefix_len : prefix_len + target_valid] = True
            if target_end >= clip_end and prefix_len + target_valid < window_size:
                end_mask[prefix_len + target_valid] = True
            samples.append(
                {
                    "latent": latent_window,
                    "indices": index_window,
                    "target_mask": target_mask,
                    "end_mask": end_mask,
                    "window_range": (source_start, source_end),
                    "target_range": (target_start, target_end),
                    "prefix_range": (prefix_start, target_start),
                    "segment_idx": int(segment_idx),
                    "num_segments": int(len(clip_boundaries)),
                    "segment_range": (clip_start, clip_end),
                    "prefix_length": int(prefix_len),
                }
            )
    return samples


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
    window_policy: str = "clip",
    sample_mode: str = "window",
    prefix_size: int = 25,
    forced_transition_margin: int = 0,
    text_model: str | None = None,
    max_text_length: int | None = None,
    rotation_source: str = "heuristic",
    rotation_calibration: str = "rest",
    world_json_path: Path | str | None = None,
    max_observation_p99_abs_z: float | None = None,
    max_observation_frac_gt_5: float | None = None,
    max_observation_frac_gt_10: float | None = None,
    progress_every: int = 0,
) -> tuple[dict[str, object], list[dict[str, str]]]:
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
    manifest = load_manifest(manifest_path)
    latents_all = []
    indices_all = []
    text_features_all = []
    text_masks_all = []
    captions = []
    sequence_ids = []
    window_ranges = []
    target_ranges = []
    prefix_ranges = []
    segment_ranges = []
    target_masks = []
    end_masks = []
    segment_idxs = []
    num_segments_all = []
    segment_progresses = []
    prefix_lengths = []
    sample_ids_all = []
    failures: list[dict[str, str]] = []
    filtered_sequences: list[dict[str, object]] = []
    observation_quality_rows: list[dict[str, object]] = []
    encoded_text_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    should_filter_observation = any(
        item is not None
        for item in (
            max_observation_p99_abs_z,
            max_observation_frac_gt_5,
            max_observation_frac_gt_10,
        )
    )
    can_record_observation_quality = hasattr(agent, "obs_mean") and hasattr(agent, "obs_std")

    with h5py.File(long_h5_path, "r") as h5:
        sequence_keys = list(h5.keys())
        total_sequence_count = len(sequence_keys)
        for sequence_number, sequence_id in enumerate(sequence_keys, start=1):
            try:
                row = manifest.get(sequence_id, {})
                group = h5[sequence_id]
                caption = str(row.get("caption") or group.attrs.get("caption", ""))
                sample_ids = row.get("sample_ids")
                if sample_ids is None:
                    sample_ids = str(group.attrs.get("sample_ids", "")).split(",")
                joints = group["joints_22"][:]
                joint_vecs = group["joint_vecs_263"][:] if "joint_vecs_263" in group else None
                state = humanml3d_joints_to_moconvq_state(
                    joints,
                    joint_vecs_263=joint_vecs,
                    fps=fps,
                    rotation_source=rotation_source,
                    rotation_calibration=rotation_calibration,
                    world_json_path=world_json_path,
                )
                observation = moconvq_state_to_observation(state)
                if should_filter_observation and not can_record_observation_quality:
                    failures.append(
                        {
                            "sequence_id": str(sequence_id),
                            "reason": "observation filtering requires agent.obs_mean and agent.obs_std",
                            "traceback_short": "AttributeError: agent.obs_mean/obs_std missing",
                        }
                    )
                    continue
                if can_record_observation_quality:
                    observation_quality = summarize_observation_abs_z(agent, observation)
                    observation_quality_rows.append({"sequence_id": str(sequence_id), **observation_quality})
                    if should_filter_observation:
                        rejection_reason = observation_quality_rejection_reason(
                            observation_quality,
                            max_p99_abs_z=max_observation_p99_abs_z,
                            max_frac_gt_5=max_observation_frac_gt_5,
                            max_frac_gt_10=max_observation_frac_gt_10,
                        )
                        if rejection_reason is not None:
                            filtered_sequences.append(
                                {
                                    "sequence_id": str(sequence_id),
                                    "reason": rejection_reason,
                                    **observation_quality,
                                }
                            )
                            continue
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
                if sample_mode == "segment_prefix" and latent_clip_boundaries:
                    sample_windows = make_segment_prefix_windows(
                        latent,
                        index,
                        window_size=window_size,
                        window_stride=window_stride,
                        clip_boundaries=latent_clip_boundaries,
                        prefix_size=prefix_size,
                        include_tail=not forced_boundaries_trimmed,
                    )
                else:
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
                    sample_windows = [
                        {
                            "latent": latent_window,
                            "indices": index_window,
                            "target_mask": index_window[:, 0] != 513,
                            "end_mask": np.concatenate(
                                [
                                    np.asarray([False] * int(np.sum(index_window[:, 0] != 513)), dtype=bool),
                                    np.asarray([True], dtype=bool)
                                    if int(np.sum(index_window[:, 0] != 513)) < window_size
                                    else np.asarray([], dtype=bool),
                                    np.asarray(
                                        [False]
                                        * max(window_size - int(np.sum(index_window[:, 0] != 513)) - 1, 0),
                                        dtype=bool,
                                    ),
                                ]
                            ),
                            "window_range": window_range,
                            "target_range": window_range,
                            "prefix_range": (window_range[0], window_range[0]),
                            "segment_idx": 0,
                            "num_segments": max(len(latent_clip_boundaries), 1),
                            "segment_range": window_range,
                            "prefix_length": 0,
                        }
                        for latent_window, index_window, window_range in windows
                    ]

                for sample_window in sample_windows:
                    latent_window = sample_window["latent"]
                    index_window = sample_window["indices"]
                    window_range = sample_window["target_range"] if sample_mode == "segment_prefix" else sample_window["window_range"]
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
                    target_masks.append(np.asarray(sample_window["target_mask"], dtype=bool))
                    end_masks.append(np.asarray(sample_window["end_mask"], dtype=bool))
                    text_features_all.append(text_feature[0])
                    text_masks_all.append(text_mask[0])
                    captions.append(window_caption)
                    sequence_ids.append(sequence_id)
                    window_ranges.append(tuple(sample_window["window_range"]))
                    target_ranges.append(tuple(sample_window["target_range"]))
                    prefix_ranges.append(tuple(sample_window["prefix_range"]))
                    segment_ranges.append(tuple(sample_window["segment_range"]))
                    segment_idx = int(sample_window["segment_idx"])
                    num_segments = int(sample_window["num_segments"])
                    segment_idxs.append(segment_idx)
                    num_segments_all.append(num_segments)
                    segment_progresses.append(float(segment_idx / max(num_segments - 1, 1)) if num_segments > 1 else 0.0)
                    prefix_lengths.append(int(sample_window["prefix_length"]))
                    sample_ids_all.append(list(sample_ids))
            except Exception as exc:  # noqa: BLE001 - failure log needs to preserve all conversion errors.
                failures.append(
                    {
                        "sequence_id": str(sequence_id),
                        "reason": str(exc),
                        "traceback_short": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                    }
                )
            finally:
                if progress_every and sequence_number % progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "processed_sequences": sequence_number,
                                "total_sequences": total_sequence_count,
                                "windows": len(latents_all),
                                "filtered_sequences": len(filtered_sequences),
                                "failed_sequences": len(failures),
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

    cache = {
        "latents": torch.from_numpy(np.stack(latents_all, axis=0)) if latents_all else torch.empty((0, window_size, 768)),
        "indices": torch.from_numpy(np.stack(indices_all, axis=0)) if indices_all else torch.empty((0, window_size, rvq_depth), dtype=torch.long),
        "text_features": torch.from_numpy(np.stack(text_features_all, axis=0)) if text_features_all else torch.empty((0, 0, 1024)),
        "text_masks": torch.from_numpy(np.stack(text_masks_all, axis=0)) if text_masks_all else torch.empty((0, 0), dtype=torch.bool),
        "target_masks": torch.from_numpy(np.stack(target_masks, axis=0)) if target_masks else torch.empty((0, window_size), dtype=torch.bool),
        "end_masks": torch.from_numpy(np.stack(end_masks, axis=0)) if end_masks else torch.empty((0, window_size), dtype=torch.bool),
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
        "filtered_sequences": filtered_sequences,
        "observation_quality": observation_quality_rows,
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
            "sample_mode": sample_mode,
            "prefix_size": prefix_size,
            "forced_transition_margin": forced_transition_margin,
            "text_model": text_model,
            "max_text_length": max_text_length,
            "rotation_source": rotation_source,
            "rotation_calibration": rotation_calibration,
            "world_json": str(world_json_path) if world_json_path is not None else str(DEFAULT_MOCONVQ_WORLD_JSON),
            "max_observation_p99_abs_z": max_observation_p99_abs_z,
            "max_observation_frac_gt_5": max_observation_frac_gt_5,
            "max_observation_frac_gt_10": max_observation_frac_gt_10,
        },
    }
    return cache, failures


def build_loaded_moconvq_agent(gpu: int, base_data: Path, motion_dataset: Path | None = None):
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
    if motion_dataset is not None:
        args["motion_dataset"] = str(motion_dataset)
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
    parser.add_argument("--window-policy", choices=("sequence", "clip"), default="clip")
    parser.add_argument("--sample-mode", choices=CACHE_SAMPLE_MODE_CHOICES, default="window")
    parser.add_argument("--prefix-size", type=int, default=25)
    parser.add_argument("--forced-transition-margin", type=int, default=0)
    parser.add_argument("--rotation-calibration", choices=ROTATION_CALIBRATION_CHOICES, default="rest")
    parser.add_argument("--rotation-source", choices=ROTATION_SOURCE_CHOICES, default="heuristic")
    parser.add_argument("--world-json", default=str(DEFAULT_MOCONVQ_WORLD_JSON))
    parser.add_argument("--max-observation-p99-abs-z", type=float, default=None)
    parser.add_argument("--max-observation-frac-gt-5", type=float, default=None)
    parser.add_argument("--max-observation-frac-gt-10", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=0)
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
        sample_mode=args.sample_mode,
        prefix_size=args.prefix_size,
        forced_transition_margin=args.forced_transition_margin,
        text_model=args.text_model,
        max_text_length=args.max_text_length,
        rotation_source=args.rotation_source,
        rotation_calibration=args.rotation_calibration,
        world_json_path=Path(args.world_json),
        max_observation_p99_abs_z=args.max_observation_p99_abs_z,
        max_observation_frac_gt_5=args.max_observation_frac_gt_5,
        max_observation_frac_gt_10=args.max_observation_frac_gt_10,
        progress_every=args.progress_every,
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
    filtered_sequences = cache.get("filtered_sequences", [])
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
                "sample_mode": cache["config"]["sample_mode"],
                "prefix_size": cache["config"]["prefix_size"],
                "forced_transition_margin": cache["config"]["forced_transition_margin"],
                "rotation_source": cache["config"]["rotation_source"],
                "filtered_sequences": len(filtered_sequences),
                "max_observation_p99_abs_z": cache["config"]["max_observation_p99_abs_z"],
                "max_observation_frac_gt_5": cache["config"]["max_observation_frac_gt_5"],
                "max_observation_frac_gt_10": cache["config"]["max_observation_frac_gt_10"],
            },
            indent=2,
        )
    )
    if failure_rate > args.max_failure_rate:
        raise SystemExit(f"failure rate {failure_rate:.3f} exceeded max {args.max_failure_rate:.3f}")


if __name__ == "__main__":
    main()
