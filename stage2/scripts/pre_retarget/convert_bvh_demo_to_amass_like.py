#!/usr/bin/env python3
"""Convert the BVH demo motions into AMASS-like NPZ files for H1 retargeting.

This is a direct pre-retarget bridge:

    BVH demo -> AMASS-like {poses, trans, betas, gender, mocap_framerate}

It intentionally does not run SMPLify. The result is suitable as input to
third_party/human2humanoid/scripts/data_process/grad_fit_h1.py, where the H1
retarget optimization happens.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


SMPL24_TO_BVH = {
    0: "RootJoint",
    1: "lHip",
    2: "rHip",
    3: "pelvis_lowerback",
    4: "lKnee",
    5: "rKnee",
    6: "lowerback_torso",
    7: "lAnkle",
    8: "rAnkle",
    9: "torso_head",
    10: "lToeJoint",
    11: "rToeJoint",
    12: None,
    13: "lTorso_Clavicle",
    14: "rTorso_Clavicle",
    15: None,
    16: "lShoulder",
    17: "rShoulder",
    18: "lElbow",
    19: "rElbow",
    20: "lWrist",
    21: "rWrist",
    22: None,
    23: None,
}


@dataclass
class BvhJoint:
    name: str
    parent: int
    is_end_site: bool = False
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    channels: list[str] = field(default_factory=list)


@dataclass
class BvhMotion:
    joints: list[BvhJoint]
    frames: int
    declared_frames: int
    frame_time: float
    data: np.ndarray


def _read_bvh(path: Path, *, allow_truncated: bool) -> BvhMotion:
    lines = path.read_text(encoding="utf-8").splitlines()

    joints: list[BvhJoint] = []
    stack: list[int] = []
    pending_joint: int | None = None
    motion_line_idx: int | None = None

    for line_idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        if line == "MOTION":
            motion_line_idx = line_idx
            break
        if line.startswith(("ROOT ", "JOINT ")):
            _, name = line.split(maxsplit=1)
            parent = stack[-1] if stack else -1
            joints.append(BvhJoint(name=name, parent=parent))
            pending_joint = len(joints) - 1
        elif line == "End Site":
            if not stack:
                raise ValueError(f"{path}: End Site without parent")
            parent_name = joints[stack[-1]].name
            joints.append(
                BvhJoint(
                    name=f"{parent_name}_end",
                    parent=stack[-1],
                    is_end_site=True,
                )
            )
            pending_joint = len(joints) - 1
        elif line == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
        elif line == "}":
            if stack:
                stack.pop()
        elif line.startswith("OFFSET "):
            if not stack:
                raise ValueError(f"{path}: OFFSET outside joint block")
            joints[stack[-1]].offset = np.array(
                [float(v) for v in line.split()[1:4]], dtype=np.float64
            )
        elif line.startswith("CHANNELS "):
            if not stack:
                raise ValueError(f"{path}: CHANNELS outside joint block")
            parts = line.split()
            channel_count = int(parts[1])
            channels = parts[2 : 2 + channel_count]
            if len(channels) != channel_count:
                raise ValueError(f"{path}: malformed CHANNELS line: {line}")
            joints[stack[-1]].channels = channels

    if motion_line_idx is None:
        raise ValueError(f"{path}: missing MOTION section")

    frames_line = lines[motion_line_idx + 1].strip()
    frame_time_line = lines[motion_line_idx + 2].strip()
    if not frames_line.startswith("Frames:"):
        raise ValueError(f"{path}: expected Frames line, got {frames_line!r}")
    if not frame_time_line.startswith("Frame Time:"):
        raise ValueError(f"{path}: expected Frame Time line, got {frame_time_line!r}")

    declared_frames = int(frames_line.split(":", maxsplit=1)[1].strip())
    frame_time = float(frame_time_line.split(":", maxsplit=1)[1].strip())
    expected_width = sum(len(joint.channels) for joint in joints)

    data_rows = []
    stopped_at: tuple[int, str] | None = None
    for source_line_idx, raw_line in enumerate(lines[motion_line_idx + 3 :], motion_line_idx + 4):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != expected_width:
            stopped_at = (
                source_line_idx,
                f"motion width is {len(tokens)}, expected {expected_width}",
            )
            if allow_truncated:
                break
            raise ValueError(f"{path}:{source_line_idx}: {stopped_at[1]}")
        try:
            data_rows.append([float(v) for v in tokens])
        except ValueError as exc:
            stopped_at = (source_line_idx, f"invalid numeric token: {exc}")
            if allow_truncated:
                break
            raise ValueError(f"{path}:{source_line_idx}: {stopped_at[1]}") from exc

    if not data_rows:
        raise ValueError(f"{path}: no valid motion rows found")

    data = np.array(data_rows, dtype=np.float64)
    if data.shape[0] != declared_frames:
        if not allow_truncated:
            raise ValueError(
                f"{path}: Frames says {declared_frames}, but found {data.shape[0]} "
                "valid rows"
            )
        reason = "reached end of file"
        if stopped_at is not None:
            reason = f"stopped at source line {stopped_at[0]}: {stopped_at[1]}"
        print(
            f"warning: {path}: using {data.shape[0]} valid rows instead of "
            f"declared {declared_frames}; {reason}",
            file=sys.stderr,
        )

    return BvhMotion(
        joints=joints,
        frames=data.shape[0],
        declared_frames=declared_frames,
        frame_time=frame_time,
        data=data,
    )


def _local_transforms(motion: BvhMotion) -> tuple[np.ndarray, list[Rotation]]:
    joint_count = len(motion.joints)
    local_pos = np.zeros((motion.frames, joint_count, 3), dtype=np.float64)
    local_rots: list[Rotation] = [Rotation.identity(motion.frames) for _ in motion.joints]

    cursor = 0
    for joint_idx, joint in enumerate(motion.joints):
        local_pos[:, joint_idx, :] = joint.offset
        if not joint.channels:
            continue

        values = motion.data[:, cursor : cursor + len(joint.channels)]
        cursor += len(joint.channels)

        rotation_order: list[str] = []
        rotation_values: list[np.ndarray] = []
        for channel_idx, channel in enumerate(joint.channels):
            axis = channel[0].upper()
            channel_values = values[:, channel_idx]
            if channel.endswith("position"):
                local_pos[:, joint_idx, "XYZ".index(axis)] = channel_values
            elif channel.endswith("rotation"):
                rotation_order.append(axis)
                rotation_values.append(channel_values)
            else:
                raise ValueError(f"Unsupported BVH channel {channel!r}")

        if rotation_order:
            euler_values = np.stack(rotation_values, axis=-1)
            local_rots[joint_idx] = Rotation.from_euler(
                "".join(rotation_order),
                euler_values,
                degrees=True,
            )

    return local_pos, local_rots


def _fps_from_frame_time(frame_time: float) -> float:
    fps = 1.0 / frame_time
    rounded = round(fps)
    if abs(fps - rounded) < 1e-2:
        return float(rounded)
    return float(fps)


def _convert_motion(
    motion: BvhMotion,
    *,
    zero_horizontal_start: bool,
) -> dict[str, np.ndarray | str | float]:
    joint_name_to_idx = {joint.name: idx for idx, joint in enumerate(motion.joints)}
    local_pos, local_rots = _local_transforms(motion)

    if "RootJoint" not in joint_name_to_idx:
        raise ValueError("BVH motion has no RootJoint")

    # BVH demo is Y-up. A +90 deg X rotation maps BVH up (Y) to AMASS-like up (Z).
    coord_rot = Rotation.from_euler("X", 90.0, degrees=True)
    coord_rot_inv = coord_rot.inv()
    smpl_to_h1_root_alignment = Rotation.from_quat([0.5, 0.5, 0.5, 0.5])

    root_idx = joint_name_to_idx["RootJoint"]
    trans = coord_rot.apply(local_pos[:, root_idx, :])
    if zero_horizontal_start:
        trans[:, :2] -= trans[0:1, :2]

    poses = np.zeros((motion.frames, 24, 3), dtype=np.float64)
    for smpl_idx, bvh_name in SMPL24_TO_BVH.items():
        if bvh_name is None:
            continue
        bvh_idx = joint_name_to_idx.get(bvh_name)
        if bvh_idx is None:
            raise ValueError(f"Mapped BVH joint {bvh_name!r} was not found")
        smpl_rot = coord_rot * local_rots[bvh_idx] * coord_rot_inv
        if smpl_idx == 0:
            # grad_fit_h1.py converts SMPL root to H1 root by multiplying this
            # fixed alignment inverse. Store the inverse operation here so an
            # upright BVH root remains upright after H1 retargeting.
            smpl_rot = smpl_rot * smpl_to_h1_root_alignment
        poses[:, smpl_idx, :] = smpl_rot.as_rotvec()

    return {
        "poses": poses.reshape(motion.frames, 72).astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": np.zeros(16, dtype=np.float32),
        "gender": "neutral",
        "mocap_framerate": np.array(_fps_from_frame_time(motion.frame_time), dtype=np.float32),
    }


def _relative_yaml_entries(output_files: Iterable[Path], amass_root: Path) -> list[str]:
    entries = []
    for output_file in output_files:
        entries.append(output_file.relative_to(amass_root).as_posix())
    return sorted(entries)


def _write_yaml(entries: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "motions:\n" + "".join(f'  - "{entry}"\n' for entry in entries)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("bvh_demo/bvh"))
    parser.add_argument(
        "--amass-root",
        type=Path,
        default=Path("third_party/human2humanoid/data/AMASS/AMASS_Complete"),
    )
    parser.add_argument("--dataset-name", default="UPSTREAM_BVH_DEMO")
    parser.add_argument("--yaml-out", type=Path, default=Path("bvh_demo/upstream_bvh_demo.yaml"))
    parser.add_argument(
        "--keep-global-translation",
        action="store_true",
        help="Do not zero the first horizontal translation after coordinate conversion.",
    )
    parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help="Use the contiguous valid prefix if a BVH export has a truncated motion tail.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    amass_root = args.amass_root.resolve()
    output_dir = amass_root / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    bvh_files = sorted(input_dir.glob("*.bvh"))
    if not bvh_files:
        raise ValueError(f"No .bvh files found under {input_dir}")

    output_files = []
    for bvh_path in bvh_files:
        motion = _read_bvh(bvh_path, allow_truncated=args.allow_truncated)
        converted = _convert_motion(
            motion,
            zero_horizontal_start=not args.keep_global_translation,
        )
        output_path = output_dir / f"{bvh_path.stem}_poses.npz"
        np.savez(output_path, **converted)
        output_files.append(output_path)
        print(
            f"{bvh_path.name} -> {output_path.relative_to(Path.cwd())} "
            f"poses={converted['poses'].shape} trans={converted['trans'].shape} "
            f"fps={float(converted['mocap_framerate']):g}"
        )

    entries = _relative_yaml_entries(output_files, amass_root)
    _write_yaml(entries, args.yaml_out)
    print(f"wrote {args.yaml_out} with {len(entries)} motion entries")


if __name__ == "__main__":
    main()
