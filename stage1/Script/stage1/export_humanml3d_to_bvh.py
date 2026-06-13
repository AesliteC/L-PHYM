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

import numpy as np
from scipy.spatial.transform import Rotation

from Script.stage1.humanml3d import load_humanml3d_catalog
from Script.stage1.real_moconvq_cache import (
    HUMANML3D_TO_MOCONVQ,
    MOCONVQ_BODY_NAMES,
    humanml3d_joint_vecs_to_global_quats_xyzw,
)
from Script.stage1.render_bvh_to_mp4 import parse_bvh


DEFAULT_TEMPLATE_BVH = Path(__file__).resolve().parents[2] / "base.bvh"
BVH_NODE_TO_MOCONVQ_BODY = {
    "RootJoint": "pelvis",
    "pelvis_lowerback": "lowerBack",
    "lowerback_torso": "torso",
    "torso_head": "head",
    "rTorso_Clavicle": "rClavicle",
    "rShoulder": "rUpperArm",
    "rElbow": "rLowerArm",
    "rWrist": "rHand",
    "lTorso_Clavicle": "lClavicle",
    "lShoulder": "lUpperArm",
    "lElbow": "lLowerArm",
    "lWrist": "lHand",
    "rHip": "rUpperLeg",
    "rKnee": "rLowerLeg",
    "rAnkle": "rFoot",
    "rToeJoint": "rToes",
    "lHip": "lUpperLeg",
    "lKnee": "lLowerLeg",
    "lAnkle": "lFoot",
    "lToeJoint": "lToes",
}


def _template_hierarchy_lines(template_bvh: Path) -> list[str]:
    lines = template_bvh.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "MOTION":
            return lines[:idx]
    raise ValueError(f"template BVH has no MOTION section: {template_bvh}")


def _moconvq_body_lookup() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(MOCONVQ_BODY_NAMES)}


def _load_humanml_motion(humanml_root: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    root = humanml_root.resolve()
    if (root / "HumanML3D").is_dir():
        root = root / "HumanML3D"
    joints = np.load(root / "new_joints" / f"{sample_id}.npy").astype(np.float32)
    joint_vecs = np.load(root / "new_joint_vecs" / f"{sample_id}.npy").astype(np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"expected HumanML3D joints shape (T, 22, 3), got {joints.shape}")
    if joint_vecs.ndim != 2 or joint_vecs.shape[1] != 263:
        raise ValueError(f"expected HumanML3D joint vecs shape (T, 263), got {joint_vecs.shape}")
    length = min(len(joints), len(joint_vecs))
    if length < 2:
        raise ValueError(f"sample {sample_id} is too short for BVH export: {length} frames")
    return joints[:length], joint_vecs[:length]


def _global_moconvq_rotations(joint_vecs_263: np.ndarray) -> np.ndarray:
    humanml_global = humanml3d_joint_vecs_to_global_quats_xyzw(joint_vecs_263)
    mapped = humanml_global[:, HUMANML3D_TO_MOCONVQ, :]
    return Rotation.from_quat(mapped.reshape(-1, 4)).as_matrix().reshape(len(mapped), len(MOCONVQ_BODY_NAMES), 3, 3)


def _local_rotation_matrix(
    node_id: int,
    node_to_body: dict[int, int],
    parent_id: int | None,
    global_mats: np.ndarray,
) -> np.ndarray:
    body_id = node_to_body[node_id]
    current = global_mats[:, body_id]
    if parent_id is None or parent_id not in node_to_body:
        return current
    parent_body_id = node_to_body[parent_id]
    parent = global_mats[:, parent_body_id]
    return np.matmul(np.swapaxes(parent, -1, -2), current)


def humanml3d_sample_to_bvh_motion(
    joints_22: np.ndarray,
    joint_vecs_263: np.ndarray,
    template_bvh: Path = DEFAULT_TEMPLATE_BVH,
) -> np.ndarray:
    nodes, _template_motion, _frame_time = parse_bvh(template_bvh)
    body_by_name = _moconvq_body_lookup()
    node_to_body: dict[int, int] = {}
    for node_id, node in enumerate(nodes):
        if not node.channels:
            continue
        body_name = BVH_NODE_TO_MOCONVQ_BODY.get(node.name)
        if body_name is None:
            raise ValueError(f"no MoConVQ body mapping for BVH node {node.name!r}")
        node_to_body[node_id] = body_by_name[body_name]

    root_positions = joints_22[:, 0, :].astype(np.float64)
    global_mats = _global_moconvq_rotations(joint_vecs_263)
    rows: list[list[float]] = []
    for frame_id in range(len(joint_vecs_263)):
        row: list[float] = []
        for node_id, node in enumerate(nodes):
            if not node.channels:
                continue
            local_mats = _local_rotation_matrix(
                node_id=node_id,
                node_to_body=node_to_body,
                parent_id=node.parent,
                global_mats=global_mats,
            )
            euler_xyz = Rotation.from_matrix(local_mats[frame_id]).as_euler("XYZ", degrees=True)
            for channel in node.channels:
                axis = channel[0].upper()
                if channel.endswith("position"):
                    component = {"X": 0, "Y": 1, "Z": 2}[axis]
                    row.append(float(root_positions[frame_id, component]))
                elif channel.endswith("rotation"):
                    component = {"X": 0, "Y": 1, "Z": 2}[axis]
                    row.append(float(euler_xyz[component]))
                else:
                    raise ValueError(f"unsupported BVH channel: {channel}")
        rows.append(row)
    motion = np.asarray(rows, dtype=np.float64)
    expected_channels = sum(len(node.channels) for node in nodes)
    if motion.shape != (len(joint_vecs_263), expected_channels):
        raise ValueError(f"unexpected exported motion shape: {motion.shape}, expected {(len(joint_vecs_263), expected_channels)}")
    if not np.isfinite(motion).all():
        raise ValueError("exported BVH motion contains non-finite values")
    return motion


def write_humanml3d_bvh(
    sample_id: str,
    humanml_root: Path,
    output_bvh: Path,
    template_bvh: Path = DEFAULT_TEMPLATE_BVH,
    output_fps: float = 20.0,
) -> dict[str, object]:
    joints, joint_vecs = _load_humanml_motion(humanml_root, sample_id)
    motion = humanml3d_sample_to_bvh_motion(joints, joint_vecs, template_bvh=template_bvh)
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
    caption = ""
    catalog = None
    try:
        catalog = load_humanml3d_catalog(humanml_root)
    except Exception:
        catalog = None
    if catalog is not None and sample_id in catalog.by_id:
        captions = catalog.by_id[sample_id].captions
        caption = captions[0]["raw"] if captions else ""
    return {
        "sample_id": sample_id,
        "output_bvh": str(output_bvh),
        "template_bvh": str(template_bvh),
        "frames": int(motion.shape[0]),
        "channels": int(motion.shape[1]),
        "frame_time": float(1.0 / float(output_fps)),
        "caption": caption,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml-root", default="../HumanML3D")
    parser.add_argument("--sample-id", action="append", default=[], help="HumanML3D sample id to export.")
    parser.add_argument("--template-bvh", default=str(DEFAULT_TEMPLATE_BVH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--summary", default="")
    args = parser.parse_args(argv)

    if not args.sample_id:
        raise SystemExit("provide at least one --sample-id")
    output_dir = Path(args.output_dir)
    summaries = [
        write_humanml3d_bvh(
            sample_id=sample_id,
            humanml_root=Path(args.humanml_root),
            output_bvh=output_dir / f"{sample_id}.bvh",
            template_bvh=Path(args.template_bvh),
            output_fps=args.fps,
        )
        for sample_id in args.sample_id
    ]
    text = json.dumps({"exports": summaries}, indent=2, ensure_ascii=False)
    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
