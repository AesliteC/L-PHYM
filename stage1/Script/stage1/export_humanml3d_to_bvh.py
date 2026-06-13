from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import random
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

from Script.stage1.humanml3d import HumanML3DCatalog, load_humanml3d_catalog
from Script.stage1.real_moconvq_cache import (
    HUMANML3D_TO_MOCONVQ,
    MOCONVQ_BODY_NAMES,
    humanml3d_joint_vecs_to_global_quats_xyzw,
)
from Script.stage1.render_bvh_to_mp4 import parse_bvh


DEFAULT_TEMPLATE_BVH = Path(__file__).resolve().parents[2] / "base.bvh"
ROTATION_SOURCE_CHOICES = ("joints_ik", "vec6d")
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


def _node_to_moconvq_body(nodes) -> dict[int, int]:
    body_by_name = _moconvq_body_lookup()
    node_to_body: dict[int, int] = {}
    for node_id, node in enumerate(nodes):
        if not node.channels:
            continue
        body_name = BVH_NODE_TO_MOCONVQ_BODY.get(node.name)
        if body_name is None:
            raise ValueError(f"no MoConVQ body mapping for BVH node {node.name!r}")
        node_to_body[node_id] = body_by_name[body_name]
    return node_to_body


def _children_by_node(nodes) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in nodes]
    for node_id, node in enumerate(nodes):
        if node.parent is not None:
            children[node.parent].append(node_id)
    return children


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


def _vec6d_local_rotation_matrices(nodes, joint_vecs_263: np.ndarray, node_to_body: dict[int, int]) -> np.ndarray:
    global_mats = _global_moconvq_rotations(joint_vecs_263)
    local_mats = np.tile(np.eye(3, dtype=np.float64), (len(joint_vecs_263), len(nodes), 1, 1))
    for node_id, node in enumerate(nodes):
        if not node.channels:
            continue
        local_mats[:, node_id] = _local_rotation_matrix(
            node_id=node_id,
            node_to_body=node_to_body,
            parent_id=node.parent,
            global_mats=global_mats,
        )
    return local_mats


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


def _unit_vector(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return vec.astype(np.float64) / norm


def _single_vector_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_unit = _unit_vector(source)
    target_unit = _unit_vector(target)
    if source_unit is None or target_unit is None:
        return np.eye(3, dtype=np.float64)
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-8:
        axis = np.cross(source_unit, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(source_unit, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    axis = np.cross(source_unit, target_unit)
    axis_norm = max(float(np.linalg.norm(axis)), 1e-8)
    angle = np.arctan2(axis_norm, dot)
    return Rotation.from_rotvec(axis / axis_norm * angle).as_matrix()


def _align_vectors_matrix(source_vectors: np.ndarray, target_vectors: np.ndarray) -> np.ndarray:
    source_units = []
    target_units = []
    for source, target in zip(source_vectors, target_vectors):
        source_unit = _unit_vector(source)
        target_unit = _unit_vector(target)
        if source_unit is None or target_unit is None:
            continue
        source_units.append(source_unit)
        target_units.append(target_unit)
    if not source_units:
        return np.eye(3, dtype=np.float64)
    if len(source_units) == 1:
        return _single_vector_alignment(source_units[0], target_units[0])
    source_arr = np.asarray(source_units)
    target_arr = np.asarray(target_units)
    if np.linalg.matrix_rank(source_arr, tol=1e-5) < 2 or np.linalg.matrix_rank(target_arr, tol=1e-5) < 2:
        return _single_vector_alignment(source_arr[0], target_arr[0])
    rotation, _rmsd = Rotation.align_vectors(target_arr, source_arr)
    return rotation.as_matrix()


def _joints_ik_local_rotation_matrices(nodes, joints_22: np.ndarray, node_to_body: dict[int, int]) -> np.ndarray:
    """Estimate BVH local rotations from HumanML3D joint positions.

    HumanML3D's 6D rotations live in the T2M skeleton frame, while `base.bvh`
    uses MoConVQ rigid-body frames.  For BVH export, matching child bone
    directions from `new_joints` is a more conservative bridge than treating the
    6D rotations as directly compatible BVH local rotations.
    """

    target_positions = joints_22[:, HUMANML3D_TO_MOCONVQ, :].astype(np.float64)
    children = _children_by_node(nodes)
    frames = len(joints_22)
    local_mats = np.tile(np.eye(3, dtype=np.float64), (frames, len(nodes), 1, 1))
    global_mats = np.tile(np.eye(3, dtype=np.float64), (frames, len(nodes), 1, 1))
    for frame_id in range(frames):
        for node_id, node in enumerate(nodes):
            parent_global = np.eye(3, dtype=np.float64)
            if node.parent is not None:
                parent_global = global_mats[frame_id, node.parent]
            if not node.channels:
                global_mats[frame_id, node_id] = parent_global
                continue

            source_vectors: list[np.ndarray] = []
            target_vectors: list[np.ndarray] = []
            body_id = node_to_body[node_id]
            for child_id in children[node_id]:
                if child_id not in node_to_body:
                    continue
                child_body_id = node_to_body[child_id]
                source_vectors.append(nodes[child_id].offset.astype(np.float64))
                target_vectors.append(target_positions[frame_id, child_body_id] - target_positions[frame_id, body_id])

            if source_vectors:
                desired_global = _align_vectors_matrix(np.asarray(source_vectors), np.asarray(target_vectors))
                local = parent_global.T @ desired_global
            else:
                local = np.eye(3, dtype=np.float64)
                desired_global = parent_global
            local_mats[frame_id, node_id] = local
            global_mats[frame_id, node_id] = desired_global
    return local_mats


def _local_eulers_xyz_degrees(nodes, local_mats: np.ndarray, unwrap_euler: bool) -> dict[int, np.ndarray]:
    eulers: dict[int, np.ndarray] = {}
    for node_id, node in enumerate(nodes):
        if not node.channels:
            continue
        radians = Rotation.from_matrix(local_mats[:, node_id]).as_euler("XYZ", degrees=False)
        if unwrap_euler:
            radians = np.unwrap(radians, axis=0)
        eulers[node_id] = np.degrees(radians)
    return eulers


def humanml3d_sample_to_bvh_motion(
    joints_22: np.ndarray,
    joint_vecs_263: np.ndarray,
    template_bvh: Path = DEFAULT_TEMPLATE_BVH,
    rotation_source: str = "joints_ik",
    unwrap_euler: bool = True,
) -> np.ndarray:
    if rotation_source not in ROTATION_SOURCE_CHOICES:
        raise ValueError(f"unknown rotation_source {rotation_source!r}; expected one of {ROTATION_SOURCE_CHOICES}")
    nodes, _template_motion, _frame_time = parse_bvh(template_bvh)
    node_to_body = _node_to_moconvq_body(nodes)
    root_positions = joints_22[:, 0, :].astype(np.float64)
    if rotation_source == "vec6d":
        local_mats = _vec6d_local_rotation_matrices(nodes, joint_vecs_263, node_to_body)
    else:
        local_mats = _joints_ik_local_rotation_matrices(nodes, joints_22, node_to_body)
    eulers = _local_eulers_xyz_degrees(nodes, local_mats, unwrap_euler=unwrap_euler)
    rows: list[list[float]] = []
    for frame_id in range(len(joint_vecs_263)):
        row: list[float] = []
        for node_id, node in enumerate(nodes):
            if not node.channels:
                continue
            euler_xyz = eulers[node_id][frame_id]
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
    rotation_source: str = "joints_ik",
    unwrap_euler: bool = True,
    catalog: HumanML3DCatalog | None = None,
) -> dict[str, object]:
    joints, joint_vecs = _load_humanml_motion(humanml_root, sample_id)
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
    caption = ""
    if catalog is None:
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
        "rotation_source": rotation_source,
        "unwrap_euler": bool(unwrap_euler),
    }


def select_humanml3d_sample_ids(
    humanml_root: Path,
    sample_ids: Iterable[str] = (),
    split: str = "",
    limit: int | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> list[str]:
    selected = [str(sample_id) for sample_id in sample_ids]
    if split:
        catalog = load_humanml3d_catalog(humanml_root)
        if split not in catalog.split_ids:
            raise ValueError(f"unknown HumanML3D split {split!r}; expected one of {sorted(catalog.split_ids)}")
        split_ids = list(catalog.split_ids[split])
        if shuffle:
            random.Random(seed).shuffle(split_ids)
        if limit is not None:
            if limit < 1:
                raise ValueError("--limit must be positive")
            split_ids = split_ids[:limit]
        selected.extend(split_ids)

    deduped: list[str] = []
    seen: set[str] = set()
    for sample_id in selected:
        if sample_id not in seen:
            deduped.append(sample_id)
            seen.add(sample_id)
    return deduped


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml-root", default="../HumanML3D")
    parser.add_argument("--sample-id", action="append", default=[], help="HumanML3D sample id to export.")
    parser.add_argument("--split", default="", help="Optionally export a reproducible sample from this HumanML3D split.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of split samples to export.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when sampling from --split.")
    parser.add_argument("--no-shuffle", action="store_true", help="Use the first --limit ids from --split instead of shuffling.")
    parser.add_argument("--template-bvh", default=str(DEFAULT_TEMPLATE_BVH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--rotation-source", choices=ROTATION_SOURCE_CHOICES, default="joints_ik")
    parser.add_argument("--no-unwrap-euler", action="store_true")
    parser.add_argument("--summary", default="")
    parser.add_argument("--quiet", action="store_true", help="Print only a compact run summary to stdout.")
    args = parser.parse_args(argv)

    humanml_root = Path(args.humanml_root)
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
    output_dir = Path(args.output_dir)
    catalog = None
    try:
        catalog = load_humanml3d_catalog(humanml_root)
    except Exception:
        catalog = None
    summaries = [
        write_humanml3d_bvh(
            sample_id=sample_id,
            humanml_root=humanml_root,
            output_bvh=output_dir / f"{sample_id}.bvh",
            template_bvh=Path(args.template_bvh),
            output_fps=args.fps,
            rotation_source=args.rotation_source,
            unwrap_euler=not args.no_unwrap_euler,
            catalog=catalog,
        )
        for sample_id in sample_ids
    ]
    text = json.dumps(
        {
            "config": {
                "humanml_root": str(humanml_root),
                "template_bvh": args.template_bvh,
                "output_dir": str(output_dir),
                "fps": float(args.fps),
                "rotation_source": args.rotation_source,
                "unwrap_euler": not args.no_unwrap_euler,
                "sample_ids": sample_ids,
                "split": args.split,
                "limit": args.limit,
                "seed": args.seed,
                "shuffle": not args.no_shuffle,
            },
            "exports": summaries,
        },
        indent=2,
        ensure_ascii=False,
    )
    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(text, encoding="utf-8")
    if args.quiet:
        print(
            json.dumps(
                {
                    "summary": args.summary,
                    "exports": len(summaries),
                    "output_dir": str(output_dir),
                    "rotation_source": args.rotation_source,
                },
                indent=2,
            )
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
