from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import glob
import importlib.util
import json
import sys

import numpy as np

if __package__ in {None, ""}:
    repo_root = str(Path(__file__).resolve().parents[2])
    if not sys.path or sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)

from Script.stage1.render_bvh_to_mp4 import Node, frame_positions, parse_bvh


DIRECT_BVH_NODE_TO_HUMANML3D = {
    "RootJoint": 0,
    "lHip": 1,
    "rHip": 2,
    "pelvis_lowerback": 3,
    "lKnee": 4,
    "rKnee": 5,
    "lowerback_torso": 6,
    "lAnkle": 7,
    "rAnkle": 8,
    "lToeJoint": 10,
    "rToeJoint": 11,
    "lTorso_Clavicle": 13,
    "rTorso_Clavicle": 14,
    "lShoulder": 16,
    "rShoulder": 17,
    "lElbow": 18,
    "rElbow": 19,
    "lWrist": 20,
    "rWrist": 21,
}

APPROXIMATION_NOTE = (
    "Approximate adapter from MoConVQ/base.bvh skeleton to HumanML3D 22-joint "
    "positions. Directly corresponding joints are copied by name; HumanML3D "
    "spine/neck/head joints 9, 12, and 15 are approximated from the BVH torso_head "
    "joint and its end site before HumanML3D process_file() builds 263-d features."
)


def collect_bvh_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.bvh")))
        else:
            matches = [Path(item) for item in sorted(glob.glob(raw))]
            files.extend(matches if matches else [path])
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def resolve_humanml_data_root(path: Path) -> Path:
    root = path.resolve()
    if (root / "new_joints").is_dir() and (root / "new_joint_vecs").is_dir():
        return root
    nested = root / "HumanML3D"
    if (nested / "new_joints").is_dir() and (nested / "new_joint_vecs").is_dir():
        return nested
    raise ValueError(f"could not find HumanML3D new_joints/new_joint_vecs under {path}")


def _load_humanml_representation_module(humanml_data_root: Path):
    repo_root = humanml_data_root.parent
    module_path = repo_root / "scripts" / "generate_motion_representation.py"
    if not module_path.exists():
        raise ValueError(f"HumanML3D motion representation script not found: {module_path}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    spec = importlib.util.spec_from_file_location("stage1_humanml_motion_representation", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not import HumanML3D representation module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node_name_to_index(nodes: list[Node]) -> dict[str, int]:
    return {node.name: idx for idx, node in enumerate(nodes)}


def _node_position(
    positions: np.ndarray,
    node_by_name: dict[str, int],
    name: str,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    node_id = node_by_name.get(name)
    if node_id is None:
        if fallback is None:
            raise ValueError(f"BVH node {name!r} is required for HumanML3D conversion")
        return fallback
    return positions[:, node_id]


def bvh_positions_to_humanml3d_joints(nodes: list[Node], positions: np.ndarray) -> np.ndarray:
    if positions.ndim != 3 or positions.shape[1] != len(nodes) or positions.shape[2] != 3:
        raise ValueError(f"expected positions shape (T, {len(nodes)}, 3), got {positions.shape}")
    node_by_name = _node_name_to_index(nodes)
    joints = np.zeros((positions.shape[0], 22, 3), dtype=np.float32)

    for node_name, joint_id in DIRECT_BVH_NODE_TO_HUMANML3D.items():
        joints[:, joint_id] = _node_position(positions, node_by_name, node_name)

    lower_torso = joints[:, 6]
    torso_head = _node_position(positions, node_by_name, "torso_head", fallback=lower_torso)
    head_end = _node_position(positions, node_by_name, "torso_head_end_0", fallback=torso_head)

    # HumanML3D has a longer spine chain than base.bvh.  These interpolants keep
    # the chain non-degenerate before HumanML3D's own uniform_skeleton step.
    joints[:, 9] = 0.5 * lower_torso + 0.5 * torso_head
    joints[:, 12] = torso_head
    joints[:, 15] = head_end
    return joints


def resample_positions(positions: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if len(positions) < 2:
        raise ValueError("at least two BVH frames are required")
    if abs(source_fps - target_fps) < 1e-6:
        return positions.astype(np.float32, copy=True)

    source_times = np.arange(len(positions), dtype=np.float64) / float(source_fps)
    target_times = np.arange(0.0, source_times[-1] + 1e-9, 1.0 / float(target_fps))
    if len(target_times) < 2:
        target_times = np.asarray([0.0, source_times[-1]], dtype=np.float64)
    flat = positions.reshape(len(positions), -1)
    resampled = np.empty((len(target_times), flat.shape[1]), dtype=np.float64)
    for dim in range(flat.shape[1]):
        resampled[:, dim] = np.interp(target_times, source_times, flat[:, dim])
    return resampled.reshape(len(target_times), positions.shape[1], positions.shape[2]).astype(np.float32)


def load_bvh_as_humanml3d_joints(path: Path, target_fps: float = 20.0) -> tuple[np.ndarray, dict[str, object]]:
    nodes, motion, frame_time = parse_bvh(path)
    source_fps = 1.0 / frame_time if frame_time > 0 else target_fps
    positions = np.stack([frame_positions(nodes, row) for row in motion], axis=0)
    resampled = resample_positions(positions, source_fps=source_fps, target_fps=target_fps)
    joints = bvh_positions_to_humanml3d_joints(nodes, resampled)
    return joints, {
        "input_bvh": str(path),
        "source_frames": int(motion.shape[0]),
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "resampled_frames": int(joints.shape[0]),
        "approximation_note": APPROXIMATION_NOTE,
    }


def convert_bvh_to_humanml3d_features(
    path: Path,
    humanml_data_root: Path,
    output_vecs: Path,
    output_joints: Path | None = None,
    target_fps: float = 20.0,
    feet_threshold: float = 0.002,
    example_id: str = "000021",
) -> dict[str, object]:
    module = _load_humanml_representation_module(humanml_data_root)
    joints, summary = load_bvh_as_humanml3d_joints(path, target_fps=target_fps)
    context = module.build_context(str(humanml_data_root / "new_joints"), example_id)
    features = module.process_file(joints, feet_threshold, context).astype(np.float32)
    if features.ndim != 2 or features.shape[1] != 263:
        raise ValueError(f"expected HumanML3D 263-d features, got {features.shape}")

    output_vecs.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_vecs, features)
    if output_joints is not None:
        output_joints.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_joints, joints.astype(np.float32))

    summary.update(
        {
            "output_vecs": str(output_vecs),
            "output_joints": str(output_joints) if output_joints is not None else "",
            "feature_frames": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "feet_threshold": float(feet_threshold),
            "example_id": str(example_id),
        }
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="BVH files, directories, or glob patterns")
    parser.add_argument("--humanml-root", default="/home/chenjie/cc/robotics/HumanML3D")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-joints", action="store_true")
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--feet-threshold", type=float, default=0.002)
    parser.add_argument("--example-id", default="000021")
    parser.add_argument("--summary", default="")
    args = parser.parse_args(argv)

    humanml_data_root = resolve_humanml_data_root(Path(args.humanml_root))
    output_dir = Path(args.output_dir)
    vecs_dir = output_dir / "new_joint_vecs"
    joints_dir = output_dir / "new_joints" if args.save_joints else None
    summaries = []
    for bvh in collect_bvh_files(args.inputs):
        output_vecs = vecs_dir / f"{bvh.stem}.npy"
        output_joints = joints_dir / f"{bvh.stem}.npy" if joints_dir is not None else None
        summaries.append(
            convert_bvh_to_humanml3d_features(
                bvh,
                humanml_data_root=humanml_data_root,
                output_vecs=output_vecs,
                output_joints=output_joints,
                target_fps=args.target_fps,
                feet_threshold=args.feet_threshold,
                example_id=args.example_id,
            )
        )

    payload = {
        "humanml_data_root": str(humanml_data_root),
        "output_dir": str(output_dir),
        "count": len(summaries),
        "approximation_note": APPROXIMATION_NOTE,
        "rows": summaries,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
