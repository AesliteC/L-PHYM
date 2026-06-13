from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json

import h5py
import numpy as np
import torch

from Script.stage1.real_moconvq_cache import (
    DEFAULT_MOCONVQ_WORLD_JSON,
    MOCONVQ_BODY_NAMES,
    ROTATION_CALIBRATION_CHOICES,
    ROTATION_SOURCE_CHOICES,
    build_loaded_moconvq_agent,
    humanml3d_joints_to_moconvq_state,
    moconvq_state_to_observation,
)


def _summarize_abs_z(abs_z: np.ndarray) -> dict[str, float]:
    flat = abs_z.reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
        "frac_gt_3": float(np.mean(flat > 3.0)),
        "frac_gt_5": float(np.mean(flat > 5.0)),
        "frac_gt_10": float(np.mean(flat > 10.0)),
    }


def describe_observation_dim(dim: int, num_body: int = 20) -> dict[str, object]:
    if dim < 0 or dim >= 16 * num_body + 3:
        raise ValueError(f"observation dim out of range for {num_body} bodies: {dim}")
    names = list(MOCONVQ_BODY_NAMES)
    if len(names) < num_body:
        names.extend([f"body_{idx}" for idx in range(len(names), num_body)])
    if dim < 3 * num_body:
        offset = dim
        return {
            "section": "local_pos",
            "body": names[offset // 3],
            "body_id": offset // 3,
            "component": ["x", "y", "z"][offset % 3],
        }
    if dim < 9 * num_body:
        offset = dim - 3 * num_body
        return {
            "section": "local_rot_6d",
            "body": names[offset // 6],
            "body_id": offset // 6,
            "component": int(offset % 6),
        }
    if dim < 12 * num_body:
        offset = dim - 9 * num_body
        return {
            "section": "local_vel",
            "body": names[offset // 3],
            "body_id": offset // 3,
            "component": ["x", "y", "z"][offset % 3],
        }
    if dim < 15 * num_body:
        offset = dim - 12 * num_body
        return {
            "section": "local_avel",
            "body": names[offset // 3],
            "body_id": offset // 3,
            "component": ["x", "y", "z"][offset % 3],
        }
    if dim < 16 * num_body:
        offset = dim - 15 * num_body
        return {
            "section": "height",
            "body": names[offset],
            "body_id": offset,
            "component": "y",
        }
    return {
        "section": "local_up_dir",
        "body": "root",
        "body_id": 0,
        "component": ["x", "y", "z"][dim - 16 * num_body],
    }


def diagnose_long_h5_observation_distribution(
    long_h5_path: Path,
    agent,
    fps: int = 20,
    max_sequences: int | None = None,
    rotation_source: str = "heuristic",
    rotation_calibration: str = "rest",
    world_json_path: Path | str | None = None,
) -> dict[str, object]:
    obs_mean = agent.obs_mean.detach().cpu().numpy().astype(np.float32)
    obs_std = agent.obs_std.detach().cpu().numpy().astype(np.float32)
    if obs_mean.shape != (323,) or obs_std.shape != (323,):
        raise ValueError(f"unexpected MoConVQ observation statistics: {obs_mean.shape}, {obs_std.shape}")

    sequence_rows: list[dict[str, object]] = []
    all_abs_z: list[np.ndarray] = []
    converted = 0

    with h5py.File(long_h5_path, "r") as h5:
        for sequence_id in h5.keys():
            if max_sequences is not None and converted >= max_sequences:
                break
            joints = h5[sequence_id]["joints_22"][:]
            joint_vecs = h5[sequence_id]["joint_vecs_263"][:] if "joint_vecs_263" in h5[sequence_id] else None
            state = humanml3d_joints_to_moconvq_state(
                joints,
                joint_vecs_263=joint_vecs,
                fps=fps,
                rotation_source=rotation_source,
                rotation_calibration=rotation_calibration,
                world_json_path=world_json_path,
            )
            observation = moconvq_state_to_observation(state)
            z = (observation - obs_mean) / (obs_std + 1e-8)
            abs_z = np.abs(z).astype(np.float32)
            all_abs_z.append(abs_z)
            row = {
                "sequence_id": str(sequence_id),
                "frames": int(observation.shape[0]),
                **_summarize_abs_z(abs_z),
            }
            sequence_rows.append(row)
            converted += 1

    if not all_abs_z:
        raise ValueError(f"no sequences found in {long_h5_path}")
    concat = np.concatenate(all_abs_z, axis=0)
    dim_p99 = np.percentile(concat, 99, axis=0)
    worst_dims = np.argsort(dim_p99)[-10:][::-1]
    return {
        "long_h5": str(long_h5_path),
        "fps": fps,
        "max_sequences": max_sequences,
        "rotation_source": rotation_source,
        "rotation_calibration": rotation_calibration,
        "world_json": str(world_json_path) if world_json_path is not None else str(DEFAULT_MOCONVQ_WORLD_JSON),
        "converted_sequences": converted,
        "aggregate_abs_z": _summarize_abs_z(concat),
        "worst_dimensions_by_p99_abs_z": [
            {
                "dim": int(dim),
                **describe_observation_dim(int(dim)),
                "p99_abs_z": float(dim_p99[dim]),
                "mean": float(obs_mean[dim]),
                "std": float(obs_std[dim]),
            }
            for dim in worst_dims
        ],
        "sequences": sequence_rows,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-h5", required=True)
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-sequences", type=int, default=20)
    parser.add_argument("--rotation-source", choices=ROTATION_SOURCE_CHOICES, default="heuristic")
    parser.add_argument("--rotation-calibration", choices=ROTATION_CALIBRATION_CHOICES, default="rest")
    parser.add_argument("--world-json", default=str(DEFAULT_MOCONVQ_WORLD_JSON))
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    agent = build_loaded_moconvq_agent(gpu=args.gpu, base_data=Path(args.base_data))
    agent.eval()
    with torch.no_grad():
        summary = diagnose_long_h5_observation_distribution(
            long_h5_path=Path(args.long_h5),
            agent=agent,
            fps=args.fps,
            max_sequences=args.max_sequences,
            rotation_source=args.rotation_source,
            rotation_calibration=args.rotation_calibration,
            world_json_path=Path(args.world_json),
        )

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
