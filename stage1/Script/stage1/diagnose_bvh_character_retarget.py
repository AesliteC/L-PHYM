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

import h5py
import numpy as np
import torch

from Script.stage1.diagnose_observation_distribution import _summarize_abs_z
from Script.stage1.diagnose_token_distribution import (
    compact_stats,
    compare_distributions,
    token_distribution_stats,
)
from Script.stage1.real_moconvq_cache import (
    _indices_to_time_depth,
    build_loaded_moconvq_agent,
)


def extract_bvh_with_moconvq_character(
    bvh_files: list[Path],
    agent,
    fps: int = 20,
    flip: bool = False,
) -> dict[str, np.ndarray]:
    """Use MoConVQ's original BVH-to-character path to obtain state/observation."""

    from MoConVQCore.Utils.motion_dataset import MotionDataSet

    motion_data = MotionDataSet(fps)
    for bvh_file in bvh_files:
        motion_data.add_bvh_with_character(str(bvh_file), agent.env.sim_character, flip=flip)
    if motion_data.observation is None or motion_data.state is None:
        raise ValueError("MoConVQ MotionDataSet did not produce state/observation")
    return {
        "state": np.asarray(motion_data.state, dtype=np.float32),
        "observation": np.asarray(motion_data.observation, dtype=np.float32),
        "done": np.asarray(motion_data.done),
    }


def summarize_observation_against_agent(agent, observation: np.ndarray) -> dict[str, object]:
    obs_mean = agent.obs_mean.detach().cpu().numpy().astype(np.float32)
    obs_std = agent.obs_std.detach().cpu().numpy().astype(np.float32)
    if observation.ndim != 2 or observation.shape[-1] != obs_mean.shape[0]:
        raise ValueError(
            "observation shape does not match MoConVQ statistics: "
            f"observation={observation.shape}, obs_mean={obs_mean.shape}"
        )
    abs_z = np.abs((observation.astype(np.float32) - obs_mean) / (obs_std + 1e-8))
    dim_p99 = np.percentile(abs_z, 99, axis=0)
    worst_dims = np.argsort(dim_p99)[-10:][::-1]
    return {
        "aggregate_abs_z": _summarize_abs_z(abs_z),
        "worst_dimensions_by_p99_abs_z": [
            {
                "dim": int(dim),
                "p99_abs_z": float(dim_p99[dim]),
                "mean": float(obs_mean[dim]),
                "std": float(obs_std[dim]),
            }
            for dim in worst_dims
        ],
    }


def encode_observation_indices(agent, observation: np.ndarray, rvq_depth: int = 4) -> torch.Tensor:
    with torch.no_grad():
        info = agent.encode_seq_all(None, observation)
    return torch.as_tensor(_indices_to_time_depth(info["indexs"], rvq_depth=rvq_depth), dtype=torch.long)


def summarize_native_observation_tokens(
    native_h5: Path,
    observation_key: str,
    agent,
    rvq_depth: int = 4,
) -> dict[str, object]:
    with h5py.File(native_h5, "r") as handle:
        observation = np.asarray(handle[observation_key], dtype=np.float32)
    indices = encode_observation_indices(agent, observation, rvq_depth=rvq_depth)
    return {
        "kind": "native_h5",
        "path": str(native_h5),
        "observation_key": observation_key,
        "observation_shape": list(observation.shape),
        "shape": list(indices.shape),
        "stats": token_distribution_stats(indices),
    }


def diagnose_bvh_character_retarget(
    bvh_files: list[Path],
    agent,
    fps: int = 20,
    rvq_depth: int = 4,
    flip: bool = False,
    native_h5: Path | None = None,
    native_observation_key: str = "walk1_subject5/observation",
) -> dict[str, object]:
    motion = extract_bvh_with_moconvq_character(bvh_files, agent=agent, fps=fps, flip=flip)
    observation = motion["observation"]
    indices = encode_observation_indices(agent, observation, rvq_depth=rvq_depth)
    bvh_summary = {
        "kind": "bvh_character",
        "paths": [str(path) for path in bvh_files],
        "fps": fps,
        "flip": flip,
        "state_shape": list(motion["state"].shape),
        "observation_shape": list(observation.shape),
        "shape": list(indices.shape),
        "observation_z": summarize_observation_against_agent(agent, observation),
        "stats": token_distribution_stats(indices),
    }

    summaries: list[dict[str, object]] = [bvh_summary]
    comparisons: list[dict[str, object]] = []
    if native_h5 is not None:
        native_summary = summarize_native_observation_tokens(
            native_h5,
            observation_key=native_observation_key,
            agent=agent,
            rvq_depth=rvq_depth,
        )
        summaries.append(native_summary)
        comparisons.append(
            {
                "left": "bvh_character",
                "right": str(native_h5),
                "by_depth": compare_distributions(bvh_summary, native_summary),
            }
        )

    return {
        "summaries": summaries,
        "comparisons": comparisons,
    }


def _compact_payload(payload: dict[str, object]) -> dict[str, object]:
    compact = dict(payload)
    compact["summaries"] = [compact_stats(summary) for summary in payload["summaries"]]  # type: ignore[index]
    return compact


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bvh_files", nargs="+")
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--motion-dataset", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--rvq-depth", type=int, default=4)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--native-h5", default="")
    parser.add_argument("--native-observation-key", default="walk1_subject5/observation")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    agent = build_loaded_moconvq_agent(
        gpu=args.gpu,
        base_data=Path(args.base_data),
        motion_dataset=Path(args.motion_dataset) if args.motion_dataset else None,
    )
    agent.eval()
    payload = diagnose_bvh_character_retarget(
        [Path(path) for path in args.bvh_files],
        agent=agent,
        fps=args.fps,
        rvq_depth=args.rvq_depth,
        flip=args.flip,
        native_h5=Path(args.native_h5) if args.native_h5 else None,
        native_observation_key=args.native_observation_key,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(_compact_payload(payload), indent=2))


if __name__ == "__main__":
    main()
