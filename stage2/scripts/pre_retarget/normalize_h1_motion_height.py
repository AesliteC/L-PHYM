#!/usr/bin/env python3
"""Normalize H1 retargeted motion height.

This post-processes a HOVER/H1 motion pkl by shifting root_trans_offset[:, 2].
It is useful when a retargeted motion starts above the ground and the robot
falls before the policy can stabilize.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
HUMAN2HUMANOID_PHC = REPO_ROOT / "third_party" / "human2humanoid" / "phc"
if str(HUMAN2HUMANOID_PHC) not in sys.path:
    sys.path.insert(0, str(HUMAN2HUMANOID_PHC))

from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch  # noqa: E402


def compute_min_height(
    h1_fk: Humanoid_Batch,
    motion: dict,
    mode: str,
) -> float:
    pose = torch.as_tensor(motion["pose_aa"], dtype=torch.float32)[None]
    trans = torch.as_tensor(motion["root_trans_offset"], dtype=torch.float32)[None]
    fps = int(motion.get("fps", 30))
    fk_return = h1_fk.fk_batch(pose, trans, return_full=True, dt=1.0 / fps)
    z = fk_return.global_translation_extend[0, :, :, 2]
    if mode == "first-frame":
        return float(z[0].min().item())
    if mode == "global":
        return float(z.min().item())
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pkl", type=Path)
    parser.add_argument("output_pkl", type=Path)
    parser.add_argument(
        "--mode",
        choices=["first-frame", "global"],
        default="first-frame",
        help="Which body minimum height to align to target-min-z.",
    )
    parser.add_argument(
        "--target-min-z",
        type=float,
        default=0.08,
        help="Target minimum link height after shifting.",
    )
    parser.add_argument(
        "--mjcf-file",
        type=Path,
        default=REPO_ROOT / "neural_wbc" / "data" / "data" / "motion_lib" / "h1.xml",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print per-motion height diagnostics without writing output.",
    )
    args = parser.parse_args()

    data = joblib.load(args.input_pkl)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict motion dataset, got {type(data).__name__}: {args.input_pkl}")

    h1_fk = Humanoid_Batch(
        mjcf_file=str(args.mjcf_file),
        extend_hand=True,
        extend_head=True,
        device=torch.device("cpu"),
    )

    output = {}
    for key, motion in data.items():
        motion_out = dict(motion)
        min_z = compute_min_height(h1_fk=h1_fk, motion=motion, mode=args.mode)
        shift_z = args.target_min_z - min_z
        root_trans = np.asarray(motion["root_trans_offset"]).copy()
        print(f"{key}: {args.mode} min_z={min_z:.4f}, shift_z={shift_z:+.4f}")
        if not args.inspect_only:
            root_trans[:, 2] += shift_z
            motion_out["root_trans_offset"] = root_trans
        output[key] = motion_out

    if args.inspect_only:
        return

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, args.output_pkl)
    print(f"Wrote normalized motion pkl: {args.output_pkl}")


if __name__ == "__main__":
    main()
