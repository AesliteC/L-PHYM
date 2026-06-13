from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import traceback

import h5py
import numpy as np

from Script.stage1.real_moconvq_cache import (
    DEFAULT_MOCONVQ_WORLD_JSON,
    ROTATION_CALIBRATION_CHOICES,
    ROTATION_SOURCE_CHOICES,
    humanml3d_joints_to_moconvq_state,
    load_manifest,
    moconvq_state_to_observation,
)


def convert_long_h5_to_moconvq_observation(
    long_h5_path: Path,
    manifest_path: Path,
    output_h5_path: Path,
    fps: int = 20,
    rotation_source: str = "heuristic",
    rotation_calibration: str = "rest",
    world_json_path: Path | str | None = None,
) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    converted = 0
    frame_counts: list[int] = []

    with h5py.File(long_h5_path, "r") as source, h5py.File(output_h5_path, "w") as target:
        for sequence_id in source.keys():
            try:
                row = manifest.get(sequence_id, {})
                source_group = source[sequence_id]
                joints = source_group["joints_22"][:]
                joint_vecs = source_group["joint_vecs_263"][:] if "joint_vecs_263" in source_group else None
                state = humanml3d_joints_to_moconvq_state(
                    joints,
                    joint_vecs_263=joint_vecs,
                    fps=fps,
                    rotation_source=rotation_source,
                    rotation_calibration=rotation_calibration,
                    world_json_path=world_json_path,
                )
                observation = moconvq_state_to_observation(state)

                group = target.create_group(sequence_id)
                group.create_dataset("state_20x13", data=state, compression="gzip")
                group.create_dataset("observation_323", data=observation, compression="gzip")
                if "clip_boundaries" in source_group:
                    group.create_dataset("clip_boundaries", data=source_group["clip_boundaries"][:])
                if "transition_scores" in source_group:
                    group.create_dataset("transition_scores", data=source_group["transition_scores"][:])

                caption = str(row.get("caption") or source_group.attrs.get("caption", ""))
                group.attrs["caption"] = caption
                group.attrs["sample_ids"] = ",".join(row.get("sample_ids", [])) or str(source_group.attrs.get("sample_ids", ""))
                group.attrs["split"] = str(row.get("split") or source_group.attrs.get("split", ""))
                converted += 1
                frame_counts.append(int(observation.shape[0]))
            except Exception as exc:  # noqa: BLE001 - this script is a batch converter.
                failures.append(
                    {
                        "sequence_id": str(sequence_id),
                        "reason": str(exc),
                        "traceback_short": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                    }
                )

    return {
        "converted_sequences": converted,
        "failed_sequences": len(failures),
        "avg_frames": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "output_h5": str(output_h5_path),
        "failures": failures,
        "config": {
            "long_h5": str(long_h5_path),
            "manifest": str(manifest_path),
            "fps": fps,
            "rotation_source": rotation_source,
            "rotation_calibration": rotation_calibration,
            "world_json": str(world_json_path) if world_json_path is not None else str(DEFAULT_MOCONVQ_WORLD_JSON),
        },
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-h5", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--rotation-source", choices=ROTATION_SOURCE_CHOICES, default="heuristic")
    parser.add_argument("--rotation-calibration", choices=ROTATION_CALIBRATION_CHOICES, default="rest")
    parser.add_argument("--world-json", default=str(DEFAULT_MOCONVQ_WORLD_JSON))
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args(argv)

    summary = convert_long_h5_to_moconvq_observation(
        long_h5_path=Path(args.long_h5),
        manifest_path=Path(args.manifest),
        output_h5_path=Path(args.output_h5),
        fps=args.fps,
        rotation_source=args.rotation_source,
        rotation_calibration=args.rotation_calibration,
        world_json_path=Path(args.world_json),
    )
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if summary["failed_sequences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
