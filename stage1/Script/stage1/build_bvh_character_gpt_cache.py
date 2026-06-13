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
import torch

from Script.stage1.build_native_moconvq_gpt_cache import (
    build_native_cache_from_h5,
    summarize_cache,
)
from Script.stage1.diagnose_bvh_character_retarget import extract_bvh_with_moconvq_character
from Script.stage1.real_moconvq_cache import build_loaded_moconvq_agent, build_t5_text_encoder


def parse_bvh_specs(values: list[str]) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("BVH specs must be formatted as '<path.bvh>=<caption>'")
        path, caption = value.split("=", 1)
        bvh_path = Path(path.strip())
        caption = caption.strip()
        if not str(bvh_path) or not caption:
            raise ValueError(f"invalid BVH spec: {value!r}")
        specs.append((bvh_path, caption))
    return specs


def specs_from_quality_summary(path: Path, accepted_only: bool = True) -> list[tuple[Path, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs: list[tuple[Path, str]] = []
    for row in payload.get("rows", []):
        if accepted_only and not bool(row.get("accepted")):
            continue
        bvh_path = Path(str(row["path"]))
        caption = str(row.get("caption", "")).strip() or bvh_path.stem
        specs.append((bvh_path, caption))
    if not specs:
        raise ValueError(f"quality summary produced no BVH specs: {path}")
    return specs


def _write_observation_h5(path: Path, rows: list[tuple[str, np.ndarray, str]]) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, observation, caption in rows:
            group = handle.create_group(key)
            group.create_dataset("observation", data=np.asarray(observation, dtype=np.float32), compression="gzip")
            group.attrs["caption"] = caption


def build_bvh_character_cache(
    bvh_specs: list[tuple[Path, str]],
    agent,
    text_encoder,
    output_observation_h5: Path,
    window_size: int,
    window_stride: int,
    rvq_depth: int = 4,
    fps: int = 20,
    flip: bool = False,
    text_model: str | None = None,
    max_text_length: int | None = None,
) -> dict[str, object]:
    if not bvh_specs:
        raise ValueError("at least one BVH spec is required")
    rows: list[tuple[str, np.ndarray, str]] = []
    motion_specs: list[tuple[str, str]] = []
    for idx, (bvh_path, caption) in enumerate(bvh_specs):
        if not bvh_path.exists():
            raise FileNotFoundError(str(bvh_path))
        motion = extract_bvh_with_moconvq_character([bvh_path], agent=agent, fps=fps, flip=flip)
        key = f"{idx:04d}_{bvh_path.stem}"
        rows.append((key, motion["observation"], caption))
        motion_specs.append((key, caption))
    _write_observation_h5(output_observation_h5, rows)
    cache = build_native_cache_from_h5(
        native_h5_path=output_observation_h5,
        motion_specs=motion_specs,
        agent=agent,
        text_encoder=text_encoder,
        window_size=window_size,
        window_stride=window_stride,
        rvq_depth=rvq_depth,
        text_model=text_model,
        max_text_length=max_text_length,
    )
    cache["config"]["source"] = "bvh_character_moconvq_observation"  # type: ignore[index]
    cache["config"]["bvh_specs"] = [  # type: ignore[index]
        {"path": str(path), "caption": caption} for path, caption in bvh_specs
    ]
    cache["config"]["fps"] = fps  # type: ignore[index]
    cache["config"]["flip"] = flip  # type: ignore[index]
    cache["config"]["intermediate_observation_h5"] = str(output_observation_h5)  # type: ignore[index]
    return cache


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh",
        action="append",
        default=[],
        help="BVH spec formatted as '<path.bvh>=<caption>'.",
    )
    parser.add_argument(
        "--motion",
        action="append",
        default=[],
        help="Alias for --bvh; accepted for consistency with native cache specs.",
    )
    parser.add_argument(
        "--quality-summary",
        default="",
        help="Optional summarize_bvh_retarget_quality.py JSON; accepted rows are converted to BVH specs.",
    )
    parser.add_argument(
        "--include-rejected-quality",
        action="store_true",
        help="Use all quality-summary rows instead of accepted rows only.",
    )
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--motion-dataset", default="")
    parser.add_argument("--text-model", default="t5-large")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--window-stride", type=int, default=25)
    parser.add_argument("--rvq-depth", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observation-h5", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args(argv)

    raw_specs = args.bvh + args.motion
    bvh_specs = parse_bvh_specs(raw_specs)
    if args.quality_summary:
        bvh_specs.extend(
            specs_from_quality_summary(
                Path(args.quality_summary),
                accepted_only=not args.include_rejected_quality,
            )
        )
    if not bvh_specs:
        raise SystemExit("provide at least one --bvh '<path.bvh>=<caption>' or --quality-summary")

    import MoConVQCore.Utils.pytorch_utils as ptu

    agent = build_loaded_moconvq_agent(
        gpu=args.gpu,
        base_data=Path(args.base_data),
        motion_dataset=Path(args.motion_dataset) if args.motion_dataset else None,
    )
    text_encoder = build_t5_text_encoder(args.text_model, device=str(ptu.device), max_length=args.max_text_length)
    cache = build_bvh_character_cache(
        bvh_specs=bvh_specs,
        agent=agent,
        text_encoder=text_encoder,
        output_observation_h5=Path(args.observation_h5),
        window_size=args.window_size,
        window_stride=args.window_stride,
        rvq_depth=args.rvq_depth,
        fps=args.fps,
        flip=args.flip,
        text_model=args.text_model,
        max_text_length=args.max_text_length,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    summary = summarize_cache(cache)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
