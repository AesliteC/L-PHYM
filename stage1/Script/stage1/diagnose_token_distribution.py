from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import math

import h5py
import numpy as np
import torch


def _entropy_bits(counts: torch.Tensor) -> float:
    total = counts.sum().clamp_min(1.0)
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * torch.log2(probs)).sum())


def _js_divergence_bits(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    left = left / left.sum().clamp_min(1.0)
    right = right / right.sum().clamp_min(1.0)
    mid = 0.5 * (left + right)

    def kl(prob: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        valid = prob > 0
        return (prob[valid] * torch.log2(prob[valid] / ref[valid].clamp_min(1e-12))).sum()

    return float(0.5 * kl(left, mid) + 0.5 * kl(right, mid))


def token_distribution_stats(
    indices: torch.Tensor,
    *,
    target_mask: torch.Tensor | None = None,
    vocab_size: int = 512,
    pad_index: int = 513,
    top_k: int = 10,
) -> list[dict[str, object]]:
    if indices.ndim == 2:
        indices = indices.unsqueeze(0)
    if indices.ndim != 3:
        raise ValueError(f"expected indices shape (N,T,D) or (T,D), got {tuple(indices.shape)}")
    if target_mask is None:
        target_mask = torch.ones(indices.shape[:2], dtype=torch.bool, device=indices.device)
    elif target_mask.ndim == 1:
        target_mask = target_mask.unsqueeze(0)
    target_mask = target_mask.to(device=indices.device, dtype=torch.bool)
    if target_mask.shape != indices.shape[:2]:
        raise ValueError(f"target mask shape {tuple(target_mask.shape)} does not match {tuple(indices.shape[:2])}")

    rows = []
    for depth in range(indices.shape[-1]):
        values = indices[:, :, depth][target_mask & (indices[:, :, depth] != pad_index)]
        values = values[(values >= 0) & (values < vocab_size)]
        counts = torch.bincount(values.to(torch.long), minlength=vocab_size).float().cpu()
        top = torch.topk(counts, k=min(top_k, vocab_size))
        total = counts.sum().clamp_min(1.0)
        rows.append(
            {
                "depth": depth,
                "tokens": int(counts.sum().item()),
                "unique": int((counts > 0).sum().item()),
                "entropy_bits": _entropy_bits(counts),
                "top_ids": [int(x) for x in top.indices.tolist()],
                "top_fracs": [float(x / total) for x in top.values.tolist()],
                "counts": counts.tolist(),
            }
        )
    return rows


def summarize_cache(cache_path: Path) -> dict[str, object]:
    cache = torch.load(cache_path, map_location="cpu")
    target_mask = cache.get("target_masks")
    if target_mask is None:
        indices = torch.as_tensor(cache["indices"], dtype=torch.long)
        target_mask = indices[:, :, 0] != 513
    stats = token_distribution_stats(
        torch.as_tensor(cache["indices"], dtype=torch.long),
        target_mask=torch.as_tensor(target_mask, dtype=torch.bool),
    )
    return {
        "kind": "cache",
        "path": str(cache_path),
        "stats": stats,
        "shape": list(torch.as_tensor(cache["indices"]).shape),
    }


def summarize_native_h5(
    h5_path: Path,
    *,
    observation_key: str,
    base_data: Path,
    gpu: int,
    rvq_depth: int,
) -> dict[str, object]:
    from Script.stage1.real_moconvq_cache import build_loaded_moconvq_agent, _indices_to_time_depth

    with h5py.File(h5_path, "r") as handle:
        observation = np.asarray(handle[observation_key], dtype=np.float32)
    agent = build_loaded_moconvq_agent(gpu=gpu, base_data=base_data)
    agent.eval()
    with torch.no_grad():
        info = agent.encode_seq_all(None, observation)
    indices = torch.as_tensor(_indices_to_time_depth(info["indexs"], rvq_depth=rvq_depth), dtype=torch.long)
    stats = token_distribution_stats(indices)
    return {
        "kind": "native_h5",
        "path": str(h5_path),
        "observation_key": observation_key,
        "observation_shape": list(observation.shape),
        "shape": list(indices.shape),
        "stats": stats,
    }


def compact_stats(summary: dict[str, object]) -> dict[str, object]:
    compact = dict(summary)
    compact["stats"] = [
        {key: value for key, value in row.items() if key != "counts"}
        for row in summary["stats"]  # type: ignore[index]
    ]
    return compact


def compare_distributions(left: dict[str, object], right: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    left_stats = left["stats"]  # type: ignore[index]
    right_stats = right["stats"]  # type: ignore[index]
    for left_row, right_row in zip(left_stats, right_stats):
        left_counts = torch.as_tensor(left_row["counts"], dtype=torch.float32)
        right_counts = torch.as_tensor(right_row["counts"], dtype=torch.float32)
        rows.append(
            {
                "depth": int(left_row["depth"]),
                "js_divergence_bits": _js_divergence_bits(left_counts, right_counts),
                "left_entropy_bits": float(left_row["entropy_bits"]),
                "right_entropy_bits": float(right_row["entropy_bits"]),
                "left_unique": int(left_row["unique"]),
                "right_unique": int(right_row["unique"]),
            }
        )
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="append", default=[])
    parser.add_argument("--native-h5", default="")
    parser.add_argument("--native-observation-key", default="walk1_subject5/observation")
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rvq-depth", type=int, default=4)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    summaries: list[dict[str, object]] = []
    for cache_path in args.cache:
        summaries.append(summarize_cache(Path(cache_path)))
    if args.native_h5:
        summaries.append(
            summarize_native_h5(
                Path(args.native_h5),
                observation_key=args.native_observation_key,
                base_data=Path(args.base_data),
                gpu=args.gpu,
                rvq_depth=args.rvq_depth,
            )
        )

    comparisons = []
    if len(summaries) >= 2:
        first = summaries[0]
        for other in summaries[1:]:
            comparisons.append(
                {
                    "left": first["path"],
                    "right": other["path"],
                    "by_depth": compare_distributions(first, other),
                }
            )

    payload = {
        "summaries": summaries,
        "comparisons": comparisons,
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"summaries": [compact_stats(x) for x in summaries], "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
