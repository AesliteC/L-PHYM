from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import csv
import json


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _label_from_path(path: str) -> str:
    return Path(path).stem


def _top_fraction(stats: list[dict[str, object]], depth: int) -> float:
    for row in stats:
        if int(row["depth"]) == depth:
            top_fracs = row.get("top_fracs") or []
            return float(top_fracs[0]) if top_fracs else 0.0
    return 0.0


def _unique_count(stats: list[dict[str, object]], depth: int) -> int:
    for row in stats:
        if int(row["depth"]) == depth:
            return int(row["unique"])
    return 0


def _entropy_bits(stats: list[dict[str, object]], depth: int) -> float:
    for row in stats:
        if int(row["depth"]) == depth:
            return float(row["entropy_bits"])
    return 0.0


def _metrics_by_label(metrics_payload: dict[str, object] | None) -> dict[str, dict[str, object]]:
    if metrics_payload is None:
        return {}
    return {str(row["label"]): row for row in metrics_payload.get("rows", [])}  # type: ignore[union-attr]


def _captions_by_id(export_payload: dict[str, object] | None) -> dict[str, str]:
    if export_payload is None:
        return {}
    captions: dict[str, str] = {}
    for row in export_payload.get("exports", []):  # type: ignore[union-attr]
        sample_id = str(row.get("sample_id", ""))
        if sample_id:
            captions[sample_id] = str(row.get("caption", ""))
    return captions


def _reject_reasons(
    *,
    frames: int,
    tokens: int,
    p99_abs_z: float,
    max_abs_z: float,
    frac_gt_5: float,
    depth0_top_frac: float,
    depth0_unique: int,
    min_frames: int,
    min_tokens: int,
    max_p99_abs_z: float,
    max_max_abs_z: float,
    max_frac_gt_5: float,
    max_depth0_top_frac: float,
    min_depth0_unique: int,
) -> list[str]:
    reasons: list[str] = []
    if frames < min_frames:
        reasons.append(f"frames<{min_frames}")
    if tokens < min_tokens:
        reasons.append(f"tokens<{min_tokens}")
    if p99_abs_z > max_p99_abs_z:
        reasons.append(f"p99_abs_z>{max_p99_abs_z:g}")
    if max_abs_z > max_max_abs_z:
        reasons.append(f"max_abs_z>{max_max_abs_z:g}")
    if frac_gt_5 > max_frac_gt_5:
        reasons.append(f"frac_gt_5>{max_frac_gt_5:g}")
    if depth0_top_frac > max_depth0_top_frac:
        reasons.append(f"depth0_top_frac>{max_depth0_top_frac:g}")
    if depth0_unique < min_depth0_unique:
        reasons.append(f"depth0_unique<{min_depth0_unique}")
    return reasons


def summarize_retarget_quality(
    retarget_payload: dict[str, object],
    metrics_payload: dict[str, object] | None = None,
    export_payload: dict[str, object] | None = None,
    *,
    min_frames: int = 120,
    min_tokens: int = 20,
    max_p99_abs_z: float = 8.0,
    max_max_abs_z: float = 50.0,
    max_frac_gt_5: float = 0.05,
    max_depth0_top_frac: float = 0.25,
    min_depth0_unique: int = 16,
) -> dict[str, object]:
    metrics = _metrics_by_label(metrics_payload)
    captions = _captions_by_id(export_payload)
    rows: list[dict[str, object]] = []
    for item in retarget_payload.get("per_file", []):  # type: ignore[union-attr]
        path = str(item["path"])
        label = _label_from_path(path)
        z = item["observation_z"]["aggregate_abs_z"]  # type: ignore[index]
        stats = item["stats"]  # type: ignore[assignment]
        frames = int(item["state_shape"][0])  # type: ignore[index]
        tokens = int(item["shape"][0])  # type: ignore[index]
        depth0_top_frac = _top_fraction(stats, depth=0)  # type: ignore[arg-type]
        depth0_unique = _unique_count(stats, depth=0)  # type: ignore[arg-type]
        depth0_entropy = _entropy_bits(stats, depth=0)  # type: ignore[arg-type]
        reasons = _reject_reasons(
            frames=frames,
            tokens=tokens,
            p99_abs_z=float(z["p99"]),
            max_abs_z=float(z["max"]),
            frac_gt_5=float(z["frac_gt_5"]),
            depth0_top_frac=depth0_top_frac,
            depth0_unique=depth0_unique,
            min_frames=min_frames,
            min_tokens=min_tokens,
            max_p99_abs_z=max_p99_abs_z,
            max_max_abs_z=max_max_abs_z,
            max_frac_gt_5=max_frac_gt_5,
            max_depth0_top_frac=max_depth0_top_frac,
            min_depth0_unique=min_depth0_unique,
        )
        metric_row = metrics.get(label, {})
        rows.append(
            {
                "label": label,
                "path": path,
                "caption": captions.get(label, ""),
                "accepted": not reasons,
                "reject_reasons": reasons,
                "frames": frames,
                "tokens": tokens,
                "duration_sec": metric_row.get("duration_sec"),
                "early_stop": metric_row.get("early_stop", frames < min_frames),
                "p95_abs_z": float(z["p95"]),
                "p99_abs_z": float(z["p99"]),
                "max_abs_z": float(z["max"]),
                "frac_gt_5": float(z["frac_gt_5"]),
                "frac_gt_10": float(z["frac_gt_10"]),
                "depth0_top_frac": depth0_top_frac,
                "depth0_unique": depth0_unique,
                "depth0_entropy_bits": depth0_entropy,
                "pose_velocity_mean": metric_row.get("pose_velocity_mean"),
                "pose_variance_mean": metric_row.get("pose_variance_mean"),
            }
        )

    rows.sort(key=lambda row: (bool(row["accepted"]), -float(row["p99_abs_z"]), -float(row["depth0_top_frac"])))
    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]
    return {
        "metric_notes": {
            "scope": "Stage1 engineering quality filter for BVH-to-character cache candidates; not a paper-level metric.",
            "thresholds_are_preliminary": True,
            "acceptance": "A sample is accepted only if all configured engineering thresholds pass.",
        },
        "thresholds": {
            "min_frames": min_frames,
            "min_tokens": min_tokens,
            "max_p99_abs_z": max_p99_abs_z,
            "max_max_abs_z": max_max_abs_z,
            "max_frac_gt_5": max_frac_gt_5,
            "max_depth0_top_frac": max_depth0_top_frac,
            "min_depth0_unique": min_depth0_unique,
        },
        "counts": {
            "total": len(rows),
            "accepted": len(accepted),
            "rejected": len(rejected),
        },
        "accepted_paths": [str(row["path"]) for row in accepted],
        "rejected_labels": [str(row["label"]) for row in rejected],
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "accepted",
        "reject_reasons",
        "frames",
        "tokens",
        "p99_abs_z",
        "max_abs_z",
        "frac_gt_5",
        "depth0_top_frac",
        "depth0_unique",
        "depth0_entropy_bits",
        "pose_velocity_mean",
        "caption",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["reject_reasons"] = ";".join(str(reason) for reason in row.get("reject_reasons", []))
            writer.writerow(out)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retarget-json", required=True)
    parser.add_argument("--bvh-metrics-json", default="")
    parser.add_argument("--export-summary", default="")
    parser.add_argument("--min-frames", type=int, default=120)
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--max-p99-abs-z", type=float, default=8.0)
    parser.add_argument("--max-max-abs-z", type=float, default=50.0)
    parser.add_argument("--max-frac-gt-5", type=float, default=0.05)
    parser.add_argument("--max-depth0-top-frac", type=float, default=0.25)
    parser.add_argument("--min-depth0-unique", type=int, default=16)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--quiet", action="store_true", help="Print only counts and output paths to stdout.")
    args = parser.parse_args(argv)

    payload = summarize_retarget_quality(
        _load_json(Path(args.retarget_json)),
        metrics_payload=_load_json(Path(args.bvh_metrics_json)) if args.bvh_metrics_json else None,
        export_payload=_load_json(Path(args.export_summary)) if args.export_summary else None,
        min_frames=args.min_frames,
        min_tokens=args.min_tokens,
        max_p99_abs_z=args.max_p99_abs_z,
        max_max_abs_z=args.max_max_abs_z,
        max_frac_gt_5=args.max_frac_gt_5,
        max_depth0_top_frac=args.max_depth0_top_frac,
        min_depth0_unique=args.min_depth0_unique,
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if args.output_csv:
        _write_csv(Path(args.output_csv), payload["rows"])  # type: ignore[arg-type]
    if args.quiet:
        print(
            json.dumps(
                {
                    "output_json": args.output_json,
                    "output_csv": args.output_csv,
                    "counts": payload["counts"],
                },
                indent=2,
            )
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
