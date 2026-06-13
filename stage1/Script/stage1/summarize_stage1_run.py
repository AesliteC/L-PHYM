from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json

from Script.stage1.summarize_bvh_comparison import format_value, summarize_metrics_file


def _load_json(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    rows: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def _token_top_fractions(payload: dict[str, object] | None) -> dict[str, float]:
    if payload is None:
        return {}
    summaries = payload.get("summaries")
    if not isinstance(summaries, list) or not summaries:
        return {}
    first = summaries[0] if isinstance(summaries[0], dict) else {}
    rows = first.get("depths") or first.get("stats") if isinstance(first, dict) else None
    if not isinstance(rows, list):
        return {}
    result: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "top_frac" in row:
            top_frac = row["top_frac"]
        else:
            top_fracs = row.get("top_fracs") or []
            top_frac = top_fracs[0] if top_fracs else None
        if top_frac is not None:
            result[f"depth{int(row['depth'])}"] = float(top_frac)
    return result


def _cache_summary(payload: dict[str, object] | None) -> dict[str, object]:
    if payload is None:
        return {}
    keys = ("windows", "valid_tokens", "unique_sequences", "indices_shape", "text_features_shape")
    return {key: payload[key] for key in keys if key in payload}


def _training_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    first = rows[0]
    last = rows[-1]
    return {
        "epochs": len(rows),
        "first_epoch": first.get("epoch"),
        "last_epoch": last.get("epoch"),
        "first_train_loss": first.get("train", {}).get("loss") if isinstance(first.get("train"), dict) else None,
        "last_train_loss": last.get("train", {}).get("loss") if isinstance(last.get("train"), dict) else None,
        "first_val_loss": first.get("val", {}).get("loss") if isinstance(first.get("val"), dict) else None,
        "last_val_loss": last.get("val", {}).get("loss") if isinstance(last.get("val"), dict) else None,
        "first_val_acc": first.get("val", {}).get("token_accuracy") if isinstance(first.get("val"), dict) else None,
        "last_val_acc": last.get("val", {}).get("token_accuracy") if isinstance(last.get("val"), dict) else None,
    }


def _quality_summary(payload: dict[str, object] | None) -> dict[str, object]:
    if payload is None:
        return {}
    counts = payload.get("counts", {})
    rows = payload.get("rows", [])
    total = int(counts.get("total", 0)) if isinstance(counts, dict) else 0
    accepted = int(counts.get("accepted", 0)) if isinstance(counts, dict) else 0
    reasons: dict[str, int] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            for reason in row.get("reject_reasons", []) or []:
                reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return {
        "counts": counts,
        "accepted_rate": accepted / total if total else None,
        "reject_reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
    }


def _model_average(summary: dict[str, object], model: str, metric: str) -> float | None:
    model_summary = summary.get("model_summary", {})
    if not isinstance(model_summary, dict):
        return None
    row = model_summary.get(model)
    if not isinstance(row, dict):
        return None
    value = row.get(f"avg_{metric}")
    return float(value) if value is not None else None


def _generation_improvements(comparison: dict[str, object]) -> dict[str, object]:
    metrics = ("frames", "early_stop", "root_path_length", "pose_velocity_mean", "pose_variance_mean")
    out: dict[str, object] = {}
    for metric in metrics:
        baseline = _model_average(comparison, "baseline_top_p", metric)
        tuned = _model_average(comparison, "finetuned_top_p", metric)
        out[metric] = {
            "baseline": baseline,
            "finetuned": tuned,
            "delta": None if baseline is None or tuned is None else tuned - baseline,
        }
    return out


def build_stage1_run_report(
    *,
    run_name: str,
    quality_summary: Path | None,
    train_cache_summary: Path | None,
    val_cache_summary: Path | None,
    train_token_distribution: Path | None,
    val_token_distribution: Path | None,
    train_log: Path | None,
    metrics_json: Path,
    comparison_video_summary: Path | None,
    evaluation_readiness: Path | None,
    checkpoint: str,
    notes: str,
) -> dict[str, object]:
    quality = _quality_summary(_load_json(quality_summary))
    train_rows = _load_jsonl(train_log)
    comparison = summarize_metrics_file(metrics_json)
    readiness = _load_json(evaluation_readiness) or {}
    video_summary = _load_json(comparison_video_summary) or {}
    report = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "notes": notes,
        "quality": quality,
        "cache": {
            "train": _cache_summary(_load_json(train_cache_summary)),
            "val": _cache_summary(_load_json(val_cache_summary)),
        },
        "token_top_fractions": {
            "train": _token_top_fractions(_load_json(train_token_distribution)),
            "val": _token_top_fractions(_load_json(val_token_distribution)),
        },
        "training": {
            "summary": _training_summary(train_rows),
            "log": train_rows,
        },
        "generation": {
            "comparison": comparison,
            "improvements": _generation_improvements(comparison),
            "video_dir": video_summary.get("video_dir"),
            "side_by_side_videos": video_summary.get("side_by_side_videos", []),
            "individual_videos": video_summary.get("individual_videos", []),
        },
        "paper_metrics": {
            "ready": bool(readiness.get("paper_metrics_ready", False)),
            "missing": readiness.get("paper_metrics_missing", []),
            "paper_metrics": readiness.get("paper_metrics", ["FID", "R-precision"]),
        },
    }
    return report


def report_to_markdown(report: dict[str, object]) -> str:
    quality = report.get("quality", {})
    cache = report.get("cache", {})
    tokens = report.get("token_top_fractions", {})
    training = report.get("training", {})
    generation = report.get("generation", {})
    paper = report.get("paper_metrics", {})

    lines = [f"# Stage1 Run Report: {report['run_name']}", ""]
    if report.get("notes"):
        lines.extend([str(report["notes"]), ""])
    lines.extend(["## Checkpoint", "", f"`{report.get('checkpoint', '')}`", ""])

    lines.extend(["## Data Quality", ""])
    q_counts = quality.get("counts", {}) if isinstance(quality, dict) else {}
    lines.append(f"- total: {q_counts.get('total', '-') if isinstance(q_counts, dict) else '-'}")
    lines.append(f"- accepted: {q_counts.get('accepted', '-') if isinstance(q_counts, dict) else '-'}")
    lines.append(f"- rejected: {q_counts.get('rejected', '-') if isinstance(q_counts, dict) else '-'}")
    lines.append(f"- accepted rate: {format_value(quality.get('accepted_rate') if isinstance(quality, dict) else None)}")
    reasons = quality.get("reject_reasons", {}) if isinstance(quality, dict) else {}
    if isinstance(reasons, dict) and reasons:
        lines.extend(["", "| reject reason | count |", "| --- | ---: |"])
        for reason, count in reasons.items():
            lines.append(f"| {reason} | {count} |")
    lines.append("")

    lines.extend(["## Cache", "", "| split | windows | valid tokens | unique sequences | token top fractions |"])
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for split in ("train", "val"):
        split_cache = cache.get(split, {}) if isinstance(cache, dict) else {}
        split_tokens = tokens.get(split, {}) if isinstance(tokens, dict) else {}
        token_text = ", ".join(f"{key}={value:.3f}" for key, value in split_tokens.items()) if isinstance(split_tokens, dict) else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    format_value(split_cache.get("windows") if isinstance(split_cache, dict) else None),
                    format_value(split_cache.get("valid_tokens") if isinstance(split_cache, dict) else None),
                    format_value(split_cache.get("unique_sequences") if isinstance(split_cache, dict) else None),
                    token_text or "-",
                ]
            )
            + " |"
        )
    lines.append("")

    lines.extend(["## Training", ""])
    train_summary = training.get("summary", {}) if isinstance(training, dict) else {}
    if isinstance(train_summary, dict) and train_summary:
        lines.append(
            f"- val loss: {format_value(train_summary.get('first_val_loss'))} -> "
            f"{format_value(train_summary.get('last_val_loss'))}"
        )
        lines.append(
            f"- val accuracy: {format_value(train_summary.get('first_val_acc'))} -> "
            f"{format_value(train_summary.get('last_val_acc'))}"
        )
    else:
        lines.append("- no training log provided")
    lines.append("")

    lines.extend(["## Generation", ""])
    improvements = generation.get("improvements", {}) if isinstance(generation, dict) else {}
    lines.append("| metric | baseline | finetuned | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    if isinstance(improvements, dict):
        for metric, row in improvements.items():
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {metric} | {format_value(row.get('baseline'))} | "
                f"{format_value(row.get('finetuned'))} | {format_value(row.get('delta'))} |"
            )
    lines.append("")
    videos = generation.get("side_by_side_videos", []) if isinstance(generation, dict) else []
    if videos:
        lines.extend(["Side-by-side videos:", ""])
        for path in videos:
            lines.append(f"- `{path}`")
        lines.append("")

    lines.extend(["## Paper Metrics Gate", ""])
    lines.append(f"- ready: {str(paper.get('ready', False)).lower() if isinstance(paper, dict) else 'false'}")
    missing = paper.get("missing", []) if isinstance(paper, dict) else []
    if missing:
        for item in missing:
            lines.append(f"- missing: {item}")
    lines.append("")
    lines.append(
        "Paper-level FID/R-precision should not be claimed unless this gate is ready."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--quality-summary", default="")
    parser.add_argument("--train-cache-summary", default="")
    parser.add_argument("--val-cache-summary", default="")
    parser.add_argument("--train-token-distribution", default="")
    parser.add_argument("--val-token-distribution", default="")
    parser.add_argument("--train-log", default="")
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--comparison-video-summary", default="")
    parser.add_argument("--evaluation-readiness", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args(argv)

    report = build_stage1_run_report(
        run_name=args.run_name,
        quality_summary=Path(args.quality_summary) if args.quality_summary else None,
        train_cache_summary=Path(args.train_cache_summary) if args.train_cache_summary else None,
        val_cache_summary=Path(args.val_cache_summary) if args.val_cache_summary else None,
        train_token_distribution=Path(args.train_token_distribution) if args.train_token_distribution else None,
        val_token_distribution=Path(args.val_token_distribution) if args.val_token_distribution else None,
        train_log=Path(args.train_log) if args.train_log else None,
        metrics_json=Path(args.metrics_json),
        comparison_video_summary=Path(args.comparison_video_summary) if args.comparison_video_summary else None,
        evaluation_readiness=Path(args.evaluation_readiness) if args.evaluation_readiness else None,
        checkpoint=args.checkpoint,
        notes=args.notes,
    )
    text_json = json.dumps(report, indent=2, ensure_ascii=False)
    text_md = report_to_markdown(report)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text_json, encoding="utf-8")
    if args.output_md:
        output = Path(args.output_md)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text_md, encoding="utf-8")
    print(text_md)


if __name__ == "__main__":
    main()
