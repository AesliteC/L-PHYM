from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import math


DEFAULT_METRICS = [
    "frames",
    "duration_sec",
    "early_stop",
    "root_path_length",
    "root_displacement",
    "pose_velocity_mean",
    "pose_variance_mean",
    "lag_20_repeat_fraction_0.995",
]


def split_prompt_and_model(label: str) -> tuple[str, str]:
    if "__" not in label:
        raise ValueError(f"metric label does not contain model separator '__': {label}")
    prompt, model = label.rsplit("__", 1)
    if not prompt or not model:
        raise ValueError(f"invalid metric label: {label}")
    return prompt, model


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def summarize_rows(rows: list[dict[str, object]], metrics: list[str] | None = None) -> dict[str, object]:
    metrics = metrics or DEFAULT_METRICS
    by_prompt: dict[str, dict[str, dict[str, object]]] = {}
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        prompt, model = split_prompt_and_model(str(row["label"]))
        by_prompt.setdefault(prompt, {})[model] = row
        by_model.setdefault(model, []).append(row)

    model_summary: dict[str, dict[str, object]] = {}
    for model, model_rows in sorted(by_model.items()):
        summary: dict[str, object] = {"count": len(model_rows)}
        for metric in metrics:
            values = [_as_float(row.get(metric)) for row in model_rows]
            valid = [value for value in values if value is not None]
            summary[f"avg_{metric}"] = _mean(valid)
        model_summary[model] = summary

    paired: dict[str, object] = {}
    if "baseline_top_p" in by_model and "finetuned_top_p" in by_model:
        prompt_rows = []
        for prompt, models in sorted(by_prompt.items()):
            if "baseline_top_p" not in models or "finetuned_top_p" not in models:
                continue
            baseline = models["baseline_top_p"]
            finetuned = models["finetuned_top_p"]
            row: dict[str, object] = {"prompt": prompt}
            for metric in metrics:
                base_value = _as_float(baseline.get(metric))
                tuned_value = _as_float(finetuned.get(metric))
                row[f"baseline_{metric}"] = base_value
                row[f"finetuned_{metric}"] = tuned_value
                row[f"delta_{metric}"] = (
                    None if base_value is None or tuned_value is None else tuned_value - base_value
                )
            prompt_rows.append(row)
        paired = {
            "models": ["baseline_top_p", "finetuned_top_p"],
            "prompts": prompt_rows,
        }

    return {
        "metrics": metrics,
        "model_summary": model_summary,
        "paired_comparison": paired,
    }


def format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0:
            return f"{value:.4g}"
        return f"{value:.3f}"
    return str(value)


def summary_to_markdown(summary: dict[str, object]) -> str:
    model_summary = summary["model_summary"]
    assert isinstance(model_summary, dict)
    metrics = [str(metric) for metric in summary["metrics"]]

    lines = ["# BVH Comparison Summary", ""]
    lines.append("## Model Averages")
    lines.append("")
    header = ["model", "count", *[f"avg_{metric}" for metric in metrics]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |")
    for model, row in sorted(model_summary.items()):
        assert isinstance(row, dict)
        values = [model, format_value(row.get("count"))]
        values.extend(format_value(row.get(f"avg_{metric}")) for metric in metrics)
        lines.append("| " + " | ".join(values) + " |")

    paired = summary.get("paired_comparison")
    if isinstance(paired, dict) and paired.get("prompts"):
        lines.extend(["", "## Per-Prompt Delta", ""])
        delta_metrics = [
            "frames",
            "duration_sec",
            "pose_velocity_mean",
            "pose_variance_mean",
            "lag_20_repeat_fraction_0.995",
        ]
        header = ["prompt"]
        for metric in delta_metrics:
            header.extend([f"baseline_{metric}", f"finetuned_{metric}", f"delta_{metric}"])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |")
        for row in paired["prompts"]:
            assert isinstance(row, dict)
            values = [str(row["prompt"])]
            for metric in delta_metrics:
                values.extend(
                    [
                        format_value(row.get(f"baseline_{metric}")),
                        format_value(row.get(f"finetuned_{metric}")),
                        format_value(row.get(f"delta_{metric}")),
                    ]
                )
            lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    return "\n".join(lines)


def summarize_metrics_file(path: Path, metrics: list[str] | None = None) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"metrics file has no rows list: {path}")
    summary = summarize_rows(rows, metrics=metrics)
    summary["source"] = str(path)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    args = parser.parse_args(argv)

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    summary = summarize_metrics_file(Path(args.metrics_json), metrics=metrics)
    text_json = json.dumps(summary, indent=2, ensure_ascii=False)
    text_md = summary_to_markdown(summary)
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
