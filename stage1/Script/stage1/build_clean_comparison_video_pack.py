from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import re
import shutil
import subprocess
import sys

from Script.stage1.run_text_gpt_comparison import read_prompts
from Script.stage1.summarize_bvh_comparison import summarize_metrics_file


def _clean_prompt_id(prompt: str) -> str:
    match = re.fullmatch(r"(?:train|val|prompt)_0*([0-9]+)", prompt)
    if match:
        return f"prompt_{int(match.group(1)):03d}"
    match = re.fullmatch(r"(?:test_long|long100)_0*([0-9]+)", prompt)
    if match:
        return f"long100_{int(match.group(1)):03d}"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", prompt).strip("_").lower()
    return cleaned or "prompt"


def _prompt_from_label(label: str) -> str:
    if "__" not in label:
        raise ValueError(f"expected metric label with model suffix: {label}")
    return label.rsplit("__", 1)[0]


def _select_prompt_ids(metrics_json: Path, limit: int) -> list[str]:
    summary = summarize_metrics_file(metrics_json)
    paired = summary.get("paired_comparison", {})
    rows = paired.get("prompts", []) if isinstance(paired, dict) else []
    scored: list[tuple[float, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row["prompt"])
        delta_frames = float(row.get("delta_frames") or 0.0)
        delta_root = float(row.get("delta_root_path_length") or 0.0)
        delta_velocity = float(row.get("delta_pose_velocity_mean") or 0.0)
        base_stop = float(row.get("baseline_early_stop") or 0.0)
        tuned_stop = float(row.get("finetuned_early_stop") or 0.0)
        stop_penalty = 500.0 if tuned_stop > base_stop else 0.0
        score = delta_frames / 120.0 + delta_root - 0.02 * max(delta_velocity, 0.0) - stop_penalty
        scored.append((score, prompt))
    scored.sort(reverse=True)
    return [prompt for _score, prompt in scored[:limit]]


def _load_prompt_texts(prompts_tsv: Path) -> dict[str, dict[str, object]]:
    prompts = {}
    for record in read_prompts(prompts_tsv):
        prompts[record.name] = {
            "text": record.text,
            "segments": list(record.segments),
            "segment_lengths": list(record.segment_lengths),
        }
    return prompts


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _resolve_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    local = Path(sys.executable).resolve().parent / name
    if local.exists():
        return str(local)
    return name


def _make_side_by_side(left: Path, right: Path, output: Path, ffmpeg: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = "[0:v]scale=960:720,setsar=1[left];[1:v]scale=960:720,setsar=1[right];[left][right]hstack=inputs=2[v]"
    _run(
        [
            _resolve_executable(ffmpeg),
            "-y",
            "-v",
            "error",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def _probe_video(path: Path, ffprobe: str) -> dict[str, object]:
    proc = subprocess.run(
        [
            _resolve_executable(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": float(stream.get("duration") or 0.0),
        "frames": int(stream.get("nb_frames") or 0),
    }


def _clean_metric_payload(metric_payload: dict[str, object]) -> dict[str, object]:
    cleaned = dict(metric_payload)
    if cleaned.get("prompt") is not None:
        cleaned["prompt"] = _clean_prompt_id(str(cleaned["prompt"]))
    return cleaned


def build_pack(
    *,
    source_dir: Path,
    output_dir: Path,
    suite_name: str,
    prompts: Iterable[str],
    prompts_tsv: Path,
    metrics_json: Path,
    ffmpeg: str,
    ffprobe: str,
    render_fps: int,
    max_video_frames: int | None,
) -> list[dict[str, object]]:
    source_bvh_dir = source_dir / "bvh"
    clean_bvh_dir = output_dir / "bvh" / suite_name
    individual_dir = output_dir / "individual" / suite_name
    comparison_dir = output_dir / "comparison_videos" / suite_name
    clean_bvh_dir.mkdir(parents=True, exist_ok=True)
    individual_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    prompt_texts = _load_prompt_texts(prompts_tsv)
    metrics_summary = summarize_metrics_file(metrics_json)
    metric_rows = metrics_summary.get("paired_comparison", {}).get("prompts", [])  # type: ignore[union-attr]
    metrics_by_prompt = {
        str(row["prompt"]): row
        for row in metric_rows
        if isinstance(row, dict) and row.get("prompt") is not None
    }

    rows: list[dict[str, object]] = []
    for prompt in prompts:
        clean_id = _clean_prompt_id(prompt)
        baseline_src = source_bvh_dir / f"{prompt}__baseline_top_p.bvh"
        finetuned_src = source_bvh_dir / f"{prompt}__finetuned_top_p.bvh"
        if not baseline_src.exists() or not finetuned_src.exists():
            raise FileNotFoundError(f"missing baseline/finetuned BVH for {prompt} under {source_bvh_dir}")
        baseline_bvh = clean_bvh_dir / f"{clean_id}__baseline_top_p.bvh"
        finetuned_bvh = clean_bvh_dir / f"{clean_id}__finetuned_top_p.bvh"
        shutil.copy2(baseline_src, baseline_bvh)
        shutil.copy2(finetuned_src, finetuned_bvh)

        render_args = [
            sys.executable,
            "Script/stage1/render_bvh_to_mp4.py",
            "--output-dir",
            str(individual_dir),
            "--ffmpeg",
            ffmpeg,
            "--fps",
            str(render_fps),
            "--keep-root-motion",
        ]
        if max_video_frames is not None:
            render_args.extend(["--max-video-frames", str(max_video_frames)])
        _run([*render_args, "--input", str(baseline_bvh)])
        _run([*render_args, "--input", str(finetuned_bvh)])

        baseline_mp4 = individual_dir / f"{clean_id}__baseline_top_p.mp4"
        finetuned_mp4 = individual_dir / f"{clean_id}__finetuned_top_p.mp4"
        comparison_mp4 = comparison_dir / f"comparison_{suite_name}_{clean_id}_baseline_vs_finetuned.mp4"
        _make_side_by_side(baseline_mp4, finetuned_mp4, comparison_mp4, ffmpeg=ffmpeg)
        probe = _probe_video(comparison_mp4, ffprobe=ffprobe)

        prompt_payload = prompt_texts.get(prompt, {})
        metric_payload = metrics_by_prompt.get(prompt, {})
        rows.append(
            {
                "suite": suite_name,
                "source_prompt_id": clean_id,
                "clean_prompt_id": clean_id,
                "video": str(comparison_mp4.relative_to(output_dir)),
                "baseline_video": str(baseline_mp4.relative_to(output_dir)),
                "finetuned_video": str(finetuned_mp4.relative_to(output_dir)),
                "prompt": prompt_payload.get("text", ""),
                "segments": prompt_payload.get("segments", []),
                "segment_lengths": prompt_payload.get("segment_lengths", []),
                "metrics": _clean_metric_payload(metric_payload),
                "probe": probe,
            }
        )
    return rows


def _write_manifest(output_dir: Path, rows: list[dict[str, object]], selected: dict[str, list[str]]) -> None:
    lines = [
        "# Stage1 Clean-Label Comparison Videos",
        "",
        "All videos in this local-only review pack hide the internal HumanML3D",
        "split prefix in the file name and upper-left rendered label.  The left panel is the",
        "original MoConVQ text GPT baseline; the right panel is the Stage1",
        "fine-tuned model.",
        "",
        "## Selected Prompt IDs",
        "",
    ]
    for suite, prompts in selected.items():
        clean_prompts = [_clean_prompt_id(prompt) for prompt in prompts]
        lines.append(f"- `{suite}`: {', '.join(clean_prompts)}")
    lines.extend(["", "## Videos", ""])
    lines.append("| video | prompt | frames baseline -> fine-tuned | root path baseline -> fine-tuned | early stop baseline -> fine-tuned |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for row in rows:
        metrics = row.get("metrics", {})
        assert isinstance(metrics, dict)
        lines.append(
            "| `{video}` | {prompt} | {base_frames} -> {tuned_frames} | {base_root:.3f} -> {tuned_root:.3f} | {base_stop} -> {tuned_stop} |".format(
                video=row["video"],
                prompt=str(row.get("prompt", "")).replace("|", "\\|"),
                base_frames=int(metrics.get("baseline_frames") or 0),
                tuned_frames=int(metrics.get("finetuned_frames") or 0),
                base_root=float(metrics.get("baseline_root_path_length") or 0.0),
                tuned_root=float(metrics.get("finetuned_root_path_length") or 0.0),
                base_stop=bool(metrics.get("baseline_early_stop")),
                tuned_stop=bool(metrics.get("finetuned_early_stop")),
            )
        )
    lines.extend(["", "## Probe", ""])
    lines.append("| video | resolution | duration | frames |")
    lines.append("| --- | --- | ---: | ---: |")
    for row in rows:
        probe = row.get("probe", {})
        assert isinstance(probe, dict)
        lines.append(
            f"| `{row['video']}` | {probe.get('width')}x{probe.get('height')} | {float(probe.get('duration') or 0.0):.2f}s | {probe.get('frames')} |"
        )
    lines.append("")
    (output_dir / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    clean_selected = {
        suite: [_clean_prompt_id(prompt) for prompt in prompts]
        for suite, prompts in selected.items()
    }
    (output_dir / "manifest.json").write_text(
        json.dumps({"rows": rows, "selected": clean_selected}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--max-video-frames", type=int, default=None)
    parser.add_argument(
        "--suite",
        action="append",
        required=True,
        help="JSON object with name, source_dir, prompts_tsv, metrics_json, and either prompts or auto_limit.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    selected: dict[str, list[str]] = {}
    for raw_suite in args.suite:
        suite = json.loads(raw_suite)
        name = str(suite["name"])
        metrics_json = Path(str(suite["metrics_json"]))
        if suite.get("prompts"):
            prompts = [str(item) for item in suite["prompts"]]
        else:
            prompts = _select_prompt_ids(metrics_json, limit=int(suite.get("auto_limit", 8)))
        selected[name] = prompts
        rows.extend(
            build_pack(
                source_dir=Path(str(suite["source_dir"])),
                output_dir=output_dir,
                suite_name=name,
                prompts=prompts,
                prompts_tsv=Path(str(suite["prompts_tsv"])),
                metrics_json=metrics_json,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                render_fps=args.render_fps,
                max_video_frames=args.max_video_frames,
            )
        )
    _write_manifest(output_dir, rows, selected)
    print(json.dumps({"output_dir": str(output_dir), "videos": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
