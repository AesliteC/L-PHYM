from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import shlex
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Script.stage1.evaluate_bvh_metrics import evaluate_bvh_files, load_bvh_motion


class PromptRecord(tuple):
    def __new__(
        cls,
        name: str,
        text: str,
        segments: Iterable[str] | None = None,
        segment_lengths: Iterable[int] | None = None,
    ):
        obj = super().__new__(cls, (name, text))
        obj.segments = tuple(segments or ())
        obj.segment_lengths = tuple(segment_lengths or ())
        return obj

    @property
    def name(self) -> str:
        return self[0]

    @property
    def text(self) -> str:
        return self[1]


def _parse_prompt_segments(raw: str, path: Path, line_no: int) -> tuple[str, ...]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid prompt segment JSON at {path}:{line_no}: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"prompt segment field must be a JSON list at {path}:{line_no}")
    segments = tuple(str(item).strip() for item in parsed)
    if not segments or any(not item for item in segments):
        raise ValueError(f"prompt segment field contains empty segments at {path}:{line_no}")
    return segments


def _parse_prompt_segment_lengths(raw: str, path: Path, line_no: int) -> tuple[int, ...]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid prompt segment-length JSON at {path}:{line_no}: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"prompt segment-length field must be a JSON list at {path}:{line_no}")
    lengths = tuple(int(item) for item in parsed)
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError(f"prompt segment-length field contains non-positive lengths at {path}:{line_no}")
    return lengths


def format_prompt_tsv_line(prompt: PromptRecord | tuple[str, str]) -> str:
    name, text = prompt
    segments = getattr(prompt, "segments", ())
    segment_lengths = getattr(prompt, "segment_lengths", ())
    if segments:
        line = f"{name}\t{text}\t{json.dumps(list(segments), ensure_ascii=False)}"
        if segment_lengths:
            line += f"\t{json.dumps(list(segment_lengths), ensure_ascii=False)}"
        return line + "\n"
    return f"{name}\t{text}\n"


def prompt_summary(prompt: PromptRecord | tuple[str, str]) -> dict[str, object]:
    name, text = prompt
    summary: dict[str, object] = {"name": name, "text": text}
    segments = getattr(prompt, "segments", ())
    if segments:
        summary["segments"] = list(segments)
    segment_lengths = getattr(prompt, "segment_lengths", ())
    if segment_lengths:
        summary["segment_lengths"] = list(segment_lengths)
    return summary


def read_prompts(path: Path) -> list[PromptRecord]:
    prompts: list[PromptRecord] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" not in stripped:
            raise ValueError(f"expected TSV prompt at {path}:{line_no}")
        fields = stripped.split("\t")
        if len(fields) not in {2, 3, 4}:
            raise ValueError(f"expected 2, 3, or 4 TSV fields at {path}:{line_no}")
        name, text = fields[0], fields[1]
        name = name.strip()
        text = text.strip()
        if not name or not text:
            raise ValueError(f"empty prompt field at {path}:{line_no}")
        segments = _parse_prompt_segments(fields[2].strip(), path, line_no) if len(fields) == 3 and fields[2].strip() else ()
        if len(fields) == 4:
            segments = _parse_prompt_segments(fields[2].strip(), path, line_no) if fields[2].strip() else ()
            segment_lengths = _parse_prompt_segment_lengths(fields[3].strip(), path, line_no) if fields[3].strip() else ()
            if segment_lengths and not segments:
                raise ValueError(f"segment lengths require explicit segments at {path}:{line_no}")
            if segments and segment_lengths and len(segments) != len(segment_lengths):
                raise ValueError(f"segment count and length count differ at {path}:{line_no}")
        else:
            segment_lengths = ()
        prompts.append(PromptRecord(name, text, segments, segment_lengths))
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def run_command(command: list[str], log_path: Path | None = None) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    if log_path is None:
        subprocess.run(command, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, command)


def count_bvh_frames(path: Path) -> int:
    motion, _ = load_bvh_motion(path)
    return int(motion.shape[0])


def make_side_by_side(left: Path, right: Path, output: Path, ffmpeg: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = "[0:v]scale=960:720,setsar=1[left];[1:v]scale=960:720,setsar=1[right];[left][right]hstack=inputs=2[v]"
    run_command(
        [
            ffmpeg,
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


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--baseline-checkpoint", default="text_generation_GPT.pth")
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--text-model", default="t5-large")
    parser.add_argument("--text-encoder", choices=("t5", "hash"), default="t5")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=75)
    parser.add_argument("--generation-mode", choices=("auto", "rolling", "segmented"), default="auto")
    parser.add_argument("--segment-joiner", default=" then ")
    parser.add_argument("--context-size", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--segment-length", type=int, default=None)
    parser.add_argument("--segment-lengths", default=None)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--progress-conditioning", choices=("none", "scalar", "auto"), default="auto")
    parser.add_argument("--baseline-progress-conditioning", choices=("none", "scalar", "auto"), default="none")
    parser.add_argument("--progress-scale", type=float, default=1.0)
    parser.add_argument("--progress-context-size", type=int, default=None)
    parser.add_argument("--progress-prefix-cap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bvh-dir", default="")
    parser.add_argument("--video-dir", default="")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--expected-min-frames", type=int, default=1200)
    parser.add_argument("--sample-stride", type=int, default=6)
    parser.add_argument("--lags", default="5,10,20,30")
    parser.add_argument("--allow-early-stop", dest="allow_early_stop", action="store_true", default=True)
    parser.add_argument("--no-allow-early-stop", dest="allow_early_stop", action="store_false")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args(argv)

    prompts = read_prompts(Path(args.prompts))
    bvh_dir = Path(args.bvh_dir) if args.bvh_dir else Path("stage1_artifacts/generated_bvh_compare") / args.run_id
    video_dir = Path(args.video_dir) if args.video_dir else Path("stage1_artifacts/generated_video_compare") / args.run_id
    individual_video_dir = video_dir / "individual"
    bvh_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    (bvh_dir / "prompts.tsv").write_text("".join(format_prompt_tsv_line(prompt) for prompt in prompts), encoding="utf-8")

    generated_bvh: list[dict[str, object]] = []
    logs: list[str] = []
    for prompt in prompts:
        prompt_name, prompt_text = prompt
        for model_name, checkpoint in (
            ("baseline_top_p", args.baseline_checkpoint),
            ("finetuned_top_p", args.finetuned_checkpoint),
        ):
            progress_conditioning = (
                args.baseline_progress_conditioning if model_name.startswith("baseline") else args.progress_conditioning
            )
            bvh_path = bvh_dir / f"{prompt_name}__{model_name}.bvh"
            log_path = bvh_dir / f"{prompt_name}__{model_name}.log"
            if not args.skip_generation:
                command = [
                    sys.executable,
                    "Script/stage1/generate_long_motion.py",
                    "--checkpoint",
                    checkpoint,
                    "--text",
                    prompt_text,
                    "--output-bvh",
                    str(bvh_path),
                    "--base-data",
                    args.base_data,
                    "--text-encoder",
                    args.text_encoder,
                    "--text-model",
                    args.text_model,
                    "--max-text-length",
                    str(args.max_text_length),
                    "--max-length",
                    str(args.max_length),
                    "--generation-mode",
                    args.generation_mode,
                    "--segment-joiner",
                    args.segment_joiner,
                    "--context-size",
                    str(args.context_size),
                    "--chunk-size",
                    str(args.chunk_size),
                    "--top-k",
                    str(args.top_k),
                    "--top-p",
                    str(args.top_p),
                    "--temperature",
                    str(args.temperature),
                    "--progress-conditioning",
                    progress_conditioning,
                    "--progress-scale",
                    str(args.progress_scale),
                    "--gpu",
                    str(args.gpu),
                    "--seed",
                    str(args.seed),
                ]
                if args.progress_context_size is not None:
                    command.extend(["--progress-context-size", str(args.progress_context_size)])
                if args.progress_prefix_cap is not None:
                    command.extend(["--progress-prefix-cap", str(args.progress_prefix_cap)])
                if args.allow_early_stop:
                    command.append("--allow-early-stop")
                if args.segment_length is not None:
                    command.extend(["--segment-length", str(args.segment_length)])
                if prompt.segment_lengths:
                    command.extend(["--segment-lengths", ",".join(str(item) for item in prompt.segment_lengths)])
                elif args.segment_lengths:
                    command.extend(["--segment-lengths", args.segment_lengths])
                if prompt.segments:
                    command.extend(["--segments-json", json.dumps(list(prompt.segments), ensure_ascii=False)])
                run_command(command, log_path=log_path)
            generated_bvh.append(
                {
                    "path": str(bvh_path),
                    "prompt": prompt_name,
                    "model": model_name,
                    "frames": count_bvh_frames(bvh_path) if bvh_path.exists() else None,
                    "size": bvh_path.stat().st_size if bvh_path.exists() else None,
                }
            )
            logs.append(str(log_path))

    individual_videos: list[str] = []
    side_by_side_videos: list[str] = []
    if not args.skip_render:
        for item in generated_bvh:
            bvh_path = Path(str(item["path"]))
            run_command(
                [
                    sys.executable,
                    "Script/stage1/render_bvh_to_mp4.py",
                    "--input",
                    str(bvh_path),
                    "--output-dir",
                    str(individual_video_dir),
                    "--ffmpeg",
                    args.ffmpeg,
                    "--fps",
                    str(args.render_fps),
                    "--keep-root-motion",
                ]
            )
            individual_videos.append(str((individual_video_dir / bvh_path.name).with_suffix(".mp4")))

        for prompt in prompts:
            prompt_name = prompt.name
            left = individual_video_dir / f"{prompt_name}__baseline_top_p.mp4"
            right = individual_video_dir / f"{prompt_name}__finetuned_top_p.mp4"
            output = video_dir / f"{prompt_name}__baseline_top_p_vs_finetuned_top_p.mp4"
            make_side_by_side(
                left=left,
                right=right,
                output=output,
                ffmpeg=args.ffmpeg,
            )
            side_by_side_videos.append(str(output))

    lags = tuple(int(item) for item in args.lags.split(",") if item.strip())
    metrics = evaluate_bvh_files(
        [str(bvh_dir / "*.bvh")],
        sample_stride=args.sample_stride,
        lags=lags,
        expected_min_frames=args.expected_min_frames,
    )
    metrics_path = bvh_dir / "summary_metrics_script.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "run_id": args.run_id,
        "sampling": {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "progress_conditioning": args.progress_conditioning,
            "baseline_progress_conditioning": args.baseline_progress_conditioning,
            "progress_scale": args.progress_scale,
            "progress_context_size": args.progress_context_size,
            "progress_prefix_cap": args.progress_prefix_cap,
            "seed": args.seed,
            "max_length": args.max_length,
            "generation_mode": args.generation_mode,
            "segment_joiner": args.segment_joiner,
            "context_size": args.context_size,
            "chunk_size": args.chunk_size,
            "segment_length": args.segment_length,
            "segment_lengths": args.segment_lengths,
            "allow_early_stop": args.allow_early_stop,
        },
        "baseline_checkpoint": args.baseline_checkpoint,
        "finetuned_checkpoint": args.finetuned_checkpoint,
        "prompts": [prompt_summary(prompt) for prompt in prompts],
        "bvh_dir": str(bvh_dir),
        "video_dir": str(video_dir),
        "bvh": generated_bvh,
        "logs": logs,
        "individual_videos": individual_videos,
        "side_by_side_videos": side_by_side_videos,
        "metrics": str(metrics_path),
    }
    (video_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
