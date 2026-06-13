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
from Script.stage1.run_text_gpt_comparison import format_prompt_tsv_line, prompt_summary, read_prompts, run_command


DEFAULT_PROMPTS = (
    ("walk_turn_wave", "a person walks forward then turns around then waves both arms"),
    ("circle_crouch_stand", "a person walks in a circle then crouches down then stands up"),
    ("walk_jump_dance", "a person walks forward then jumps then dances"),
    ("sidestep_kick_turn", "a person sidesteps to the left then kicks with the right foot then turns around"),
)


def write_default_prompts(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\t{text}\n" for name, text in DEFAULT_PROMPTS), encoding="utf-8")


def count_bvh_frames(path: Path) -> int:
    motion, _frame_time = load_bvh_motion(path)
    return int(motion.shape[0])


def _command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def generate_gpt_bvh(
    *,
    prompt_name: str,
    prompt_text: str,
    prompt_segments: Iterable[str] = (),
    prompt_segment_lengths: Iterable[int] = (),
    model_name: str,
    checkpoint: str,
    output_bvh: Path,
    log_path: Path,
    args: argparse.Namespace,
    progress_conditioning: str,
) -> dict[str, object]:
    motion_dataset = resolve_motion_dataset(args)
    command = [
        sys.executable,
        "Script/stage1/generate_long_motion.py",
        "--checkpoint",
        checkpoint,
        "--text",
        prompt_text,
        "--output-bvh",
        str(output_bvh),
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
    if motion_dataset is not None:
        command.extend(["--motion-dataset", str(motion_dataset)])
    if args.progress_context_size is not None:
        command.extend(["--progress-context-size", str(args.progress_context_size)])
    if args.progress_prefix_cap is not None:
        command.extend(["--progress-prefix-cap", str(args.progress_prefix_cap)])
    if args.allow_early_stop:
        command.append("--allow-early-stop")
    if args.segment_length is not None:
        command.extend(["--segment-length", str(args.segment_length)])
    prompt_segment_lengths = tuple(prompt_segment_lengths)
    if prompt_segment_lengths:
        command.extend(["--segment-lengths", ",".join(str(item) for item in prompt_segment_lengths)])
    elif args.segment_lengths:
        command.extend(["--segment-lengths", args.segment_lengths])
    prompt_segments = tuple(prompt_segments)
    if prompt_segments:
        command.extend(["--segments-json", json.dumps(list(prompt_segments), ensure_ascii=False)])
    reused_existing = bool(args.reuse_existing_bvh and output_bvh.exists())
    if not args.skip_generation and not reused_existing:
        run_command(command, log_path=log_path)
    return {
        "prompt": prompt_name,
        "model": model_name,
        "path": str(output_bvh),
        "log": str(log_path),
        "command": _command_to_string(command),
        "reused_existing": reused_existing,
        "frames": count_bvh_frames(output_bvh) if output_bvh.exists() else None,
    }


def ensure_example_bank(args: argparse.Namespace, suite_dir: Path) -> Path | None:
    if not args.backup_cache and not args.example_bank:
        return None
    if args.example_bank:
        return Path(args.example_bank)
    bank = suite_dir / "llm_backup" / "example_bank.jsonl"
    command = [
        sys.executable,
        "Script/stage1/llm_token_planning.py",
        "export-bank",
        "--cache",
        args.backup_cache,
        "--output",
        str(bank),
        "--max-examples",
        str(args.backup_max_examples),
        "--max-tokens-per-example",
        str(args.backup_max_tokens_per_example),
        "--min-tokens-per-example",
        str(args.backup_min_tokens_per_example),
    ]
    run_command(command, log_path=suite_dir / "llm_backup" / "export_bank.log")
    return bank


def resolve_motion_dataset(args: argparse.Namespace) -> Path | None:
    if args.motion_dataset:
        return Path(args.motion_dataset)
    base_data = Path(args.base_data)
    candidate = base_data.parent / "simple_motion_data.h5"
    return candidate if candidate.exists() else None


def run_retrieval_backup(
    *,
    prompt_name: str,
    prompt_text: str,
    bank: Path,
    bvh_dir: Path,
    suite_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    backup_dir = suite_dir / "llm_backup" / prompt_name
    backup_dir.mkdir(parents=True, exist_ok=True)
    tokens = backup_dir / "retrieval_tokens.json"
    validation = backup_dir / "retrieval_validation.json"
    prompt_file = backup_dir / "prompt.txt"
    retrieval_json = backup_dir / "retrieval.json"
    build_prompt_cmd = [
        sys.executable,
        "Script/stage1/llm_token_planning.py",
        "build-prompt",
        "--bank",
        str(bank),
        "--text",
        prompt_text,
        "--top-k",
        str(args.backup_top_k),
        "--segment-token-count",
        str(args.backup_segment_token_count),
        "--max-tokens-per-example",
        str(args.backup_prompt_tokens_per_example),
        "--output-prompt",
        str(prompt_file),
        "--output-json",
        str(retrieval_json),
    ]
    retrieval_plan_cmd = [
        sys.executable,
        "Script/stage1/llm_token_planning.py",
        "retrieval-plan",
        "--bank",
        str(bank),
        "--text",
        prompt_text,
        "--top-k",
        str(args.backup_top_k),
        "--segment-token-count",
        str(args.backup_segment_token_count),
        "--max-consecutive-repeat",
        str(args.backup_max_consecutive_repeat),
        "--output-tokens",
        str(tokens),
        "--validation-json",
        str(validation),
    ]
    if args.backup_trim_repeat_runs:
        retrieval_plan_cmd.append("--trim-repeat-runs")
    bvh = bvh_dir / f"{prompt_name}__backup_retrieval.bvh"
    motion_dataset = resolve_motion_dataset(args)
    decode_cmd = [
        sys.executable,
        "Script/stage1/llm_token_planning.py",
        "decode-bvh",
        "--tokens",
        str(tokens),
        "--base-data",
        args.base_data,
        "--gpu",
        str(args.gpu),
        "--output-bvh",
        str(bvh),
    ]
    if motion_dataset is not None:
        decode_cmd.extend(["--motion-dataset", str(motion_dataset)])
    if not args.skip_backup:
        run_command(build_prompt_cmd, log_path=backup_dir / "build_prompt.log")
        run_command(retrieval_plan_cmd, log_path=backup_dir / "retrieval_plan.log")
        if not args.skip_generation:
            run_command(decode_cmd, log_path=backup_dir / "decode_retrieval.log")
    return {
        "prompt": prompt_name,
        "model": "backup_retrieval",
        "path": str(bvh),
        "prompt_file": str(prompt_file),
        "tokens": str(tokens),
        "validation": str(validation),
        "command": _command_to_string(decode_cmd),
        "frames": count_bvh_frames(bvh) if bvh.exists() else None,
    }


def run_llm_response_backup(
    *,
    prompt_name: str,
    response_path: Path,
    bvh_dir: Path,
    suite_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    backup_dir = suite_dir / "llm_backup" / prompt_name
    backup_dir.mkdir(parents=True, exist_ok=True)
    tokens = backup_dir / "llm_tokens.json"
    validation = backup_dir / "llm_validation.json"
    validate_cmd = [
        sys.executable,
        "Script/stage1/llm_token_planning.py",
        "validate",
        "--response-file",
        str(response_path),
        "--output-tokens",
        str(tokens),
        "--validation-json",
        str(validation),
        "--min-length",
        str(args.llm_min_length),
        "--max-consecutive-repeat",
        str(args.llm_max_consecutive_repeat),
    ]
    if args.llm_repair:
        validate_cmd.append("--repair")
    bvh = bvh_dir / f"{prompt_name}__backup_llm.bvh"
    motion_dataset = resolve_motion_dataset(args)
    decode_cmd = [
        sys.executable,
        "Script/stage1/llm_token_planning.py",
        "decode-bvh",
        "--tokens",
        str(tokens),
        "--base-data",
        args.base_data,
        "--gpu",
        str(args.gpu),
        "--output-bvh",
        str(bvh),
    ]
    if motion_dataset is not None:
        decode_cmd.extend(["--motion-dataset", str(motion_dataset)])
    if not args.skip_backup:
        run_command(validate_cmd, log_path=backup_dir / "validate_llm.log")
        if not args.skip_generation:
            run_command(decode_cmd, log_path=backup_dir / "decode_llm.log")
    return {
        "prompt": prompt_name,
        "model": "backup_llm",
        "path": str(bvh),
        "response": str(response_path),
        "tokens": str(tokens),
        "validation": str(validation),
        "command": _command_to_string(decode_cmd),
        "frames": count_bvh_frames(bvh) if bvh.exists() else None,
    }


def load_llm_response_map(path: str | None) -> dict[str, Path]:
    if not path:
        return {}
    mapping_path = Path(path)
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--llm-response-map must be a JSON object mapping prompt names to response files")
    mapped: dict[str, Path] = {}
    for key, value in payload.items():
        response = Path(str(value))
        mapped[str(key)] = response if response.is_absolute() else mapping_path.parent / response
    return mapped


def summarize_by_model(metrics: dict[str, object]) -> dict[str, dict[str, float]]:
    rows = metrics.get("rows", [])
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:  # type: ignore[assignment]
        label = str(row.get("label", ""))
        model = label.split("__", 1)[1] if "__" in label else label
        grouped.setdefault(model, []).append(row)
    summary: dict[str, dict[str, float]] = {}
    for model, model_rows in grouped.items():
        numeric_keys = (
            "frames",
            "duration_sec",
            "root_path_length",
            "root_displacement",
            "pose_velocity_mean",
            "pose_variance_mean",
            "lag_20_repeat_fraction_0.995",
        )
        summary[model] = {}
        for key in numeric_keys:
            values = [float(row[key]) for row in model_rows if row.get(key) is not None]
            if values:
                summary[model][f"avg_{key}"] = sum(values) / len(values)
        early = [bool(row["early_stop"]) for row in model_rows if "early_stop" in row]
        if early:
            summary[model]["early_stop_rate"] = sum(1.0 for value in early if value) / len(early)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompts", default="")
    parser.add_argument("--baseline-checkpoint", default="text_generation_GPT.pth")
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--motion-dataset", default="")
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
    parser.add_argument("--expected-min-frames", type=int, default=1200)
    parser.add_argument("--sample-stride", type=int, default=6)
    parser.add_argument("--lags", default="5,10,20,30")
    parser.add_argument("--allow-early-stop", dest="allow_early_stop", action="store_true", default=True)
    parser.add_argument("--no-allow-early-stop", dest="allow_early_stop", action="store_false")
    parser.add_argument("--suite-dir", default="")
    parser.add_argument("--bvh-dir", default="")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--reuse-existing-bvh", action="store_true")
    parser.add_argument("--skip-gpt", action="store_true")
    parser.add_argument("--skip-backup", action="store_true")
    parser.add_argument("--backup-cache", default="")
    parser.add_argument("--example-bank", default="")
    parser.add_argument("--backup-max-examples", type=int, default=400)
    parser.add_argument("--backup-max-tokens-per-example", type=int, default=32)
    parser.add_argument("--backup-min-tokens-per-example", type=int, default=8)
    parser.add_argument("--backup-top-k", type=int, default=3)
    parser.add_argument("--backup-segment-token-count", type=int, default=25)
    parser.add_argument("--backup-prompt-tokens-per-example", type=int, default=16)
    parser.add_argument("--backup-max-consecutive-repeat", type=int, default=5)
    parser.add_argument("--backup-trim-repeat-runs", dest="backup_trim_repeat_runs", action="store_true", default=True)
    parser.add_argument("--no-backup-trim-repeat-runs", dest="backup_trim_repeat_runs", action="store_false")
    parser.add_argument("--llm-response-map", default="")
    parser.add_argument("--llm-min-length", type=int, default=20)
    parser.add_argument("--llm-max-consecutive-repeat", type=int, default=5)
    parser.add_argument("--llm-repair", action="store_true")
    args = parser.parse_args(argv)

    suite_dir = Path(args.suite_dir) if args.suite_dir else Path("stage1_artifacts/model_suite") / args.run_id
    bvh_dir = Path(args.bvh_dir) if args.bvh_dir else suite_dir / "bvh"
    suite_dir.mkdir(parents=True, exist_ok=True)
    bvh_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = Path(args.prompts) if args.prompts else suite_dir / "prompts.tsv"
    if not args.prompts:
        write_default_prompts(prompts_path)
    prompts = read_prompts(prompts_path)
    (suite_dir / "prompts.tsv").write_text("".join(format_prompt_tsv_line(prompt) for prompt in prompts), encoding="utf-8")

    generated: list[dict[str, object]] = []
    if not args.skip_gpt:
        for prompt in prompts:
            prompt_name, prompt_text = prompt
            generated.append(
                generate_gpt_bvh(
                    prompt_name=prompt_name,
                    prompt_text=prompt_text,
                    prompt_segments=prompt.segments,
                    prompt_segment_lengths=prompt.segment_lengths,
                    model_name="baseline_top_p",
                    checkpoint=args.baseline_checkpoint,
                    output_bvh=bvh_dir / f"{prompt_name}__baseline_top_p.bvh",
                    log_path=bvh_dir / f"{prompt_name}__baseline_top_p.log",
                    args=args,
                    progress_conditioning=args.baseline_progress_conditioning,
                )
            )
            generated.append(
                generate_gpt_bvh(
                    prompt_name=prompt_name,
                    prompt_text=prompt_text,
                    prompt_segments=prompt.segments,
                    prompt_segment_lengths=prompt.segment_lengths,
                    model_name="finetuned_top_p",
                    checkpoint=args.finetuned_checkpoint,
                    output_bvh=bvh_dir / f"{prompt_name}__finetuned_top_p.bvh",
                    log_path=bvh_dir / f"{prompt_name}__finetuned_top_p.log",
                    args=args,
                    progress_conditioning=args.progress_conditioning,
                )
            )

    llm_response_map = load_llm_response_map(args.llm_response_map)
    bank = None if args.skip_backup else ensure_example_bank(args, suite_dir)
    if not args.skip_backup:
        for prompt_name, prompt_text in prompts:
            if bank is not None:
                generated.append(
                    run_retrieval_backup(
                        prompt_name=prompt_name,
                        prompt_text=prompt_text,
                        bank=bank,
                        bvh_dir=bvh_dir,
                        suite_dir=suite_dir,
                        args=args,
                    )
                )
            if prompt_name in llm_response_map:
                generated.append(
                    run_llm_response_backup(
                        prompt_name=prompt_name,
                        response_path=llm_response_map[prompt_name],
                        bvh_dir=bvh_dir,
                        suite_dir=suite_dir,
                        args=args,
                    )
                )

    existing_bvhs = sorted(bvh_dir.glob("*.bvh"))
    metrics = evaluate_bvh_files(
        [str(path) for path in existing_bvhs],
        sample_stride=args.sample_stride,
        lags=tuple(int(item) for item in args.lags.split(",") if item.strip()),
        expected_min_frames=args.expected_min_frames,
    )
    metrics_path = suite_dir / "summary_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    motion_dataset = resolve_motion_dataset(args)
    summary = {
        "run_id": args.run_id,
        "suite_dir": str(suite_dir),
        "bvh_dir": str(bvh_dir),
        "prompts": [prompt_summary(prompt) for prompt in prompts],
        "generated": generated,
        "metrics": str(metrics_path),
        "model_averages": summarize_by_model(metrics),
        "config": {
            "baseline_checkpoint": args.baseline_checkpoint,
            "finetuned_checkpoint": args.finetuned_checkpoint,
            "backup_cache": args.backup_cache,
            "example_bank": str(bank) if bank is not None else None,
            "base_data": args.base_data,
            "motion_dataset": str(motion_dataset) if motion_dataset is not None else None,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "max_length": args.max_length,
            "generation_mode": args.generation_mode,
            "segment_joiner": args.segment_joiner,
            "context_size": args.context_size,
            "chunk_size": args.chunk_size,
            "progress_conditioning": args.progress_conditioning,
            "baseline_progress_conditioning": args.baseline_progress_conditioning,
            "progress_scale": args.progress_scale,
            "progress_context_size": args.progress_context_size,
            "progress_prefix_cap": args.progress_prefix_cap,
            "seed": args.seed,
            "expected_min_frames": args.expected_min_frames,
            "reuse_existing_bvh": args.reuse_existing_bvh,
            "backup_trim_repeat_runs": args.backup_trim_repeat_runs,
            "backup_max_consecutive_repeat": args.backup_max_consecutive_repeat,
        },
    }
    summary_path = suite_dir / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
