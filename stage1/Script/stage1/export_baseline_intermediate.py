from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import MoConVQCore.Utils.pytorch_utils as ptu
from Script.stage1.generate_long_motion import (
    encode_text_with_hash,
    encode_text_with_t5,
    resolve_generation_mode,
    resolve_segment_lengths,
    split_text_segments,
)
from Script.stage1.intermediate_motion_format import (
    DEFAULT_CONTROL_FPS,
    DEFAULT_MOTION_FPS,
    DYNAMIC_CONTROL_DIM,
    reshape_sample_indices,
    validate_intermediate_npz,
    write_format_markdown,
    write_intermediate_npz,
)
from Script.stage1.real_moconvq_cache import build_loaded_moconvq_agent
from Script.stage1.segment_conditioning import (
    PROGRESS_CONDITIONING_CHOICES,
    add_progress_to_clip_feature,
)
from Script.stage1.train_text_gpt import build_text_gpt_model, gpt_config


DEFAULT_PROMPTS = [
    "A person walks forward and turns left.",
    "A person waves with the right hand while standing.",
    "A person walks forward then raises both arms.",
]


def sample_id_from_prompt(index: int, prompt: str) -> str:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"baseline_{index:03d}_{digest}"


def _load_text_gpt(checkpoint: Path, base_data: Path, device: torch.device):
    model = build_text_gpt_model(gpt_config(), device=device, base_data_path=str(base_data))
    state = torch.load(checkpoint, map_location="cpu")
    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _encode_text(text_encoder: str, prompt: str, text_model: str, max_text_length: int, device: torch.device):
    if text_encoder == "t5":
        return encode_text_with_t5(prompt, model_name=text_model, max_length=max_text_length, device=str(device))
    if text_encoder == "hash":
        return encode_text_with_hash(prompt, device=str(device))
    raise ValueError(f"unknown text_encoder: {text_encoder}")


def _sample_chunk_with_indices(
    model,
    *,
    clip_feature: torch.Tensor,
    bert_feature: torch.Tensor,
    bert_mask: torch.Tensor,
    current_chunk: int,
    pre_latent: torch.Tensor | None,
    categorical: bool,
    allow_early_stop: bool,
    top_k: int,
    top_p: float,
    temperature: float,
) -> tuple[torch.Tensor, np.ndarray]:
    context_len = 0 if pre_latent is None else int(pre_latent.shape[1])
    sampled, raw_indices = model.sample(
        clip_feature,
        bert_feature,
        bert_mask,
        if_categorial=categorical,
        max_length=current_chunk + 1,
        pre_latent=pre_latent,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    new_latents = sampled[:, context_len:, :]
    if new_latents.shape[1] < current_chunk:
        if not allow_early_stop:
            raise RuntimeError(f"GPT returned too few latents: expected {current_chunk}, got {new_latents.shape[1]}")
        current_chunk = int(new_latents.shape[1])
    new_latents = new_latents[:, :current_chunk, :]
    indices = reshape_sample_indices(raw_indices.detach().cpu().numpy(), current_chunk)
    return new_latents, indices


def _sample_rolling_with_indices(
    model,
    *,
    clip_feature: torch.Tensor,
    bert_feature: torch.Tensor,
    bert_mask: torch.Tensor,
    max_length: int,
    context_size: int,
    chunk_size: int,
    categorical: bool,
    allow_early_stop: bool,
    top_k: int,
    top_p: float,
    temperature: float,
) -> tuple[torch.Tensor, np.ndarray]:
    block_size = int(model.get_block_size())
    max_context = block_size - 1
    context_size = min(int(context_size), max_context)
    generated: torch.Tensor | None = None
    index_parts: list[np.ndarray] = []
    produced = 0
    while produced < max_length:
        remaining = max_length - produced
        current_chunk = min(int(chunk_size), remaining)
        pre_latent = None
        if generated is not None:
            effective_context = min(context_size, max_context - current_chunk)
            if effective_context < 1:
                raise ValueError(
                    f"chunk_size {current_chunk} is too large for block_size {block_size}; "
                    f"use chunk_size <= {max_context - 1}"
                )
            pre_latent = generated[:, -effective_context:, :]
        new_latents, indices = _sample_chunk_with_indices(
            model,
            clip_feature=clip_feature,
            bert_feature=bert_feature,
            bert_mask=bert_mask,
            current_chunk=current_chunk,
            pre_latent=pre_latent,
            categorical=categorical,
            allow_early_stop=allow_early_stop,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        if new_latents.shape[1] == 0:
            break
        generated = new_latents if generated is None else torch.cat([generated, new_latents], dim=1)
        index_parts.append(indices[: int(new_latents.shape[1])])
        produced += int(new_latents.shape[1])
        if int(new_latents.shape[1]) < current_chunk:
            break
    if generated is None or not index_parts:
        raise RuntimeError("GPT generated no latents")
    return generated[:, :max_length, :], np.concatenate(index_parts, axis=0)[:max_length]


def _sample_segmented_with_indices(
    model,
    *,
    clip_feature: torch.Tensor,
    text_segments: list[str],
    text_encoder: str,
    text_model: str,
    max_text_length: int,
    device: torch.device,
    segment_lengths: list[int],
    context_size: int,
    chunk_size: int,
    categorical: bool,
    allow_early_stop: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    progress_conditioning: str,
    progress_scale: float,
) -> tuple[torch.Tensor, np.ndarray]:
    block_size = int(model.get_block_size())
    max_context = block_size - 1
    context_size = min(int(context_size), max_context)
    generated: torch.Tensor | None = None
    index_parts: list[np.ndarray] = []
    total_segments = len(text_segments)
    for segment_idx, (segment, segment_len) in enumerate(zip(text_segments, segment_lengths)):
        bert_feature, bert_mask = _encode_text(text_encoder, segment, text_model, max_text_length, device)
        segment_clip_feature = add_progress_to_clip_feature(
            clip_feature,
            mode=progress_conditioning,
            segment_idx=segment_idx,
            num_segments=total_segments,
            segment_progress=float(segment_idx / max(total_segments - 1, 1)) if total_segments > 1 else 0.0,
            prefix_lengths=0 if generated is None else min(int(generated.shape[1]), int(context_size)),
            context_size=context_size,
            scale=progress_scale,
            has_segment_metadata=True,
            is_segmented=True,
        )
        produced = 0
        while produced < segment_len:
            remaining = segment_len - produced
            current_chunk = min(int(chunk_size), remaining)
            pre_latent = None
            if generated is not None:
                effective_context = min(context_size, max_context - current_chunk, int(generated.shape[1]))
                if effective_context < 1:
                    raise ValueError(
                        f"chunk_size {current_chunk} is too large for block_size {block_size}; "
                        f"use chunk_size <= {max_context - 1}"
                    )
                pre_latent = generated[:, -effective_context:, :]
            new_latents, indices = _sample_chunk_with_indices(
                model,
                clip_feature=segment_clip_feature,
                bert_feature=bert_feature,
                bert_mask=bert_mask,
                current_chunk=current_chunk,
                pre_latent=pre_latent,
                categorical=categorical,
                allow_early_stop=allow_early_stop,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            if new_latents.shape[1] == 0:
                break
            generated = new_latents if generated is None else torch.cat([generated, new_latents], dim=1)
            index_parts.append(indices[: int(new_latents.shape[1])])
            produced += int(new_latents.shape[1])
            if int(new_latents.shape[1]) < current_chunk:
                break
    if generated is None or not index_parts:
        raise RuntimeError("segmented generation produced no latents")
    return generated, np.concatenate(index_parts, axis=0)


def _sample_latents_and_indices(
    model,
    *,
    prompt: str,
    text_encoder: str,
    text_model: str,
    max_text_length: int,
    max_length: int,
    context_size: int,
    chunk_size: int,
    generation_mode: str,
    segment_joiner: str,
    segment_length: int | None,
    segment_lengths: str | None,
    allow_early_stop: bool,
    greedy: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    progress_conditioning: str,
    progress_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, str]:
    mode = resolve_generation_mode(generation_mode, prompt, segment_joiner)
    clip_feature = torch.zeros((1, 512), device=device)
    if mode == "rolling":
        bert_feature, bert_mask = _encode_text(text_encoder, prompt, text_model, max_text_length, device)
        latents, indices = _sample_rolling_with_indices(
            model=model,
            clip_feature=clip_feature,
            bert_feature=bert_feature,
            bert_mask=bert_mask,
            max_length=max_length,
            context_size=context_size,
            chunk_size=chunk_size,
            categorical=not greedy,
            allow_early_stop=allow_early_stop,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        return latents, indices, mode

    segments = split_text_segments(prompt, joiner=segment_joiner)
    resolved_lengths = resolve_segment_lengths(
        segment_lengths_arg=segment_lengths,
        segment_length_arg=segment_length,
        max_length=max_length,
        expected_count=len(segments),
    )
    latents, indices = _sample_segmented_with_indices(
        model=model,
        clip_feature=clip_feature,
        text_segments=segments,
        text_encoder=text_encoder,
        text_model=text_model,
        max_text_length=max_text_length,
        device=device,
        segment_lengths=resolved_lengths,
        context_size=context_size,
        chunk_size=chunk_size,
        categorical=not greedy,
        allow_early_stop=allow_early_stop,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        progress_conditioning=progress_conditioning,
        progress_scale=progress_scale,
    )
    return latents, indices[: int(latents.shape[1])], mode


def _fake_sample(index: int, max_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=20260529 + index)
    latent = rng.normal(0.0, 0.05, size=(max_length, 768)).astype(np.float32)
    dynamic = rng.normal(0.0, 0.05, size=(max_length * 4, DYNAMIC_CONTROL_DIM)).astype(np.float32)
    indices = rng.integers(0, 512, size=(max_length, 4), dtype=np.int64)
    return latent, dynamic, indices


def export_samples(args: argparse.Namespace) -> list[dict[str, object]]:
    prompts = args.prompt or DEFAULT_PROMPTS
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    write_format_markdown(output_dir / "MOCONVQ_INTERMEDIATE_FORMAT.md")

    rows: list[dict[str, object]] = []
    if args.skip_model:
        model = None
        agent = None
        device = torch.device("cpu")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        ptu.init_gpu(True, gpu_id=args.gpu)
        device = ptu.device
        agent = build_loaded_moconvq_agent(gpu=args.gpu, base_data=Path(args.base_data))
        model = _load_text_gpt(Path(args.checkpoint), Path(args.base_data), device=device)

    for idx, prompt in enumerate(prompts):
        sample_id = sample_id_from_prompt(idx, prompt)
        sample_path = samples_dir / f"{sample_id}.npz"
        if args.skip_model:
            latent_np, dynamic_np, indices_np = _fake_sample(idx, args.max_length)
            mode = resolve_generation_mode(args.generation_mode, prompt, args.segment_joiner)
        else:
            assert model is not None and agent is not None
            latents, indices_np, mode = _sample_latents_and_indices(
                model,
                prompt=prompt,
                text_encoder=args.text_encoder,
                text_model=args.text_model,
                max_text_length=args.max_text_length,
                max_length=args.max_length,
                context_size=args.context_size,
                chunk_size=args.chunk_size,
                generation_mode=args.generation_mode,
                segment_joiner=args.segment_joiner,
                segment_length=args.segment_length,
                segment_lengths=args.segment_lengths,
                allow_early_stop=args.allow_early_stop,
                greedy=args.greedy,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                progress_conditioning=args.progress_conditioning,
                progress_scale=args.progress_scale,
                device=device,
            )
            with torch.no_grad():
                dynamic = agent.posterior.decoder.decode_dynamic(latents)
            latent_np = latents.squeeze(0).detach().cpu().numpy().astype(np.float32)
            dynamic_np = dynamic.squeeze(0).detach().cpu().numpy().astype(np.float32)
            indices_np = indices_np[: latent_np.shape[0]]

        metadata = write_intermediate_npz(
            sample_path,
            motion_latent=latent_np,
            dynamic_control=dynamic_np,
            rvq_indices=indices_np,
            metadata={
                "sample_id": sample_id,
                "prompt": prompt,
                "checkpoint": str(args.checkpoint),
                "base_data": str(args.base_data),
                "text_encoder": args.text_encoder,
                "text_model": args.text_model,
                "generation_mode": mode,
                "max_length": args.max_length,
                "context_size": args.context_size,
                "chunk_size": args.chunk_size,
                "greedy": bool(args.greedy),
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "progress_conditioning": args.progress_conditioning,
                "progress_scale": args.progress_scale,
                "seed": args.seed,
                "motion_fps": DEFAULT_MOTION_FPS,
                "control_fps": DEFAULT_CONTROL_FPS,
                "skip_model": bool(args.skip_model),
            },
        )
        summary = validate_intermediate_npz(sample_path)
        row = {
            "sample_id": sample_id,
            "prompt": prompt,
            "path": str(sample_path.relative_to(output_dir)),
            "motion_latent_shape": summary["motion_latent_shape"],
            "dynamic_control_shape": summary["dynamic_control_shape"],
            "rvq_indices_shape": summary["rvq_indices_shape"],
            "generation_mode": mode,
            "dynamic_steps_per_motion_token": metadata["dynamic_steps_per_motion_token"],
        }
        rows.append(row)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# Baseline MoConVQ 中间层导出\n\n"
        "这个目录保存 baseline `text_generation_GPT.pth` 的中间层输出。"
        "读取 `.npz` 文件前，请先看 `MOCONVQ_INTERMEDIATE_FORMAT.md`。\n\n"
        f"- 样例数量：{len(rows)}\n"
        f"- GPT checkpoint：`{args.checkpoint}`\n"
        f"- MoConVQ base data：`{args.base_data}`\n"
        f"- manifest：`manifest.jsonl`\n",
        encoding="utf-8",
    )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="text_generation_GPT.pth")
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--output-dir", default="stage1_artifacts/baseline_intermediate_export")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--text-encoder", choices=("t5", "hash"), default="t5")
    parser.add_argument("--text-model", default="/home/chenjie/cc/robotics/hf_models/t5-large")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=24)
    parser.add_argument("--context-size", type=int, default=51)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--generation-mode", choices=("auto", "rolling", "segmented"), default="auto")
    parser.add_argument("--segment-joiner", default=" then ")
    parser.add_argument("--segment-length", type=int, default=None)
    parser.add_argument("--segment-lengths", default=None)
    parser.add_argument("--allow-early-stop", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--progress-conditioning", choices=PROGRESS_CONDITIONING_CHOICES, default="none")
    parser.add_argument("--progress-scale", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--make-archive", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    rows = export_samples(args)
    output_dir = Path(args.output_dir)
    if args.make_archive:
        archive_base = str(output_dir)
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=output_dir)
        print(json.dumps({"samples": len(rows), "output_dir": str(output_dir), "archive": archive_path}, indent=2))
    else:
        print(json.dumps({"samples": len(rows), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
