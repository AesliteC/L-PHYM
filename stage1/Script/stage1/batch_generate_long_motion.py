from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
import sys


def _ensure_own_repo_root_on_path(package: str | None = __package__) -> None:
    if package not in {None, ""}:
        return
    repo_root = str(Path(__file__).resolve().parents[2])
    if not sys.path or sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)


_ensure_own_repo_root_on_path()

import torch

import MoConVQCore.Utils.pytorch_utils as ptu
from Script.stage1.generate_long_motion import (
    resolve_generation_mode,
    resolve_segment_lengths,
    sample_latents_rolling,
    sample_latents_with_prefix,
    split_text_segments,
)
from Script.stage1.run_text_gpt_comparison import PromptRecord, read_prompts
from Script.stage1.segment_conditioning import add_progress_to_clip_feature
from Script.stage1.train_text_gpt import build_text_gpt_model, gpt_config


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def encode_segments_with_t5(
    segments: Iterable[str],
    *,
    model_name: str,
    max_length: int,
    device: torch.device | str,
    batch_size: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    from transformers import T5EncoderModel, T5Tokenizer

    unique_segments = list(dict.fromkeys(str(segment) for segment in segments))
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    encoder = T5EncoderModel.from_pretrained(model_name).to(device)
    encoder.eval()
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for start in range(0, len(unique_segments), batch_size):
            batch = unique_segments[start : start + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = encoder(**tokens)
            features = output.last_hidden_state.detach().cpu()
            masks = (~tokens["attention_mask"].bool()).detach().cpu()
            for idx, segment in enumerate(batch):
                encoded[segment] = (features[idx : idx + 1].contiguous(), masks[idx : idx + 1].contiguous())
            print(f"encoded T5 segments {min(start + len(batch), len(unique_segments))}/{len(unique_segments)}", flush=True)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return encoded


def encode_text_with_hash_cached(text: str, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    from Script.stage1.generate_long_motion import encode_text_with_hash

    return encode_text_with_hash(text, device=str(device))


def load_text_gpt(checkpoint: Path, *, base_data: Path, device: torch.device | str):
    model = build_text_gpt_model(gpt_config(), device=device, base_data_path=str(base_data))
    state = torch.load(checkpoint, map_location="cpu")
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): value for k, value in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def sample_prompt_latents(
    *,
    model,
    prompt: PromptRecord,
    encoded_segments: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    text_encoder: str,
    text_model: str,
    max_text_length: int,
    max_length: int,
    generation_mode: str,
    segment_joiner: str,
    context_size: int,
    chunk_size: int,
    allow_early_stop: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    progress_conditioning: str,
    progress_scale: float,
    progress_context_size: int | None,
    progress_prefix_cap: int | None,
    device: torch.device | str,
) -> torch.Tensor:
    clip_feature = torch.zeros((1, 512), device=device)
    explicit_segments = list(prompt.segments) if prompt.segments else None
    mode = resolve_generation_mode(
        generation_mode,
        prompt.text,
        segment_joiner,
        explicit_segments=explicit_segments,
    )
    if mode == "rolling":
        if text_encoder == "t5":
            assert encoded_segments is not None
            bert_feature, bert_mask = encoded_segments[prompt.text]
            bert_feature = bert_feature.to(device)
            bert_mask = bert_mask.to(device)
        else:
            bert_feature, bert_mask = encode_text_with_hash_cached(prompt.text, device)
        return sample_latents_rolling(
            model=model,
            clip_feature=clip_feature,
            bert_feature=bert_feature,
            bert_mask=bert_mask,
            max_length=max_length,
            context_size=context_size,
            chunk_size=chunk_size,
            categorical=True,
            allow_early_stop=allow_early_stop,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            progress_conditioning=progress_conditioning,
            progress_scale=progress_scale,
            progress_context_size=progress_context_size,
            progress_prefix_cap=progress_prefix_cap,
        )

    segments = explicit_segments if explicit_segments is not None else split_text_segments(prompt.text, joiner=segment_joiner)
    segment_lengths = list(prompt.segment_lengths) if prompt.segment_lengths else resolve_segment_lengths(
        segment_lengths_arg=None,
        segment_length_arg=None,
        max_length=max_length,
        expected_count=len(segments),
    )
    generated: torch.Tensor | None = None
    total_segments = len(segments)
    for segment_idx, segment in enumerate(segments):
        if text_encoder == "t5":
            assert encoded_segments is not None
            bert_feature, bert_mask = encoded_segments[segment]
            bert_feature = bert_feature.to(device)
            bert_mask = bert_mask.to(device)
        elif text_encoder == "hash":
            bert_feature, bert_mask = encode_text_with_hash_cached(segment, device)
        else:
            raise ValueError(f"unknown text encoder: {text_encoder}")
        segment_clip_feature = add_progress_to_clip_feature(
            clip_feature,
            mode=progress_conditioning,
            segment_idx=segment_idx,
            num_segments=total_segments,
            segment_progress=float(segment_idx / max(total_segments - 1, 1)) if total_segments > 1 else 0.0,
            prefix_lengths=0
            if generated is None
            else min(
                int(generated.shape[1]),
                int(progress_prefix_cap) if progress_prefix_cap is not None else int(context_size),
            ),
            context_size=progress_context_size if progress_context_size is not None else context_size,
            scale=progress_scale,
            has_segment_metadata=True,
            is_segmented=True,
        )
        segment_latents = sample_latents_with_prefix(
            model=model,
            clip_feature=segment_clip_feature,
            bert_feature=bert_feature,
            bert_mask=bert_mask,
            max_length=int(segment_lengths[segment_idx]),
            prefix_latents=generated,
            context_size=context_size,
            chunk_size=chunk_size,
            categorical=True,
            allow_early_stop=allow_early_stop,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        generated = segment_latents if generated is None else torch.cat([generated, segment_latents], dim=1)
    if generated is None:
        raise RuntimeError(f"no latents generated for prompt {prompt.name}")
    return generated


def write_latents_to_bvh(agent, latents: torch.Tensor, output_bvh: Path) -> int:
    dconv = agent.posterior.decoder.decode_dynamic(latents)

    import VclSimuBackend

    saver = VclSimuBackend.ODESim.CharacterTOBVH(agent.env.sim_character, 120)
    saver.bvh_hierarchy_no_root()
    observation, _info = agent.env.reset(0)
    for frame_idx in range(dconv.shape[1]):
        obs = observation["observation"]
        action, _info = agent.act_tracking(
            obs_history=[obs.reshape(1, 323)],
            target_latent=dconv[:, frame_idx],
        )
        action = ptu.to_numpy(action).flatten()
        for substep in range(6):
            saver.append_no_root_to_buffer()
            if substep == 0:
                step_generator = agent.env.step_core(action, using_yield=True)
            _info = next(step_generator)
        try:
            _info = next(step_generator)
        except StopIteration as exc:
            _info = exc.value
        observation, _rwd, _done, _info = _info
    output_bvh.parent.mkdir(parents=True, exist_ok=True)
    saver.to_file(str(output_bvh))
    return int(dconv.shape[1] * 6)


def generate_for_checkpoint(
    *,
    checkpoint: Path,
    model_name: str,
    prompts: list[PromptRecord],
    output_dir: Path,
    agent,
    encoded_segments: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    args: argparse.Namespace,
    progress_conditioning: str,
) -> list[dict[str, object]]:
    model = load_text_gpt(checkpoint, base_data=Path(args.base_data), device=ptu.device)
    rows: list[dict[str, object]] = []
    for idx, prompt in enumerate(prompts, start=1):
        output_bvh = output_dir / f"{prompt.name}__{model_name}.bvh"
        if args.reuse_existing and output_bvh.exists():
            rows.append({"prompt": prompt.name, "model": model_name, "path": str(output_bvh), "reused_existing": True})
            print(f"[{model_name}] reuse {idx}/{len(prompts)} {prompt.name}", flush=True)
            continue
        set_seed(args.seed)
        with torch.no_grad():
            latents = sample_prompt_latents(
                model=model,
                prompt=prompt,
                encoded_segments=encoded_segments,
                text_encoder=args.text_encoder,
                text_model=args.text_model,
                max_text_length=args.max_text_length,
                max_length=args.max_length,
                generation_mode=args.generation_mode,
                segment_joiner=args.segment_joiner,
                context_size=args.context_size,
                chunk_size=args.chunk_size,
                allow_early_stop=args.allow_early_stop,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                progress_conditioning=progress_conditioning,
                progress_scale=args.progress_scale,
                progress_context_size=args.progress_context_size,
                progress_prefix_cap=args.progress_prefix_cap,
                device=ptu.device,
            )
            frames = write_latents_to_bvh(agent, latents, output_bvh)
        rows.append(
            {
                "prompt": prompt.name,
                "model": model_name,
                "path": str(output_bvh),
                "frames": frames,
                "reused_existing": False,
            }
        )
        print(f"[{model_name}] wrote {idx}/{len(prompts)} {output_bvh} frames={frames}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def collect_texts_for_encoding(prompts: list[PromptRecord], generation_mode: str, segment_joiner: str) -> list[str]:
    texts: list[str] = []
    for prompt in prompts:
        explicit_segments = list(prompt.segments) if prompt.segments else None
        mode = resolve_generation_mode(generation_mode, prompt.text, segment_joiner, explicit_segments=explicit_segments)
        if mode == "rolling":
            texts.append(prompt.text)
        else:
            texts.extend(explicit_segments if explicit_segments is not None else split_text_segments(prompt.text, joiner=segment_joiner))
    return texts


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--motion-dataset", default="")
    parser.add_argument("--text-encoder", choices=("t5", "hash"), default="t5")
    parser.add_argument("--text-model", default="t5-large")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--t5-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=75)
    parser.add_argument("--generation-mode", choices=("auto", "rolling", "segmented"), default="auto")
    parser.add_argument("--segment-joiner", default=" then ")
    parser.add_argument("--context-size", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--baseline-progress-conditioning", choices=("none", "scalar", "auto"), default="none")
    parser.add_argument("--progress-conditioning", choices=("none", "scalar", "auto"), default="auto")
    parser.add_argument("--progress-scale", type=float, default=1.0)
    parser.add_argument("--progress-context-size", type=int, default=None)
    parser.add_argument("--progress-prefix-cap", type=int, default=None)
    parser.add_argument("--allow-early-stop", dest="allow_early_stop", action="store_true", default=True)
    parser.add_argument("--no-allow-early-stop", dest="allow_early_stop", action="store_false")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--summary", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")

    prompts = read_prompts(Path(args.prompts))
    output_dir = Path(args.output_dir)
    bvh_dir = output_dir / "bvh"
    output_dir.mkdir(parents=True, exist_ok=True)
    bvh_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts.tsv").write_text(Path(args.prompts).read_text(encoding="utf-8"), encoding="utf-8")

    ptu.init_gpu(True, gpu_id=args.gpu)
    from Script.stage1.real_moconvq_cache import build_loaded_moconvq_agent

    agent = build_loaded_moconvq_agent(
        gpu=args.gpu,
        base_data=Path(args.base_data),
        motion_dataset=Path(args.motion_dataset) if args.motion_dataset else None,
    )
    agent.eval()

    encoded_segments = None
    if args.text_encoder == "t5":
        encoded_segments = encode_segments_with_t5(
            collect_texts_for_encoding(prompts, args.generation_mode, args.segment_joiner),
            model_name=args.text_model,
            max_length=args.max_text_length,
            device=ptu.device,
            batch_size=args.t5_batch_size,
        )

    generated: list[dict[str, object]] = []
    generated.extend(
        generate_for_checkpoint(
            checkpoint=Path(args.baseline_checkpoint),
            model_name="baseline_top_p",
            prompts=prompts,
            output_dir=bvh_dir,
            agent=agent,
            encoded_segments=encoded_segments,
            args=args,
            progress_conditioning=args.baseline_progress_conditioning,
        )
    )
    generated.extend(
        generate_for_checkpoint(
            checkpoint=Path(args.finetuned_checkpoint),
            model_name="finetuned_top_p",
            prompts=prompts,
            output_dir=bvh_dir,
            agent=agent,
            encoded_segments=encoded_segments,
            args=args,
            progress_conditioning=args.progress_conditioning,
        )
    )
    payload = {
        "prompts": str(Path(args.prompts)),
        "output_dir": str(output_dir),
        "bvh_dir": str(bvh_dir),
        "num_prompts": len(prompts),
        "generated": generated,
        "config": vars(args),
    }
    summary_path = Path(args.summary) if args.summary else output_dir / "batch_generate_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "num_generated": len(generated)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
