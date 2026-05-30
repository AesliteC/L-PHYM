from __future__ import annotations

import argparse
from pathlib import Path

import torch

import MoConVQCore.Utils.pytorch_utils as ptu
from Script.stage1.train_text_gpt import build_text_gpt_model, gpt_config


def encode_text_with_t5(text: str, model_name: str, max_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    encoder = T5EncoderModel.from_pretrained(model_name).to(device)
    encoder.eval()
    encoded = tokenizer(
        [text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = encoder(**encoded)
    return output.last_hidden_state, ~encoded["attention_mask"].bool()


def encode_text_with_hash(text: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    from Script.stage1.text_encoding import encode_text_to_feature

    feature, mask = encode_text_to_feature(text)
    return (
        torch.as_tensor(feature, dtype=torch.float32, device=device),
        torch.as_tensor(mask, dtype=torch.bool, device=device),
    )


def sample_latents_rolling(
    model,
    clip_feature: torch.Tensor,
    bert_feature: torch.Tensor,
    bert_mask: torch.Tensor,
    max_length: int,
    context_size: int | None = None,
    chunk_size: int | None = None,
    categorical: bool = True,
    allow_early_stop: bool = False,
) -> torch.Tensor:
    """Generate arbitrary-length latent sequences using fixed-size GPT contexts."""
    if max_length < 1:
        raise ValueError("max_length must be positive")

    block_size = int(model.get_block_size())
    max_context = block_size - 1
    if max_context < 1:
        raise ValueError(f"model block_size is too small for rolling generation: {block_size}")
    context_size = max_context if context_size is None else min(int(context_size), max_context)
    chunk_size = context_size if chunk_size is None else int(chunk_size)
    if context_size < 1 or chunk_size < 1:
        raise ValueError("context_size and chunk_size must be positive")

    generated: torch.Tensor | None = None
    produced = 0
    while produced < max_length:
        remaining = max_length - produced
        current_chunk = min(chunk_size, remaining)
        pre_latent = None
        context_len = 0
        if generated is not None:
            effective_context = min(context_size, max_context - current_chunk)
            if effective_context < 1:
                raise ValueError(
                    f"chunk_size {current_chunk} is too large for block_size {block_size}; "
                    f"use chunk_size <= {max_context - 1}"
                )
            pre_latent = generated[:, -effective_context:, :]
            context_len = int(pre_latent.shape[1])
        sample_length = current_chunk + 1
        sampled, _ = model.sample(
            clip_feature,
            bert_feature,
            bert_mask,
            if_categorial=categorical,
            max_length=sample_length,
            pre_latent=pre_latent,
        )
        new_latents = sampled[:, context_len:, :]
        if new_latents.shape[1] < current_chunk:
            if not allow_early_stop:
                raise RuntimeError(
                    f"GPT returned too few latents for chunk: expected {current_chunk}, got {new_latents.shape[1]}"
                )
            if new_latents.shape[1] == 0:
                break
            current_chunk = int(new_latents.shape[1])
        new_latents = new_latents[:, :current_chunk, :]
        generated = new_latents if generated is None else torch.cat([generated, new_latents], dim=1)
        produced += current_chunk
        if current_chunk < min(chunk_size, remaining):
            break

    if generated is None:
        raise RuntimeError("GPT generated no latents")
    return generated[:, :max_length, :]


def sample_latents_with_prefix(
    model,
    clip_feature: torch.Tensor,
    bert_feature: torch.Tensor,
    bert_mask: torch.Tensor,
    max_length: int,
    prefix_latents: torch.Tensor | None = None,
    context_size: int | None = None,
    chunk_size: int | None = None,
    categorical: bool = True,
    allow_early_stop: bool = False,
) -> torch.Tensor:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    block_size = int(model.get_block_size())
    max_context = block_size - 1
    context_size = max_context if context_size is None else min(int(context_size), max_context)
    chunk_size = max_length if chunk_size is None else int(chunk_size)
    if context_size < 1 or chunk_size < 1:
        raise ValueError("context_size and chunk_size must be positive")

    generated = prefix_latents
    produced = 0
    new_parts: list[torch.Tensor] = []
    while produced < max_length:
        remaining = max_length - produced
        current_chunk = min(chunk_size, remaining)
        pre_latent = None
        context_len = 0
        if generated is not None:
            effective_context = min(context_size, max_context - current_chunk, int(generated.shape[1]))
            if effective_context < 1:
                raise ValueError(
                    f"chunk_size {current_chunk} is too large for block_size {block_size}; "
                    f"use chunk_size <= {max_context - 1}"
                )
            pre_latent = generated[:, -effective_context:, :]
            context_len = int(pre_latent.shape[1])
        sampled, _ = model.sample(
            clip_feature,
            bert_feature,
            bert_mask,
            if_categorial=categorical,
            max_length=current_chunk + 1,
            pre_latent=pre_latent,
        )
        new_latents = sampled[:, context_len:, :]
        if new_latents.shape[1] < current_chunk:
            if not allow_early_stop:
                raise RuntimeError(
                    f"GPT returned too few latents for chunk: expected {current_chunk}, got {new_latents.shape[1]}"
                )
            if new_latents.shape[1] == 0:
                break
            current_chunk = int(new_latents.shape[1])
        new_latents = new_latents[:, :current_chunk, :]
        new_parts.append(new_latents)
        generated = new_latents if generated is None else torch.cat([generated, new_latents], dim=1)
        produced += current_chunk
        if current_chunk < min(chunk_size, remaining):
            break

    if not new_parts:
        raise RuntimeError("GPT generated no new latents")
    return torch.cat(new_parts, dim=1)[:, :max_length, :]


def split_text_segments(text: str, joiner: str = " then ") -> list[str]:
    segments = [segment.strip() for segment in text.split(joiner)]
    return [segment for segment in segments if segment]


def resolve_generation_mode(mode: str, text: str, segment_joiner: str) -> str:
    if mode not in {"auto", "rolling", "segmented"}:
        raise ValueError(f"unknown generation mode: {mode}")
    if mode != "auto":
        return mode
    return "segmented" if len(split_text_segments(text, joiner=segment_joiner)) > 1 else "rolling"


def parse_segment_lengths(value: str | None, expected_count: int) -> list[int] | None:
    if value is None or not value.strip():
        return None
    lengths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(lengths) != expected_count:
        raise ValueError(f"expected {expected_count} segment lengths, got {len(lengths)}")
    if any(length < 1 for length in lengths):
        raise ValueError("segment lengths must be positive")
    return lengths


def resolve_segment_lengths(
    segment_lengths_arg: str | None,
    segment_length_arg: int | None,
    max_length: int,
    expected_count: int,
) -> list[int]:
    explicit_lengths = parse_segment_lengths(segment_lengths_arg, expected_count=expected_count)
    if explicit_lengths is not None:
        return explicit_lengths
    if segment_length_arg is not None:
        if segment_length_arg < 1:
            raise ValueError("segment_length must be positive")
        return [segment_length_arg for _ in range(expected_count)]
    if max_length < expected_count:
        raise ValueError(
            f"max_length {max_length} is too short for {expected_count} text segments; "
            "pass --segment-length or --segment-lengths"
        )
    base = max_length // expected_count
    extra = max_length % expected_count
    return [base + (1 if idx < extra else 0) for idx in range(expected_count)]


def sample_latents_segmented(
    model,
    clip_feature: torch.Tensor,
    text_segments: list[str],
    text_encoder: str,
    text_model: str,
    max_text_length: int,
    device: str,
    segment_length: int,
    context_size: int | None,
    chunk_size: int | None,
    categorical: bool,
    allow_early_stop: bool,
    segment_lengths: list[int] | None = None,
) -> torch.Tensor:
    if not text_segments:
        raise ValueError("text_segments must not be empty")
    if segment_length < 1:
        raise ValueError("segment_length must be positive")
    if segment_lengths is not None and len(segment_lengths) != len(text_segments):
        raise ValueError(f"expected {len(text_segments)} segment lengths, got {len(segment_lengths)}")
    generated: torch.Tensor | None = None
    for segment_idx, segment in enumerate(text_segments):
        current_segment_length = segment_lengths[segment_idx] if segment_lengths is not None else segment_length
        if text_encoder == "t5":
            bert_feature, bert_mask = encode_text_with_t5(
                segment,
                model_name=text_model,
                max_length=max_text_length,
                device=device,
            )
        elif text_encoder == "hash":
            bert_feature, bert_mask = encode_text_with_hash(segment, device=device)
        else:
            raise ValueError(f"unknown text encoder: {text_encoder}")

        segment_latents = sample_latents_with_prefix(
            model=model,
            clip_feature=clip_feature,
            bert_feature=bert_feature,
            bert_mask=bert_mask,
            max_length=current_segment_length,
            prefix_latents=generated,
            context_size=context_size,
            chunk_size=chunk_size,
            categorical=categorical,
            allow_early_stop=allow_early_stop,
        )
        generated = segment_latents if generated is None else torch.cat([generated, segment_latents], dim=1)
        if segment_latents.shape[1] < current_segment_length and allow_early_stop:
            break
    if generated is None:
        raise RuntimeError("segmented generation produced no latents")
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-bvh", required=True)
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--text-encoder", choices=("t5", "hash"), default="t5")
    parser.add_argument("--text-model", default="t5-large")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--context-size", type=int, default=51)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--generation-mode", choices=("auto", "rolling", "segmented"), default="auto")
    parser.add_argument("--segment-joiner", default=" then ")
    parser.add_argument("--segment-length", type=int, default=None)
    parser.add_argument("--segment-lengths", default=None)
    parser.add_argument("--allow-early-stop", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ptu.init_gpu(True, gpu_id=args.gpu)
    from Script.stage1.real_moconvq_cache import build_loaded_moconvq_agent

    agent = build_loaded_moconvq_agent(gpu=args.gpu, base_data=Path(args.base_data))
    agent.eval()

    model = build_text_gpt_model(gpt_config(), device=ptu.device, base_data_path=args.base_data)
    state = torch.load(args.checkpoint, map_location="cpu")
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    generation_mode = resolve_generation_mode(args.generation_mode, args.text, args.segment_joiner)
    if generation_mode == "rolling" and args.text_encoder == "t5":
        bert_feature, bert_mask = encode_text_with_t5(
            args.text,
            model_name=args.text_model,
            max_length=args.max_text_length,
            device=ptu.device,
        )
    elif generation_mode == "rolling":
        bert_feature, bert_mask = encode_text_with_hash(args.text, device=ptu.device)
    clip_feature = torch.zeros((1, 512), device=ptu.device)
    if generation_mode == "rolling":
        cur_embedding = sample_latents_rolling(
            model=model,
            clip_feature=clip_feature,
            bert_feature=bert_feature,
            bert_mask=bert_mask,
            max_length=args.max_length,
            context_size=args.context_size,
            chunk_size=args.chunk_size,
            categorical=not args.greedy,
            allow_early_stop=args.allow_early_stop,
        )
    else:
        segments = split_text_segments(args.text, joiner=args.segment_joiner)
        segment_lengths = resolve_segment_lengths(
            segment_lengths_arg=args.segment_lengths,
            segment_length_arg=args.segment_length,
            max_length=args.max_length,
            expected_count=len(segments),
        )
        cur_embedding = sample_latents_segmented(
            model=model,
            clip_feature=clip_feature,
            text_segments=segments,
            text_encoder=args.text_encoder,
            text_model=args.text_model,
            max_text_length=args.max_text_length,
            device=ptu.device,
            segment_length=segment_lengths[0],
            segment_lengths=segment_lengths,
            context_size=args.context_size,
            chunk_size=args.chunk_size,
            categorical=not args.greedy,
            allow_early_stop=args.allow_early_stop,
        )
    dconv = agent.posterior.decoder.decode_dynamic(cur_embedding)

    import VclSimuBackend

    CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
    saver = CharacterToBVH(agent.env.sim_character, 120)
    saver.bvh_hierarchy_no_root()

    observation, info = agent.env.reset(0)

    for i in range(dconv.shape[1]):
        obs = observation["observation"]
        action, info = agent.act_tracking(
            obs_history=[obs.reshape(1, 323)],
            target_latent=dconv[:, i],
        )
        action = ptu.to_numpy(action).flatten()
        for j in range(6):
            saver.append_no_root_to_buffer()
            if j == 0:
                step_generator = agent.env.step_core(action, using_yield=True)
            info = next(step_generator)

        try:
            info_ = next(step_generator)
        except StopIteration as e:
            info_ = e.value
        new_observation, rwd, done, info = info_
        observation = new_observation

    Path(args.output_bvh).parent.mkdir(parents=True, exist_ok=True)
    saver.to_file(args.output_bvh)


if __name__ == "__main__":
    main()
