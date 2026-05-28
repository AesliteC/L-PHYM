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

    if args.text_encoder == "t5":
        bert_feature, bert_mask = encode_text_with_t5(
            args.text,
            model_name=args.text_model,
            max_length=args.max_text_length,
            device=ptu.device,
        )
    else:
        bert_feature, bert_mask = encode_text_with_hash(args.text, device=ptu.device)
    clip_feature = torch.zeros((1, 512), device=ptu.device)
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
