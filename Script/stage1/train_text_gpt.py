from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import MoConVQCore.Utils.pytorch_utils as ptu
from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer

from Script.stage1.motion_bridge import extract_rvq_embeddings_from_state_dict


@dataclass
class gpt_config:
    num_vq: int = 512
    embed_dim: int = 768
    clip_dim: int = 512
    block_size: int = 52
    num_layers: int = 9
    n_head: int = 8
    drop_out_rate: float = 0.1
    fc_rate: int = 2


class Stage1CacheDataset(Dataset):
    def __init__(self, cache_path: str):
        self.cache = torch.load(cache_path, map_location="cpu")

    def __len__(self):
        return len(self.cache["indices"])

    def __getitem__(self, idx):
        return {
            "latent": torch.as_tensor(self.cache["latents"][idx], dtype=torch.float32),
            "indices": torch.as_tensor(self.cache["indices"][idx], dtype=torch.long),
            "text_feature": torch.as_tensor(self.cache["text_features"][idx], dtype=torch.float32),
            "text_mask": torch.as_tensor(self.cache["text_masks"][idx], dtype=torch.bool),
            "caption": self.cache["captions"][idx],
        }


def load_gpt_embeddings(base_data_path: str) -> list[torch.Tensor]:
    state = torch.load(base_data_path, map_location="cpu")
    embeddings = extract_rvq_embeddings_from_state_dict(state)
    return [
        torch.cat(
            [
                torch.as_tensor(embedding, dtype=torch.float32),
                torch.zeros((2, embedding.shape[1]), dtype=torch.float32),
            ],
            dim=0,
        )
        for embedding in embeddings
    ]


def build_text_gpt_model(cfg: gpt_config, device: str = "cpu", base_data_path: str = "moconvq_base.data") -> Text2Motion_Transformer:
    embeddings = load_gpt_embeddings(base_data_path)
    model = Text2Motion_Transformer(**cfg.__dict__, embeddings=embeddings).to(device)
    # Text2Motion_Transformer keeps the RVQ codebooks in a plain Python list for
    # sampling, so module.to(device) does not move them automatically.
    model.embedding = [embedding.to(device) for embedding in model.embedding]
    return model


def _load_state_dict_flexible(model: nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("missing_keys", missing[:10])
    if unexpected:
        print("unexpected_keys", unexpected[:10])


def reconstruct_latents_from_rvq_indices(
    indices: torch.Tensor,
    embeddings: list[torch.Tensor],
    pad_index: int = 513,
) -> torch.Tensor:
    if indices.ndim != 3:
        raise ValueError(f"expected indices shape (B, T, D), got {indices.shape}")
    depth = int(indices.shape[-1])
    if depth > len(embeddings):
        raise ValueError(f"need {depth} RVQ embedding tables, got {len(embeddings)}")
    if depth < 1:
        raise ValueError("indices must contain at least one RVQ depth")

    first_embedding = embeddings[0]
    if first_embedding.ndim != 2:
        raise ValueError(f"expected embedding table shape (V, C), got {first_embedding.shape}")
    latents = torch.zeros(
        (*indices.shape[:2], first_embedding.shape[-1]),
        dtype=first_embedding.dtype,
        device=indices.device,
    )
    for rvq_depth in range(depth):
        table = embeddings[rvq_depth].to(device=indices.device, dtype=first_embedding.dtype)
        safe_indices = indices[..., rvq_depth].clone()
        pad_mask = safe_indices == pad_index
        if torch.any((safe_indices < 0) | (safe_indices >= table.shape[0])):
            safe_indices = safe_indices.clamp(min=0, max=table.shape[0] - 1)
        values = table[safe_indices]
        if pad_mask.any():
            values = values.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        latents = latents + values
    return latents


def train_one_epoch(model, loader, optimizer, device, max_batches: int | None = None):
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch in loader:
        latent = batch["latent"].to(device)
        indices = batch["indices"].to(device)
        text_feature = batch["text_feature"].to(device)
        text_mask = batch["text_mask"].to(device)
        clip_feature = torch.zeros((latent.shape[0], 512), device=device)
        if latent.shape[:2] != indices.shape[:2]:
            raise ValueError(f"latent/index time mismatch: {latent.shape} vs {indices.shape}")
        gpt_latent = reconstruct_latents_from_rvq_indices(indices, model.embedding)
        logits, _ = model(gpt_latent[:, :-1, :], indices, clip_feature, text_feature, text_mask)
        rvq_logits = logits[:, :, : indices.shape[2], :]
        if rvq_logits.shape[2] != indices.shape[2]:
            raise ValueError(f"logit depth mismatch: {rvq_logits.shape} vs {indices.shape}")
        loss = F.cross_entropy(
            rvq_logits.reshape(-1, rvq_logits.shape[-1]),
            indices.reshape(-1),
            ignore_index=513,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break
    return total_loss / max(batch_count, 1)


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", default=None)
    parser.add_argument("--init-checkpoint", default="text_generation_GPT.pth")
    parser.add_argument("--output-dir", default="stage1_artifacts/checkpoints")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    ptu.init_gpu(True, gpu_id=args.gpu)
    torch.manual_seed(args.seed)

    cfg = gpt_config()
    model = build_text_gpt_model(cfg, device=ptu.device, base_data_path="moconvq_base.data").to(ptu.device)
    _load_state_dict_flexible(model, args.init_checkpoint)

    train_ds = Stage1CacheDataset(args.train_cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    max_batches = 1 if args.smoke else None
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, ptu.device, max_batches=max_batches)
        history.append({"epoch": epoch, "loss": loss})
        if args.smoke:
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "lphym_text_gpt.pth"
    torch.save(model.state_dict(), ckpt_path)
    with (output_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(history, f)


if __name__ == "__main__":
    main()
