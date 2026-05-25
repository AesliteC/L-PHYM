from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import MoConVQCore.Utils.pytorch_utils as ptu
from Script.stage1.train_text_gpt import _load_state_dict_flexible, build_text_gpt_model, gpt_config


class RealStage1CacheDataset(Dataset):
    def __init__(self, cache_path: str):
        self.cache = torch.load(cache_path, map_location="cpu")

    def __len__(self) -> int:
        return int(len(self.cache["indices"]))

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {
            "latent": torch.as_tensor(self.cache["latents"][idx], dtype=torch.float32),
            "indices": torch.as_tensor(self.cache["indices"][idx], dtype=torch.long),
            "text_feature": torch.as_tensor(self.cache["text_features"][idx], dtype=torch.float32),
            "text_mask": torch.as_tensor(self.cache["text_masks"][idx], dtype=torch.bool),
            "caption": self.cache["captions"][idx],
            "sequence_id": self.cache["sequence_ids"][idx],
            "window_range": self.cache["window_ranges"][idx],
        }


def compute_loss_and_metrics(
    rvq_logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 513,
) -> tuple[torch.Tensor, dict[str, object]]:
    if rvq_logits.shape[:-1] != targets.shape:
        raise ValueError(f"logits/target shape mismatch: {rvq_logits.shape} vs {targets.shape}")
    loss = F.cross_entropy(
        rvq_logits.reshape(-1, rvq_logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
    with torch.no_grad():
        valid = targets != ignore_index
        valid_tokens = int(valid.sum().item())
        pred = rvq_logits.argmax(dim=-1)
        correct = (pred == targets) & valid
        token_accuracy = float(correct.sum().item() / max(valid_tokens, 1))
        depth_accuracy = []
        for depth in range(targets.shape[-1]):
            depth_valid = valid[..., depth]
            depth_correct = correct[..., depth]
            depth_accuracy.append(float(depth_correct.sum().item() / max(int(depth_valid.sum().item()), 1)))
    return loss, {
        "token_accuracy": token_accuracy,
        "valid_tokens": valid_tokens,
        "depth_accuracy": depth_accuracy,
    }


def _run_epoch(model, loader, optimizer, device, train: bool, max_batches: int | None = None) -> dict[str, object]:
    model.train(mode=train)
    total_loss = 0.0
    total_valid = 0
    total_correct = 0.0
    depth_correct = None
    depth_valid = None
    batch_count = 0

    for batch in loader:
        latent = batch["latent"].to(device)
        indices = batch["indices"].to(device)
        text_feature = batch["text_feature"].to(device)
        text_mask = batch["text_mask"].to(device)
        clip_feature = torch.zeros((latent.shape[0], 512), dtype=torch.float32, device=device)

        with torch.set_grad_enabled(train):
            logits, _ = model(latent, indices, clip_feature, text_feature, text_mask)
            rvq_logits = logits[:, :, 1:, :]
            loss, metrics = compute_loss_and_metrics(rvq_logits, indices)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        valid_tokens = int(metrics["valid_tokens"])
        total_loss += float(loss.detach().cpu()) * max(valid_tokens, 1)
        total_valid += valid_tokens
        total_correct += float(metrics["token_accuracy"]) * max(valid_tokens, 1)
        depth_acc = metrics["depth_accuracy"]
        if depth_correct is None:
            depth_correct = [0.0 for _ in depth_acc]
            depth_valid = [0 for _ in depth_acc]
        valid = indices != 513
        pred = rvq_logits.argmax(dim=-1)
        for depth in range(indices.shape[-1]):
            count = int(valid[..., depth].sum().item())
            depth_valid[depth] += count
            depth_correct[depth] += float(((pred[..., depth] == indices[..., depth]) & valid[..., depth]).sum().item())

        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    return {
        "loss": total_loss / max(total_valid, 1),
        "token_accuracy": total_correct / max(total_valid, 1),
        "valid_tokens": total_valid,
        "depth_accuracy": [
            depth_correct[i] / max(depth_valid[i], 1) for i in range(len(depth_correct or []))
        ],
        "batches": batch_count,
    }


def save_checkpoint(model, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    torch.save(model.state_dict(), path)
    return path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", default=None)
    parser.add_argument("--init-checkpoint", default="text_generation_GPT.pth")
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--output-dir", default="stage1_artifacts/checkpoints/real_stage1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    ptu.init_gpu(True, gpu_id=args.gpu)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    cfg = gpt_config()
    model = build_text_gpt_model(cfg, device=ptu.device, base_data_path=args.base_data).to(ptu.device)
    _load_state_dict_flexible(model, args.init_checkpoint)

    train_ds = RealStage1CacheDataset(args.train_cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = None
    if args.val_cache:
        val_ds = RealStage1CacheDataset(args.val_cache)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")
    log_path = output_dir / "train_log.jsonl"
    max_batches = 1 if args.smoke else None

    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(args.epochs):
            started = time.time()
            train_metrics = _run_epoch(model, train_loader, optimizer, ptu.device, train=True, max_batches=max_batches)
            val_metrics = None
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = _run_epoch(model, val_loader, optimizer, ptu.device, train=False, max_batches=max_batches)
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = float(val_metrics["loss"])
                    save_checkpoint(model, output_dir, "best_val.pth")

            if (epoch + 1) % max(args.save_every, 1) == 0:
                save_checkpoint(model, output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(model, output_dir, "last.pth")
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_sec": time.time() - started,
            }
            log_file.write(json.dumps(row))
            log_file.write("\n")
            log_file.flush()
            if args.smoke:
                break


if __name__ == "__main__":
    main()
