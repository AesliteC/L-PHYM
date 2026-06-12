from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import argparse
from contextlib import contextmanager
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import MoConVQCore.Utils.pytorch_utils as ptu
from Script.stage1.segment_conditioning import (
    PROGRESS_CONDITIONING_CHOICES,
    add_progress_to_clip_feature,
)
from Script.stage1.train_text_gpt import (
    _load_state_dict_flexible,
    build_text_gpt_model,
    gpt_config,
    reconstruct_latents_from_rvq_indices,
)


class RealStage1CacheDataset(Dataset):
    def __init__(self, cache_path: str):
        self.cache = torch.load(cache_path, map_location="cpu")
        self.length = int(len(self.cache["indices"]))
        self.has_segment_metadata = all(
            key in self.cache for key in ("segment_idxs", "num_segments", "segment_progress", "prefix_lengths")
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, object]:
        window_size = int(self.cache["indices"][idx].shape[0])
        num_segments_value = self.cache.get("num_segments")
        segment_idx_value = self.cache.get("segment_idxs")
        segment_progress_value = self.cache.get("segment_progress")
        prefix_lengths_value = self.cache.get("prefix_lengths")
        target_masks_value = self.cache.get("target_masks")
        end_masks_value = self.cache.get("end_masks")
        target_mask = (
            torch.as_tensor(target_masks_value[idx], dtype=torch.bool)
            if target_masks_value is not None
            else torch.ones((window_size,), dtype=torch.bool)
        )
        end_mask = (
            torch.as_tensor(end_masks_value[idx], dtype=torch.bool)
            if end_masks_value is not None
            else torch.zeros((window_size,), dtype=torch.bool)
        )
        return {
            "latent": torch.as_tensor(self.cache["latents"][idx], dtype=torch.float32),
            "indices": torch.as_tensor(self.cache["indices"][idx], dtype=torch.long),
            "text_feature": torch.as_tensor(self.cache["text_features"][idx], dtype=torch.float32),
            "text_mask": torch.as_tensor(self.cache["text_masks"][idx], dtype=torch.bool),
            "target_mask": target_mask,
            "end_mask": end_mask,
            "segment_idx": torch.as_tensor(
                0 if segment_idx_value is None else segment_idx_value[idx],
                dtype=torch.long,
            ),
            "num_segments": torch.as_tensor(
                1 if num_segments_value is None else num_segments_value[idx],
                dtype=torch.long,
            ),
            "segment_progress": torch.as_tensor(
                0.0 if segment_progress_value is None else segment_progress_value[idx],
                dtype=torch.float32,
            ),
            "prefix_length": torch.as_tensor(
                0 if prefix_lengths_value is None else prefix_lengths_value[idx],
                dtype=torch.long,
            ),
            "has_segment_metadata": torch.as_tensor(self.has_segment_metadata, dtype=torch.bool),
            "caption": self.cache["captions"][idx],
            "sequence_id": self.cache["sequence_ids"][idx],
            "window_range": self.cache["window_ranges"][idx],
        }


def prepare_autoregressive_inputs(latents: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if latents.ndim != 3:
        raise ValueError(f"expected latents shape (B, T, C), got {latents.shape}")
    if indices.ndim != 3:
        raise ValueError(f"expected indices shape (B, T, D), got {indices.shape}")
    if latents.shape[:2] != indices.shape[:2]:
        raise ValueError(f"latent/index time mismatch: {latents.shape} vs {indices.shape}")
    if latents.shape[1] < 1:
        raise ValueError("autoregressive training needs at least one target token")
    return latents[:, :-1, :], indices


def select_rvq_logits_for_targets(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    depth = int(targets.shape[-1])
    if logits.ndim != 4:
        raise ValueError(f"expected logits shape (B, T, D+1, V), got {logits.shape}")
    if logits.shape[:2] != targets.shape[:2]:
        raise ValueError(f"logit/target time mismatch: {logits.shape} vs {targets.shape}")
    if logits.shape[2] < depth:
        raise ValueError(f"logits have only {logits.shape[2]} depth slots for {depth} targets")
    return logits[:, :, :depth, :]


def expand_target_mask(target_mask: torch.Tensor | None, targets: torch.Tensor) -> torch.Tensor | None:
    if target_mask is None:
        return None
    target_mask = target_mask.to(device=targets.device, dtype=torch.bool)
    if target_mask.ndim == targets.ndim - 1:
        target_mask = target_mask.unsqueeze(-1)
    if target_mask.shape != targets.shape:
        try:
            target_mask = target_mask.expand_as(targets)
        except RuntimeError as exc:
            raise ValueError(f"target_mask shape mismatch: {target_mask.shape} vs {targets.shape}") from exc
    return target_mask


def compute_loss_and_metrics(
    rvq_logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 513,
    depth_weights: Sequence[float] | None = None,
    teacher_logits: torch.Tensor | None = None,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    end_token_weight: float = 0.0,
    end_token_id: int = 512,
    target_mask: torch.Tensor | None = None,
    end_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    if rvq_logits.shape[:-1] != targets.shape:
        raise ValueError(f"logits/target shape mismatch: {rvq_logits.shape} vs {targets.shape}")
    valid = targets != ignore_index
    expanded_target_mask = expand_target_mask(target_mask, targets)
    if expanded_target_mask is not None:
        valid = valid & expanded_target_mask
    per_token_ce = F.cross_entropy(
        rvq_logits.reshape(-1, rvq_logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape_as(targets).to(rvq_logits.dtype)
    if depth_weights is None:
        weights = torch.ones((targets.shape[-1],), dtype=rvq_logits.dtype, device=rvq_logits.device)
    else:
        if len(depth_weights) != targets.shape[-1]:
            raise ValueError(f"expected {targets.shape[-1]} depth weights, got {len(depth_weights)}")
        weights = torch.as_tensor(depth_weights, dtype=rvq_logits.dtype, device=rvq_logits.device)
    weighted_valid = valid.to(rvq_logits.dtype) * weights.view(*([1] * (targets.ndim - 1)), -1)
    ce_loss = (per_token_ce * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
    loss = ce_loss

    kl_loss = rvq_logits.new_tensor(0.0)
    if teacher_logits is not None and kl_weight > 0.0:
        if teacher_logits.shape != rvq_logits.shape:
            raise ValueError(f"teacher logits shape mismatch: {teacher_logits.shape} vs {rvq_logits.shape}")
        if kl_temperature <= 0:
            raise ValueError("kl_temperature must be positive")
        student_log_probs = F.log_softmax(rvq_logits / kl_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits.detach() / kl_temperature, dim=-1)
        per_token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        kl_loss = (per_token_kl * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
        kl_loss = kl_loss * (kl_temperature ** 2)
        loss = loss + float(kl_weight) * kl_loss

    end_loss = rvq_logits.new_tensor(0.0)
    end_tokens = 0
    if end_token_weight > 0.0:
        padding = targets == ignore_index
        expanded_end_mask = expand_target_mask(end_mask, targets)
        if expanded_end_mask is not None:
            first_padding = padding & expanded_end_mask
        else:
            first_padding = padding & ~F.pad(padding[:, :-1, :], (0, 0, 1, 0), value=False)
            if expanded_target_mask is not None:
                supervised_valid = (targets != ignore_index) & expanded_target_mask
                follows_supervised = F.pad(supervised_valid[:, :-1, :], (0, 0, 1, 0), value=False)
                first_padding = padding & follows_supervised
        end_tokens = int(first_padding.sum().item())
        if end_tokens > 0:
            end_targets = torch.full(
                (end_tokens,),
                int(end_token_id),
                dtype=torch.long,
                device=targets.device,
            )
            end_logits = rvq_logits[first_padding]
            end_loss = F.cross_entropy(end_logits, end_targets)
            loss = loss + float(end_token_weight) * end_loss

    with torch.no_grad():
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
        "ce_loss": float(ce_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "end_loss": float(end_loss.detach().cpu()),
        "end_tokens": end_tokens,
        "token_accuracy": token_accuracy,
        "valid_tokens": valid_tokens,
        "depth_accuracy": depth_accuracy,
    }


def configure_trainable_scope(model: torch.nn.Module, train_scope: str) -> int:
    if train_scope not in {"all", "temporal_base_head", "base_head", "head"}:
        raise ValueError(f"unknown train scope: {train_scope}")
    for param in model.parameters():
        param.requires_grad = train_scope == "all"
    if train_scope in {"temporal_base_head", "base_head", "head"}:
        for param in model.trans_head.parameters():
            param.requires_grad = True
    if train_scope in {"temporal_base_head", "base_head"}:
        for param in model.trans_base.parameters():
            param.requires_grad = True
    if train_scope == "temporal_base_head":
        for param in model.trans_temporal.parameters():
            param.requires_grad = True
        for param in model.linear.parameters():
            param.requires_grad = True
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable <= 0:
        raise ValueError(f"train scope {train_scope} leaves no trainable parameters")
    return trainable


def _parse_depth_weights(value: str | None, depth: int) -> list[float] | None:
    if value is None or not value.strip():
        return None
    weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(weights) != depth:
        raise ValueError(f"expected {depth} depth weights, got {len(weights)}")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("depth weights must be non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("at least one depth weight must be positive")
    return weights


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    max_batches: int | None = None,
    depth_weights: Sequence[float] | None = None,
    teacher_model: nn.Module | None = None,
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
    end_token_weight: float = 0.0,
    progress_conditioning: str = "none",
    progress_scale: float = 1.0,
    context_size: int | None = None,
    teacher_progress_conditioning: str = "none",
) -> dict[str, object]:
    model.train(mode=train)
    if teacher_model is not None:
        teacher_model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_kl_loss = 0.0
    total_end_loss = 0.0
    total_end_tokens = 0
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
        target_mask = batch["target_mask"].to(device)
        end_mask = batch["end_mask"].to(device)
        base_clip_feature = torch.zeros((latent.shape[0], 512), dtype=torch.float32, device=device)
        clip_feature = add_progress_to_clip_feature(
            base_clip_feature,
            mode=progress_conditioning,
            segment_idx=batch["segment_idx"].to(device),
            num_segments=batch["num_segments"].to(device),
            segment_progress=batch["segment_progress"].to(device),
            prefix_lengths=batch["prefix_length"].to(device),
            context_size=context_size,
            scale=progress_scale,
            has_segment_metadata=bool(torch.as_tensor(batch["has_segment_metadata"]).any().item()),
            is_segmented=False,
        )
        teacher_clip_feature = base_clip_feature
        if teacher_progress_conditioning != "none":
            teacher_clip_feature = add_progress_to_clip_feature(
                base_clip_feature,
                mode=teacher_progress_conditioning,
                segment_idx=batch["segment_idx"].to(device),
                num_segments=batch["num_segments"].to(device),
                segment_progress=batch["segment_progress"].to(device),
                prefix_lengths=batch["prefix_length"].to(device),
                context_size=context_size,
                scale=progress_scale,
                has_segment_metadata=bool(torch.as_tensor(batch["has_segment_metadata"]).any().item()),
                is_segmented=False,
            )
        gpt_latent = reconstruct_latents_from_rvq_indices(indices, model.embedding)
        context_latent, targets = prepare_autoregressive_inputs(gpt_latent, indices)

        with torch.set_grad_enabled(train):
            logits, _ = model(context_latent, targets, clip_feature, text_feature, text_mask)
            rvq_logits = select_rvq_logits_for_targets(logits, targets)
            teacher_rvq_logits = None
            if teacher_model is not None and kl_weight > 0.0:
                with torch.no_grad():
                    teacher_logits, _ = teacher_model(
                        context_latent,
                        targets,
                        teacher_clip_feature,
                        text_feature,
                        text_mask,
                    )
                    teacher_rvq_logits = select_rvq_logits_for_targets(teacher_logits, targets)
            loss, metrics = compute_loss_and_metrics(
                rvq_logits,
                targets,
                depth_weights=depth_weights,
                teacher_logits=teacher_rvq_logits,
                kl_weight=kl_weight,
                kl_temperature=kl_temperature,
                end_token_weight=end_token_weight,
                target_mask=target_mask,
                end_mask=end_mask,
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        valid_tokens = int(metrics["valid_tokens"])
        total_loss += float(loss.detach().cpu()) * max(valid_tokens, 1)
        total_ce_loss += float(metrics["ce_loss"]) * max(valid_tokens, 1)
        total_kl_loss += float(metrics["kl_loss"]) * max(valid_tokens, 1)
        total_end_loss += float(metrics["end_loss"]) * max(int(metrics["end_tokens"]), 1)
        total_end_tokens += int(metrics["end_tokens"])
        total_valid += valid_tokens
        total_correct += float(metrics["token_accuracy"]) * max(valid_tokens, 1)
        depth_acc = metrics["depth_accuracy"]
        if depth_correct is None:
            depth_correct = [0.0 for _ in depth_acc]
            depth_valid = [0 for _ in depth_acc]
        valid = targets != 513
        expanded_mask = expand_target_mask(target_mask, targets)
        if expanded_mask is not None:
            valid = valid & expanded_mask
        pred = rvq_logits.argmax(dim=-1)
        for depth in range(targets.shape[-1]):
            count = int(valid[..., depth].sum().item())
            depth_valid[depth] += count
            depth_correct[depth] += float(((pred[..., depth] == targets[..., depth]) & valid[..., depth]).sum().item())

        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    return {
        "loss": total_loss / max(total_valid, 1),
        "ce_loss": total_ce_loss / max(total_valid, 1),
        "kl_loss": total_kl_loss / max(total_valid, 1),
        "end_loss": total_end_loss / max(total_end_tokens, 1),
        "end_tokens": total_end_tokens,
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


def validate_output_dir_for_training(output_dir: Path, append_log: bool) -> None:
    if append_log:
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}. "
            "Use a fresh --output-dir for a clean run, or pass --append-log when resuming intentionally."
        )


@contextmanager
def training_run_lock(output_dir: Path, metadata: dict[str, object] | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".train.lock"
    payload = {
        "pid": os.getpid(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    if metadata:
        payload.update(metadata)
    acquired = False
    fd = None
    try:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            try:
                existing = lock_path.read_text(encoding="utf-8").strip()
            except OSError:
                existing = "<unable to read lock file>"
            raise RuntimeError(
                f"training lock already exists: {lock_path}. "
                f"Another process may be writing this output directory. Existing lock: {existing}"
            ) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(json.dumps(payload, indent=2))
            handle.write("\n")
        acquired = True
        yield lock_path
    finally:
        if fd is not None:
            os.close(fd)
        if acquired:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", default=None)
    parser.add_argument("--init-checkpoint", default="text_generation_GPT.pth")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--output-dir", default="stage1_artifacts/checkpoints/real_stage1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-scope", choices=("all", "temporal_base_head", "base_head", "head"), default="all")
    parser.add_argument("--depth-weights", default=None)
    parser.add_argument("--baseline-kl-weight", type=float, default=0.0)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--end-token-weight", type=float, default=0.0)
    parser.add_argument("--progress-conditioning", choices=PROGRESS_CONDITIONING_CHOICES, default="auto")
    parser.add_argument("--progress-scale", type=float, default=1.0)
    parser.add_argument("--teacher-progress-conditioning", choices=PROGRESS_CONDITIONING_CHOICES, default="none")
    parser.add_argument("--context-size", type=int, default=51)
    parser.add_argument("--append-log", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    ptu.init_gpu(True, gpu_id=args.gpu)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    validate_output_dir_for_training(output_dir, append_log=args.append_log)

    with training_run_lock(
        output_dir,
        metadata={
            "script": "Script/stage1/train_real_text_gpt.py",
            "append_log": bool(args.append_log),
            "start_epoch": int(args.start_epoch),
            "epochs": int(args.epochs),
        },
    ):
        (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

        cfg = gpt_config()
        model = build_text_gpt_model(cfg, device=ptu.device, base_data_path=args.base_data).to(ptu.device)
        _load_state_dict_flexible(model, args.init_checkpoint)
        trainable_parameters = configure_trainable_scope(model, args.train_scope)
        print(f"train_scope={args.train_scope} trainable_parameters={trainable_parameters}")
        depth_weights = _parse_depth_weights(args.depth_weights, depth=4)
        teacher_model = None
        if args.baseline_kl_weight > 0.0:
            teacher_checkpoint = args.teacher_checkpoint or args.init_checkpoint
            teacher_model = build_text_gpt_model(cfg, device=ptu.device, base_data_path=args.base_data).to(ptu.device)
            _load_state_dict_flexible(teacher_model, teacher_checkpoint)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            print(
                f"baseline_kl_weight={args.baseline_kl_weight} "
                f"kl_temperature={args.kl_temperature} teacher_checkpoint={teacher_checkpoint}"
            )
        if depth_weights is not None:
            print(f"depth_weights={depth_weights}")
        if args.end_token_weight > 0.0:
            print(f"end_token_weight={args.end_token_weight}")

        train_ds = RealStage1CacheDataset(args.train_cache)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = None
        if args.val_cache:
            val_ds = RealStage1CacheDataset(args.val_cache)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        best_val_loss = float("inf")
        log_path = output_dir / "train_log.jsonl"
        max_batches = 1 if args.smoke else None

        log_mode = "a" if args.append_log else "w"
        with log_path.open(log_mode, encoding="utf-8") as log_file:
            for epoch in range(args.epochs):
                epoch_id = int(args.start_epoch) + epoch
                started = time.time()
                train_metrics = _run_epoch(
                    model,
                    train_loader,
                    optimizer,
                    ptu.device,
                    train=True,
                    max_batches=max_batches,
                    depth_weights=depth_weights,
                    teacher_model=teacher_model,
                    kl_weight=args.baseline_kl_weight,
                    kl_temperature=args.kl_temperature,
                    end_token_weight=args.end_token_weight,
                    progress_conditioning=args.progress_conditioning,
                    progress_scale=args.progress_scale,
                    context_size=args.context_size,
                    teacher_progress_conditioning=args.teacher_progress_conditioning,
                )
                val_metrics = None
                if val_loader is not None:
                    with torch.no_grad():
                        val_metrics = _run_epoch(
                            model,
                            val_loader,
                            optimizer,
                            ptu.device,
                            train=False,
                            max_batches=max_batches,
                            depth_weights=depth_weights,
                            teacher_model=teacher_model,
                            kl_weight=args.baseline_kl_weight,
                            kl_temperature=args.kl_temperature,
                            end_token_weight=args.end_token_weight,
                            progress_conditioning=args.progress_conditioning,
                            progress_scale=args.progress_scale,
                            context_size=args.context_size,
                            teacher_progress_conditioning=args.teacher_progress_conditioning,
                        )
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = float(val_metrics["loss"])
                        save_checkpoint(model, output_dir, "best_val.pth")

                if (epoch + 1) % max(args.save_every, 1) == 0:
                    save_checkpoint(model, output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                save_checkpoint(model, output_dir, "last.pth")
                row = {
                    "epoch": epoch_id,
                    "train": train_metrics,
                    "val": val_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_sec": time.time() - started,
                }
                log_file.write(json.dumps(row))
                log_file.write("\n")
                log_file.flush()
                val_text = "none" if val_metrics is None else (
                    f"{val_metrics['loss']:.4f}/acc={val_metrics['token_accuracy']:.4f}"
                )
                print(
                    f"epoch={epoch_id} "
                    f"train={train_metrics['loss']:.4f}/acc={train_metrics['token_accuracy']:.4f} "
                    f"val={val_text} elapsed={row['elapsed_sec']:.1f}s",
                    flush=True,
                )
                if args.smoke:
                    break


if __name__ == "__main__":
    main()
