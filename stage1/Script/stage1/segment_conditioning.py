from __future__ import annotations

import math
from typing import Any

import torch


PROGRESS_CONDITIONING_CHOICES = ("none", "scalar", "auto")


def _as_batch_tensor(
    value: Any,
    *,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    default: float = 0.0,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), float(default), device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1 and batch_size != 1:
        tensor = tensor.expand(batch_size)
    if tensor.numel() != batch_size:
        raise ValueError(f"expected scalar or {batch_size} values, got {tensor.numel()}")
    return tensor


def build_progress_clip_feature(
    *,
    segment_idx: Any,
    num_segments: Any,
    segment_progress: Any | None = None,
    prefix_lengths: Any | None = None,
    context_size: int | None = None,
    batch_size: int | None = None,
    dim: int = 512,
    scale: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a deterministic 512-d segment/progress condition vector.

    MoConVQ's Text2Motion GPT already has a 512-d condition pathway
    (`clip_feature`).  Stage1 uses that existing pathway for segment progress
    instead of changing the transformer architecture, so pretrained checkpoints
    remain loadable.
    """

    if dim < 16:
        raise ValueError("progress feature dim must be at least 16")
    if scale < 0.0:
        raise ValueError("progress feature scale must be non-negative")
    if batch_size is None:
        for candidate in (segment_idx, num_segments, segment_progress, prefix_lengths):
            if candidate is None:
                continue
            tensor = torch.as_tensor(candidate)
            if tensor.numel() > 1:
                batch_size = int(tensor.numel())
                break
        if batch_size is None:
            batch_size = 1

    idx = _as_batch_tensor(segment_idx, batch_size=batch_size, device=device, dtype=dtype)
    count = _as_batch_tensor(num_segments, batch_size=batch_size, device=device, dtype=dtype, default=1.0)
    count = count.clamp_min(1.0)
    relative_idx = torch.where(count > 1.0, idx.clamp_min(0.0) / (count - 1.0), torch.zeros_like(idx))
    progress = (
        relative_idx
        if segment_progress is None
        else _as_batch_tensor(segment_progress, batch_size=batch_size, device=device, dtype=dtype)
    ).clamp(0.0, 1.0)
    prefix = _as_batch_tensor(prefix_lengths, batch_size=batch_size, device=device, dtype=dtype)
    if context_size is None or context_size <= 0:
        prefix_ratio = torch.zeros_like(prefix)
    else:
        prefix_ratio = (prefix / float(context_size)).clamp(0.0, 1.0)

    feature = torch.zeros((batch_size, dim), device=device, dtype=dtype)
    feature[:, 0] = progress
    feature[:, 1] = relative_idx.clamp(0.0, 1.0)
    feature[:, 2] = ((count - 1.0) / 7.0).clamp(0.0, 1.0)
    feature[:, 3] = prefix_ratio
    feature[:, 4] = torch.sin(progress * (2.0 * math.pi))
    feature[:, 5] = torch.cos(progress * (2.0 * math.pi))
    feature[:, 6] = torch.sin(relative_idx * (2.0 * math.pi))
    feature[:, 7] = torch.cos(relative_idx * (2.0 * math.pi))
    feature[:, 8] = 1.0

    # A tiny one-hot segment-id code gives the model an explicit discrete cue
    # while keeping the representation bounded for unseen longer prompts.
    one_hot_width = min(32, dim - 16)
    if one_hot_width > 0:
        safe_idx = idx.to(torch.long).clamp(min=0, max=one_hot_width - 1)
        feature.scatter_(1, safe_idx[:, None] + 16, 1.0)

    return feature * float(scale)


def resolve_progress_conditioning(mode: str, *, has_segment_metadata: bool, is_segmented: bool) -> str:
    if mode not in PROGRESS_CONDITIONING_CHOICES:
        raise ValueError(f"unknown progress conditioning mode: {mode}")
    if mode == "auto":
        return "scalar" if (has_segment_metadata or is_segmented) else "none"
    return mode


def add_progress_to_clip_feature(
    clip_feature: torch.Tensor,
    *,
    mode: str,
    segment_idx: Any,
    num_segments: Any,
    segment_progress: Any | None = None,
    prefix_lengths: Any | None = None,
    context_size: int | None = None,
    scale: float = 1.0,
    has_segment_metadata: bool = True,
    is_segmented: bool = True,
) -> torch.Tensor:
    resolved = resolve_progress_conditioning(
        mode,
        has_segment_metadata=has_segment_metadata,
        is_segmented=is_segmented,
    )
    if resolved == "none" or scale == 0.0:
        return clip_feature
    progress_feature = build_progress_clip_feature(
        segment_idx=segment_idx,
        num_segments=num_segments,
        segment_progress=segment_progress,
        prefix_lengths=prefix_lengths,
        context_size=context_size,
        batch_size=int(clip_feature.shape[0]),
        dim=int(clip_feature.shape[-1]),
        scale=scale,
        device=clip_feature.device,
        dtype=clip_feature.dtype,
    )
    return clip_feature + progress_feature
