from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from Script.stage1.text_encoding import encode_text_batch, encode_text_to_feature


@dataclass(frozen=True)
class RVQQuantizationResult:
    latent_vq: np.ndarray
    indices: np.ndarray


def lift_motion_vec_to_latent(vecs: np.ndarray, mean: np.ndarray, std: np.ndarray, latent_dim: int = 768) -> np.ndarray:
    if vecs.ndim != 2:
        raise ValueError("vecs must be 2D")
    if mean.shape != std.shape:
        raise ValueError("mean/std shape mismatch")
    normalized = (vecs - mean) / (std + 1e-8)
    if normalized.shape[1] >= latent_dim:
        return normalized[:, :latent_dim].astype(np.float32)
    repeat = latent_dim // normalized.shape[1] + 1
    tiled = np.tile(normalized, (1, repeat))[:, :latent_dim]
    return tiled.astype(np.float32)


def quantize_rvq_sequence(latent: np.ndarray, codebooks: Sequence[np.ndarray]) -> RVQQuantizationResult:
    residual = latent.astype(np.float32).copy()
    parts = []
    idxs = []
    for codebook in codebooks:
        diff = residual[:, None, :] - codebook[None, :, :]
        dist = np.sum(diff * diff, axis=-1)
        idx = np.argmin(dist, axis=-1).astype(np.int64)
        selected = codebook[idx]
        parts.append(selected)
        idxs.append(idx[:, None])
        residual = residual - selected
    return RVQQuantizationResult(latent_vq=np.sum(parts, axis=0).astype(np.float32), indices=np.concatenate(idxs, axis=1))


def extract_rvq_embeddings_from_state_dict(state_dict: dict[str, torch.Tensor]) -> list[np.ndarray]:
    embeddings = []
    for i in range(8):
        key = f"posterior.bottle_neck_list.{i}.embedding"
        if key not in state_dict:
            raise KeyError(f"missing codebook key: {key}")
        embeddings.append(state_dict[key].detach().cpu().numpy())
    return embeddings


def build_token_inputs_from_observations(
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    codebooks: Sequence[np.ndarray],
) -> RVQQuantizationResult:
    latent = lift_motion_vec_to_latent(observations, mean, std)
    return quantize_rvq_sequence(latent, codebooks[:4])


def resample_sequence(array: np.ndarray, target_length: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError("array must be 2D")
    if len(array) == 0:
        raise ValueError("cannot resample empty sequence")
    if len(array) == target_length:
        return array.astype(np.float32)
    positions = np.linspace(0, len(array) - 1, target_length)
    indices = np.clip(np.round(positions).astype(int), 0, len(array) - 1)
    return array[indices].astype(np.float32)


def build_text_feature(text: str) -> tuple[np.ndarray, np.ndarray]:
    return encode_text_to_feature(text)


def build_text_feature_batch(texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    return encode_text_batch(texts)


def load_moconvq_model(checkpoint_path: str, gpu: int = 0):
    from MoConVQCore.Model.MoConVQ import MoConVQ
    from MoConVQCore.Utils import pytorch_utils as ptu
    from MoConVQCore.Utils.misc import load_yaml
    from MoConVQCore.Env.vclode_track_env import VCLODETrackEnv

    ptu.init_gpu(True, gpu_id=gpu)
    args = load_yaml("Data/Parameters/bigdata.yml")
    env = VCLODETrackEnv(**args)
    agent = MoConVQ(323, 12, 57, 120, env, training=False, **args)
    agent.simple_load(checkpoint_path, strict=True)
    agent.eval()
    return agent


def make_text_embeddings(texts: Iterable[str], tokenizer, encoder, device):
    encoded = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=256)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = encoder(**encoded)
    return output.last_hidden_state, ~encoded["attention_mask"].bool()


class Stage1TextDataset(torch.utils.data.Dataset):
    def __init__(self, cache_path: str):
        self.cache = torch.load(cache_path, map_location="cpu")

    def __len__(self) -> int:
        return len(self.cache["indices"])

    def __getitem__(self, idx: int):
        return {
            "latent": self.cache["latents"][idx],
            "indices": self.cache["indices"][idx],
            "caption": self.cache["captions"][idx],
        }
