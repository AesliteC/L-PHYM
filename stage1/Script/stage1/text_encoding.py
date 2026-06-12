from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _token_to_vector(token: str, position: int, dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    digest = hashlib.sha1(f"{position}:{token}".encode("utf-8")).digest()
    for offset in range(0, 20, 4):
        word = int.from_bytes(digest[offset : offset + 4], "little", signed=False)
        index = word % dim
        sign = 1.0 if (word & 1) else -1.0
        magnitude = 1.0 + ((word >> 1) % 7) / 7.0
        vec[index] += sign * magnitude
    return vec


def encode_text_to_feature(text: str, dim: int = 1024, max_len: int = 64) -> tuple[np.ndarray, np.ndarray]:
    tokens = _TOKEN_RE.findall(text.lower())
    feature = np.zeros((max_len, dim), dtype=np.float32)
    mask = np.ones((max_len,), dtype=bool)
    if not tokens:
        return feature[None, :, :], mask[None, :]

    length = min(len(tokens), max_len)
    for position, token in enumerate(tokens[:length]):
        feature[position] = _token_to_vector(token, position, dim)
        mask[position] = False

    feature[:length] /= max(len(tokens), 1)
    return feature[None, :, :], mask[None, :]


def encode_text_batch(texts: Sequence[str], dim: int = 1024, max_len: int = 64) -> tuple[np.ndarray, np.ndarray]:
    features = []
    masks = []
    for text in texts:
        feature, mask = encode_text_to_feature(text, dim=dim, max_len=max_len)
        features.append(feature)
        masks.append(mask)
    return np.stack(features, axis=0), np.stack(masks, axis=0)
