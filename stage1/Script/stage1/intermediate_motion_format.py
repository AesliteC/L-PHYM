from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_VERSION = "moconvq-intermediate-v1"
LATENT_DIM = 768
DYNAMIC_CONTROL_DIM = 256
RVQ_DEPTH = 4
DEFAULT_MOTION_FPS = 20
DEFAULT_CONTROL_FPS = 120


def _as_float32_array(value: np.ndarray, *, name: str, ndim: int, last_dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {arr.shape}")
    if arr.shape[-1] != last_dim:
        raise ValueError(f"{name} last dimension must be {last_dim}, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_int64_array(value: np.ndarray, *, name: str, shape_prefix: tuple[int, ...], last_dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != len(shape_prefix) + 1:
        raise ValueError(f"{name} has wrong rank, got shape {arr.shape}")
    if tuple(arr.shape[:-1]) != shape_prefix or arr.shape[-1] != last_dim:
        raise ValueError(f"{name} expected shape {shape_prefix + (last_dim,)}, got {arr.shape}")
    return arr


def reshape_sample_indices(raw_indices: np.ndarray, latent_length: int, rvq_depth: int = RVQ_DEPTH) -> np.ndarray:
    """Convert Text2Motion_Transformer.sample indices to (T, rvq_depth)."""
    arr = np.asarray(raw_indices, dtype=np.int64)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"raw_indices must be 1D, 2D, or 3D, got shape {arr.shape}")

    flat = arr.reshape(-1)
    needed = int(latent_length) * int(rvq_depth)
    if flat.shape[0] < needed:
        raise ValueError(f"raw_indices has {flat.shape[0]} values, need at least {needed}")
    return flat[:needed].reshape(int(latent_length), int(rvq_depth))


def build_metadata(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "format_version": FORMAT_VERSION,
        "container": "npz",
        "schema": "AMASS-style NPZ container with MoConVQ intermediate fields; not a SMPL/SMPL+H AMASS motion file.",
        "latent_dim": LATENT_DIM,
        "dynamic_control_dim": DYNAMIC_CONTROL_DIM,
        "rvq_depth": RVQ_DEPTH,
        "motion_fps": DEFAULT_MOTION_FPS,
        "control_fps": DEFAULT_CONTROL_FPS,
        "units": "meters, seconds, radians where applicable",
        "coordinate_note": "MoConVQ/VclSimuBackend simulation coordinates; downstream retargeting must handle robot-specific coordinates.",
    }
    if metadata:
        result.update(metadata)
    return result


def write_intermediate_npz(
    path: str | Path,
    *,
    motion_latent: np.ndarray,
    dynamic_control: np.ndarray,
    rvq_indices: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latent = _as_float32_array(motion_latent, name="motion_latent", ndim=2, last_dim=LATENT_DIM)
    dynamic = _as_float32_array(dynamic_control, name="dynamic_control", ndim=2, last_dim=DYNAMIC_CONTROL_DIM)
    indices = _as_int64_array(rvq_indices, name="rvq_indices", shape_prefix=(latent.shape[0],), last_dim=RVQ_DEPTH)

    if dynamic.shape[0] < latent.shape[0]:
        raise ValueError(
            f"dynamic_control length {dynamic.shape[0]} must be >= motion_latent length {latent.shape[0]}"
        )

    meta = build_metadata(metadata)
    meta["motion_latent_shape"] = list(latent.shape)
    meta["dynamic_control_shape"] = list(dynamic.shape)
    meta["rvq_indices_shape"] = list(indices.shape)
    meta["dynamic_steps_per_motion_token"] = float(dynamic.shape[0]) / float(latent.shape[0])

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
        motion_latent=latent,
        dynamic_control=dynamic,
        rvq_indices=indices,
    )
    return meta


def load_metadata(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError(f"{path} missing metadata_json")
        raw = str(data["metadata_json"].item())
    return json.loads(raw)


def validate_intermediate_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        for key in ("metadata_json", "motion_latent", "dynamic_control", "rvq_indices"):
            if key not in data.files:
                raise ValueError(f"{path} missing required key {key}")
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path} has unsupported format_version {metadata.get('format_version')!r}")
        latent = _as_float32_array(data["motion_latent"], name="motion_latent", ndim=2, last_dim=LATENT_DIM)
        dynamic = _as_float32_array(data["dynamic_control"], name="dynamic_control", ndim=2, last_dim=DYNAMIC_CONTROL_DIM)
        indices = _as_int64_array(
            data["rvq_indices"],
            name="rvq_indices",
            shape_prefix=(latent.shape[0],),
            last_dim=RVQ_DEPTH,
        )
    return {
        "path": str(path),
        "format_version": metadata["format_version"],
        "sample_id": metadata.get("sample_id", ""),
        "prompt": metadata.get("prompt", ""),
        "motion_latent_shape": list(latent.shape),
        "dynamic_control_shape": list(dynamic.shape),
        "rvq_indices_shape": list(indices.shape),
    }


def format_markdown() -> str:
    return f"""# MoConVQ 中间层运动 NPZ 格式

版本：`{FORMAT_VERSION}`

这个数据包用于把 baseline MoConVQ text-GPT 的中间层输出交给负责机器人重定向的同学。文件使用 AMASS 风格的 `.npz` 容器，是因为下游已经能处理 NumPy motion package；但它**不是**标准 AMASS SMPL/SMPL+H 运动文件。这里不会伪造 AMASS 的 `poses` 或 `trans` 数组。

## 必需数组

| key | dtype | shape | 含义 |
| --- | --- | --- | --- |
| `metadata_json` | string scalar | `()` | 样例和格式的 JSON 元信息。 |
| `motion_latent` | `float32` | `(T, {LATENT_DIM})` | baseline `Text2Motion_Transformer.sample()` 的直接输出，已去掉额外的 lookahead/end step。这是 4 层 RVQ codebook 向量求和后的 MoConVQ motion latent。 |
| `rvq_indices` | `int64` | `(T, {RVQ_DEPTH})` | text GPT 为每个 motion token 采样出的 4 层 RVQ codebook id。普通 token id 范围是 `0..511`；只有允许 early stop 时才可能出现模型 end-token id。 |
| `dynamic_control` | `float32` | `(T_dyn, {DYNAMIC_CONTROL_DIM})` | `agent.posterior.decoder.decode_dynamic(motion_latent)` 的输出，也就是进入 MoConVQ tracking policy 之前的控制目标。它位于 simulator rollout / BVH 导出之前。 |

## Metadata 字段

`metadata_json` 中至少包含：

| 字段 | 含义 |
| --- | --- |
| `format_version` | `{FORMAT_VERSION}` |
| `schema` | 人类可读说明：这是 MoConVQ 中间层数据，不是 SMPL/SMPL+H AMASS motion。 |
| `sample_id` | 稳定的文件名/样例 id。 |
| `prompt` | 生成该 motion 时使用的文本指令。 |
| `checkpoint` | GPT checkpoint 路径或名称，通常是 `text_generation_GPT.pth`。 |
| `base_data` | MoConVQ base checkpoint 路径或名称，通常是 `moconvq_base.data`。 |
| `text_encoder` | 导出时使用的文本编码器，通常是 `t5`。 |
| `text_model` | T5 模型路径或名称。 |
| `generation_mode` | `rolling` 或 `segmented`。 |
| `motion_fps` | MoConVQ motion-token 的名义频率，默认 `{DEFAULT_MOTION_FPS}` Hz。 |
| `control_fps` | decoder/tracking 控制信号的名义频率，默认 `{DEFAULT_CONTROL_FPS}` Hz。 |
| `dynamic_steps_per_motion_token` | `T_dyn / T`，当前 decoder 后通常是 4。 |

允许出现额外 metadata 字段；读取脚本应该忽略未知字段。

## 给下游同学的读取建议

1. 如果下游希望拿到最紧凑、最接近 GPT 输出的表示，读取 `motion_latent`。
2. 如果下游希望拿到进入 policy tracking 前的信号，读取 `dynamic_control`。
3. 不要把这些 `.npz` 直接传给要求标准 AMASS `poses/trans/betas/dmpls/gender/mocap_framerate` 的脚本。标准 AMASS 文件保存的是 SMPL-family axis-angle body pose 和 root translation；这个包保存的是 MoConVQ latent/control tensor。
4. 后续 Unitree/H1 重定向可以选择两条路线：通过 MoConVQ tracking/simulation stack 使用 `dynamic_control`，或者单独训练/优化一个从这些 latent 到机器人 reference motion 的 adapter。
"""


def write_format_markdown(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_markdown(), encoding="utf-8")
