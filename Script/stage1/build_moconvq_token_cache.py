from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import torch

from Script.stage1.humanml3d import load_humanml3d_catalog
from Script.stage1.motion_bridge import (
    build_text_feature,
    extract_rvq_embeddings_from_state_dict,
    quantize_rvq_sequence,
    resample_sequence,
    lift_motion_vec_to_latent,
)


def load_manifest(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_cache(manifest_rows, catalog_root: Path, base_data_path: Path, target_length: int = 50):
    catalog = load_humanml3d_catalog(catalog_root)
    state_dict = torch.load(base_data_path, map_location="cpu")
    codebooks = extract_rvq_embeddings_from_state_dict(state_dict)
    codebooks = [np.concatenate([codebook, np.zeros((2, codebook.shape[1]), dtype=codebook.dtype)], axis=0) for codebook in codebooks[:4]]

    latents = []
    indices = []
    text_features = []
    text_masks = []
    captions = []
    sequence_ids = []
    sample_ids_all = []

    for row in manifest_rows:
        sample_ids = row["sample_ids"]
        vecs = [np.load(catalog.by_id[sample_id].vecs_path) for sample_id in sample_ids]
        concatenated = np.concatenate(vecs, axis=0)
        resampled = resample_sequence(concatenated, target_length)
        latent = lift_motion_vec_to_latent(resampled, np.load(catalog.root / "Mean.npy"), np.load(catalog.root / "Std.npy"))
        quantized = quantize_rvq_sequence(latent, codebooks)
        text_feature, text_mask = build_text_feature(str(row["caption"]))

        latents.append(quantized.latent_vq.astype(np.float32))
        indices.append(quantized.indices.astype(np.int64))
        text_features.append(text_feature.astype(np.float32))
        text_masks.append(text_mask.astype(bool))
        captions.append(str(row["caption"]))
        sequence_ids.append(str(row["sequence_id"]))
        sample_ids_all.append(sample_ids)

    return {
        "latents": torch.from_numpy(np.stack(latents, axis=0)),
        "indices": torch.from_numpy(np.stack(indices, axis=0)),
        "text_features": torch.from_numpy(np.stack(text_features, axis=0)).squeeze(1),
        "text_masks": torch.from_numpy(np.stack(text_masks, axis=0)).squeeze(1),
        "captions": captions,
        "sequence_ids": sequence_ids,
        "sample_ids": sample_ids_all,
        "target_length": target_length,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--catalog-root", default="../HumanML3D/HumanML3D")
    parser.add_argument("--base-data", default="moconvq_base.data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-length", type=int, default=50)
    args = parser.parse_args()

    manifest_rows = load_manifest(Path(args.manifest))
    cache = build_cache(
        manifest_rows,
        catalog_root=Path(args.catalog_root),
        base_data_path=Path(args.base_data),
        target_length=args.target_length,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)


if __name__ == "__main__":
    main()
