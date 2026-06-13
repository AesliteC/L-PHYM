from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import argparse
import glob
import json
import re
import sys

import numpy as np
import torch
from scipy import linalg

if __package__ in {None, ""}:
    repo_root = str(Path(__file__).resolve().parents[2])
    if not sys.path or sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)

from Script.stage1.bvh_to_humanml3d_features import (
    APPROXIMATION_NOTE,
    convert_bvh_to_humanml3d_features,
    resolve_humanml_data_root,
)
from Script.stage1.export_humanml3d_to_bvh import select_humanml3d_sample_ids
from Script.stage1.prepare_t2m_evaluator_assets import check_t2m_assets


@dataclass(frozen=True)
class PromptSpec:
    name: str
    text: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class MotionItem:
    name: str
    model: str
    prompt: PromptSpec
    feature_path: Path
    source_bvh: Path | None = None


def collect_input_files(inputs: Iterable[str], suffix: str = ".bvh") -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob(f"*{suffix}")))
        else:
            matches = [Path(item) for item in sorted(glob.glob(raw))]
            files.extend(matches if matches else [path])
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def simple_caption_tokens(text: str) -> tuple[str, ...]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return tuple(f"{word}/OTHER" for word in words)


def _normalize_token(token: str) -> str:
    token = token.strip()
    if not token:
        return "unk/OTHER"
    return token if "/" in token else f"{token}/OTHER"


def _parse_json_list_literal(raw: str) -> list[object] | None:
    raw = raw.strip()
    if not raw.startswith("["):
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def read_prompt_specs(path: Path) -> dict[str, PromptSpec]:
    specs: dict[str, PromptSpec] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) not in {2, 3, 4}:
            raise ValueError(
                f"expected TSV with name, text, optional tokens or explicit segment metadata at {path}:{line_no}"
            )
        name = parts[0].strip()
        text = parts[1].strip()
        if not name or not text:
            raise ValueError(f"empty prompt name/text at {path}:{line_no}")

        raw_third = parts[2].strip() if len(parts) >= 3 else ""
        raw_fourth = parts[3].strip() if len(parts) >= 4 else ""
        third_json = _parse_json_list_literal(raw_third) if raw_third else None
        fourth_json = _parse_json_list_literal(raw_fourth) if raw_fourth else None

        if len(parts) == 4:
            if raw_third and third_json is None:
                raise ValueError(f"expected JSON segment list in third TSV column at {path}:{line_no}")
            if raw_fourth and fourth_json is None:
                raise ValueError(f"expected JSON segment-length list in fourth TSV column at {path}:{line_no}")

        if raw_third and third_json is None:
            tokens = tuple(_normalize_token(token) for token in parts[2].split())
        else:
            tokens = simple_caption_tokens(text)
        specs[name] = PromptSpec(name=name, text=text, tokens=tokens)
    if not specs:
        raise ValueError(f"no prompts found in {path}")
    return specs


def parse_generated_bvh_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" not in stem:
        return stem, "generated"
    prompt_name, model_name = stem.split("__", 1)
    return prompt_name, model_name


def convert_bvhs_to_features(
    bvh_files: list[Path],
    *,
    prompts: dict[str, PromptSpec],
    humanml_root: Path,
    output_dir: Path,
    target_fps: float,
    feet_threshold: float,
    example_id: str,
    allow_missing_prompts: bool = False,
    save_joints: bool = False,
) -> tuple[list[MotionItem], list[dict[str, object]]]:
    vec_dir = output_dir / "generated_new_joint_vecs"
    joints_dir = output_dir / "generated_new_joints" if save_joints else None
    items: list[MotionItem] = []
    rows: list[dict[str, object]] = []
    for bvh in bvh_files:
        prompt_name, model_name = parse_generated_bvh_name(bvh)
        prompt = prompts.get(prompt_name)
        if prompt is None:
            if not allow_missing_prompts:
                raise ValueError(f"no prompt text found for generated BVH {bvh}; expected key {prompt_name!r}")
            prompt = PromptSpec(
                name=prompt_name,
                text=prompt_name.replace("_", " "),
                tokens=simple_caption_tokens(prompt_name.replace("_", " ")),
            )
        output_vecs = vec_dir / f"{bvh.stem}.npy"
        output_joints = joints_dir / f"{bvh.stem}.npy" if joints_dir is not None else None
        summary = convert_bvh_to_humanml3d_features(
            bvh,
            humanml_data_root=humanml_root,
            output_vecs=output_vecs,
            output_joints=output_joints,
            target_fps=target_fps,
            feet_threshold=feet_threshold,
            example_id=example_id,
        )
        item = MotionItem(
            name=prompt_name,
            model=model_name,
            prompt=prompt,
            feature_path=output_vecs,
            source_bvh=bvh,
        )
        items.append(item)
        rows.append(
            {
                "prompt": prompt_name,
                "model": model_name,
                "source_bvh": str(bvh),
                "feature_path": str(output_vecs),
                "conversion": summary,
            }
        )
    return items, rows


def load_reference_items(
    *,
    humanml_root: Path,
    split: str,
    limit: int,
    seed: int,
    no_shuffle: bool = False,
) -> list[MotionItem]:
    sample_ids = select_humanml3d_sample_ids(
        humanml_root=humanml_root,
        split=split,
        limit=limit,
        seed=seed,
        shuffle=not no_shuffle,
    )
    items: list[MotionItem] = []
    for sample_id in sample_ids:
        items.append(
            MotionItem(
                name=sample_id,
                model="reference",
                prompt=PromptSpec(name=sample_id, text=sample_id, tokens=simple_caption_tokens(sample_id)),
                feature_path=humanml_root / "new_joint_vecs" / f"{sample_id}.npy",
            )
        )
    return items


def prepare_motion_array(
    feature_path: Path,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    max_motion_length: int,
    unit_length: int,
) -> tuple[np.ndarray, int]:
    features = np.load(feature_path).astype(np.float32)
    if features.ndim != 2 or features.shape[1] != 263:
        raise ValueError(f"expected HumanML3D feature shape (T, 263), got {features.shape} at {feature_path}")
    usable = min(int(features.shape[0]), int(max_motion_length))
    usable = (usable // int(unit_length)) * int(unit_length)
    if usable < int(unit_length):
        raise ValueError(f"motion is too short after unit-length alignment: {feature_path}")
    normalized = (features[:usable] - mean) / std
    padded = np.zeros((max_motion_length, features.shape[1]), dtype=np.float32)
    padded[:usable] = normalized
    return padded, usable


def prepare_text_arrays(prompt: PromptSpec, word_vectorizer, max_text_len: int) -> tuple[np.ndarray, np.ndarray, int]:
    tokens = [_normalize_token(token) for token in prompt.tokens]
    if len(tokens) < max_text_len:
        tokens = ["sos/OTHER", *tokens, "eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
    else:
        tokens = ["sos/OTHER", *tokens[:max_text_len], "eos/OTHER"]
        sent_len = len(tokens)
    word_embeddings = []
    pos_one_hots = []
    for token in tokens:
        word_vec, pos_ohot = word_vectorizer[token]
        word_embeddings.append(word_vec[None, :])
        pos_one_hots.append(pos_ohot[None, :])
    return (
        np.concatenate(word_embeddings, axis=0).astype(np.float32),
        np.concatenate(pos_one_hots, axis=0).astype(np.float32),
        int(sent_len),
    )


def calculate_activation_statistics(activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if activations.ndim != 2 or activations.shape[0] < 2:
        raise ValueError(f"expected at least two activation rows, got {activations.shape}")
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    diff = np.atleast_1d(mu1) - np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"FID covariance product has imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def euclidean_distance_matrix(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    d1 = -2.0 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    return np.sqrt(np.maximum(d1 + d2 + d3, 0.0))


def calculate_r_precision(text_embeddings: np.ndarray, motion_embeddings: np.ndarray, top_k: int = 3) -> tuple[np.ndarray, float]:
    if text_embeddings.shape != motion_embeddings.shape:
        raise ValueError(f"expected paired embedding arrays with equal shape, got {text_embeddings.shape} and {motion_embeddings.shape}")
    sample_count = text_embeddings.shape[0]
    top_k = min(int(top_k), sample_count)
    dist_mat = euclidean_distance_matrix(text_embeddings, motion_embeddings)
    matching_score = float(np.trace(dist_mat) / sample_count)
    ranked = np.argsort(dist_mat, axis=1)
    gt = np.arange(sample_count)[:, None]
    correct = ranked[:, :top_k] == gt
    cumulative = np.maximum.accumulate(correct, axis=1)
    return cumulative.mean(axis=0).astype(np.float64), matching_score


def _insert_evaluator_root(evaluator_root: Path) -> None:
    root = str(evaluator_root.resolve())
    if not sys.path or sys.path[0] != root:
        sys.path.insert(0, root)


def build_t2m_evaluator(evaluator_root: Path, device: torch.device, unit_length: int):
    _insert_evaluator_root(evaluator_root)
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt
    from utils.word_vectorizer import WordVectorizer

    opt_path = evaluator_root / "checkpoints/t2m/text_mot_match/opt.txt"
    opt = get_opt(str(opt_path), device)
    opt.dataset_name = "t2m"
    opt.checkpoints_dir = str(evaluator_root / "checkpoints")
    opt.unit_length = int(getattr(opt, "unit_length", unit_length))
    opt.device = device
    wrapper = EvaluatorModelWrapper(opt)
    word_vectorizer = WordVectorizer(str(evaluator_root / "glove"), "our_vab")
    return wrapper, word_vectorizer


def embed_motion_items(
    items: list[MotionItem],
    *,
    eval_wrapper,
    word_vectorizer,
    mean: np.ndarray,
    std: np.ndarray,
    max_motion_length: int,
    max_text_len: int,
    unit_length: int,
    include_text: bool,
) -> tuple[np.ndarray | None, np.ndarray, list[dict[str, object]]]:
    rows = []
    word_embs = []
    pos_ohots = []
    cap_lens = []
    motions = []
    m_lens = []
    for item in items:
        motion, m_len = prepare_motion_array(
            item.feature_path,
            mean=mean,
            std=std,
            max_motion_length=max_motion_length,
            unit_length=unit_length,
        )
        motions.append(motion[None, :])
        m_lens.append(m_len)
        row = {
            "name": item.name,
            "model": item.model,
            "text": item.prompt.text,
            "feature_path": str(item.feature_path),
            "source_bvh": str(item.source_bvh) if item.source_bvh is not None else "",
            "motion_length": int(m_len),
        }
        rows.append(row)
        if include_text:
            word_emb, pos_ohot, cap_len = prepare_text_arrays(item.prompt, word_vectorizer, max_text_len=max_text_len)
            word_embs.append(word_emb[None, :])
            pos_ohots.append(pos_ohot[None, :])
            cap_lens.append(cap_len)

    motion_tensor = torch.from_numpy(np.concatenate(motions, axis=0))
    m_len_tensor = torch.as_tensor(m_lens, dtype=torch.long)
    if include_text:
        order = np.argsort(cap_lens)[::-1].copy()
        word_tensor = torch.from_numpy(np.concatenate(word_embs, axis=0)[order])
        pos_tensor = torch.from_numpy(np.concatenate(pos_ohots, axis=0)[order])
        cap_len_tensor = torch.as_tensor(np.asarray(cap_lens, dtype=np.int64)[order], dtype=torch.long)
        motion_tensor = motion_tensor[order]
        m_len_tensor = m_len_tensor[order]
        rows = [rows[int(idx)] for idx in order]
        text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
            word_tensor,
            pos_tensor,
            cap_len_tensor,
            motion_tensor,
            m_len_tensor,
        )
        return text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy(), rows

    motion_embeddings = eval_wrapper.get_motion_embeddings(motion_tensor, m_len_tensor)
    return None, motion_embeddings.cpu().numpy(), rows


def summarize_model_metrics(
    generated_items: list[MotionItem],
    *,
    reference_motion_embeddings: np.ndarray,
    eval_wrapper,
    word_vectorizer,
    mean: np.ndarray,
    std: np.ndarray,
    max_motion_length: int,
    max_text_len: int,
    unit_length: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    by_model: dict[str, list[MotionItem]] = {}
    for item in generated_items:
        by_model.setdefault(item.model, []).append(item)

    ref_mu, ref_cov = calculate_activation_statistics(reference_motion_embeddings)
    metrics: dict[str, object] = {}
    all_rows: list[dict[str, object]] = []
    for model_name, items in sorted(by_model.items()):
        if len(items) < 2:
            raise ValueError(f"at least two generated samples are required for model {model_name!r}, got {len(items)}")
        text_embeddings, motion_embeddings, rows = embed_motion_items(
            items,
            eval_wrapper=eval_wrapper,
            word_vectorizer=word_vectorizer,
            mean=mean,
            std=std,
            max_motion_length=max_motion_length,
            max_text_len=max_text_len,
            unit_length=unit_length,
            include_text=True,
        )
        assert text_embeddings is not None
        gen_mu, gen_cov = calculate_activation_statistics(motion_embeddings)
        r_precision, matching_score = calculate_r_precision(text_embeddings, motion_embeddings, top_k=3)
        padded_r = [float(value) for value in r_precision]
        while len(padded_r) < 3:
            padded_r.append(None)  # type: ignore[arg-type]
        metrics[model_name] = {
            "samples": len(items),
            "fid_vs_reference": calculate_frechet_distance(ref_mu, ref_cov, gen_mu, gen_cov),
            "r_precision_top1": padded_r[0],
            "r_precision_top2": padded_r[1],
            "r_precision_top3": padded_r[2],
            "matching_score": matching_score,
            "prompt_names": [row["name"] for row in rows],
        }
        all_rows.extend(rows)
    return metrics, all_rows


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Generated BVH files, directories, or glob patterns.")
    parser.add_argument("--prompts", required=True, help="TSV: prompt_name<TAB>caption[<TAB>HumanML3D tokens].")
    parser.add_argument("--humanml-root", default="/home/chenjie/cc/robotics/HumanML3D")
    parser.add_argument("--evaluator-root", default="/tmp/stage1_t2m_evaluator_assets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--reference-split", default="test")
    parser.add_argument("--reference-limit", type=int, default=200)
    parser.add_argument("--reference-seed", type=int, default=13)
    parser.add_argument("--reference-no-shuffle", action="store_true")
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--feet-threshold", type=float, default=0.002)
    parser.add_argument("--example-id", default="000021")
    parser.add_argument("--max-motion-length", type=int, default=196)
    parser.add_argument("--max-text-len", type=int, default=20)
    parser.add_argument("--unit-length", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-missing-prompts", action="store_true")
    parser.add_argument("--save-joints", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Only report readiness and planned inputs; do not import evaluator.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    humanml_root = resolve_humanml_data_root(Path(args.humanml_root))
    evaluator_root = Path(args.evaluator_root)
    prompts = read_prompt_specs(Path(args.prompts))
    bvh_files = collect_input_files(args.inputs, suffix=".bvh")
    readiness = check_t2m_assets(evaluator_root)
    payload: dict[str, object] = {
        "paper_metric_route": "approximate_t2m_evaluator_adapter",
        "paper_metrics": ["FID", "R-precision", "matching_score"],
        "ready": bool(readiness["ready"]),
        "readiness": readiness,
        "config": {
            "humanml_root": str(humanml_root),
            "evaluator_root": str(evaluator_root),
            "output_dir": str(output_dir),
            "prompts": str(Path(args.prompts)),
            "inputs": [str(path) for path in bvh_files],
            "reference_split": args.reference_split,
            "reference_limit": int(args.reference_limit),
            "reference_seed": int(args.reference_seed),
            "max_motion_length": int(args.max_motion_length),
            "max_text_len": int(args.max_text_len),
            "unit_length": int(args.unit_length),
        },
        "caveats": [
            "Generated MoConVQ BVHs are converted through an approximate MoConVQ/base.bvh to HumanML3D 22-joint adapter.",
            APPROXIMATION_NOTE,
            "The T2M evaluator consumes at most max_motion_length frames, default 196 at 20 FPS, so long generated sequences are truncated for this metric route.",
        ],
    }

    summary_path = Path(args.summary) if args.summary else output_dir / "t2m_paper_metrics_summary.json"
    if not readiness["ready"] or args.check_only:
        prompt_names = sorted({parse_generated_bvh_name(path)[0] for path in bvh_files})
        payload["planned_prompts"] = prompt_names
        payload["planned_models"] = sorted({parse_generated_bvh_name(path)[1] for path in bvh_files})
        payload["error"] = "" if readiness["ready"] else "T2M evaluator sources/assets are incomplete; see readiness.missing_* fields."
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if not readiness["ready"] and not args.check_only:
            raise SystemExit(2)
        return

    generated_items, conversion_rows = convert_bvhs_to_features(
        bvh_files,
        prompts=prompts,
        humanml_root=humanml_root,
        output_dir=output_dir,
        target_fps=args.target_fps,
        feet_threshold=args.feet_threshold,
        example_id=args.example_id,
        allow_missing_prompts=args.allow_missing_prompts,
        save_joints=args.save_joints,
    )
    reference_items = load_reference_items(
        humanml_root=humanml_root,
        split=args.reference_split,
        limit=args.reference_limit,
        seed=args.reference_seed,
        no_shuffle=args.reference_no_shuffle,
    )
    if len(reference_items) < 2:
        raise ValueError("at least two reference motions are required for FID")
    if len(generated_items) < 2:
        raise ValueError("at least two generated motions are required for FID/R-precision")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    eval_wrapper, word_vectorizer = build_t2m_evaluator(evaluator_root, device=device, unit_length=args.unit_length)
    mean = np.load(humanml_root / "Mean.npy").astype(np.float32)
    std = np.maximum(np.load(humanml_root / "Std.npy").astype(np.float32), 1e-8)
    _ref_text, reference_motion_embeddings, reference_rows = embed_motion_items(
        reference_items,
        eval_wrapper=eval_wrapper,
        word_vectorizer=word_vectorizer,
        mean=mean,
        std=std,
        max_motion_length=args.max_motion_length,
        max_text_len=args.max_text_len,
        unit_length=args.unit_length,
        include_text=False,
    )
    model_metrics, generated_rows = summarize_model_metrics(
        generated_items,
        reference_motion_embeddings=reference_motion_embeddings,
        eval_wrapper=eval_wrapper,
        word_vectorizer=word_vectorizer,
        mean=mean,
        std=std,
        max_motion_length=args.max_motion_length,
        max_text_len=args.max_text_len,
        unit_length=args.unit_length,
    )
    payload.update(
        {
            "ready": True,
            "device": str(device),
            "conversion_rows": conversion_rows,
            "reference_rows": reference_rows,
            "generated_rows": generated_rows,
            "metrics_by_model": model_metrics,
        }
    )
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
