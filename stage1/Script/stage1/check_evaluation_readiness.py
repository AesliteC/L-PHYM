from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json


PAPER_METRICS = (
    "FID on HumanML3D/SMPL motion features",
    "R-precision text-motion retrieval",
)

ENGINEERING_METRICS = (
    "BVH duration and early-stop rate",
    "root path length and displacement",
    "pose velocity and pose variance",
    "lagged centered-pose cosine / repeat fraction",
    "RVQ token distribution and repetition diagnostics",
    "MoConVQ observation z-score diagnostics",
)


T2M_EVALUATOR_SOURCE_HINTS = (
    "models/evaluator_wrapper.py",
    "utils/eval_trans.py",
    "options/get_eval_option.py",
)

T2M_REQUIRED_ASSETS = (
    "checkpoints/t2m/text_mot_match/model/finest.tar",
    "checkpoints/t2m/text_mot_match/opt.txt",
    "glove/our_vab_data.npy",
    "glove/our_vab_words.pkl",
)


def _exists_any(root: Path, patterns: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    if not root.exists():
        return matches
    for pattern in patterns:
        matches.extend(str(path) for path in root.rglob(pattern))
    return sorted(set(matches))


def _existing_relative_paths(root: Path, relative_paths: tuple[str, ...]) -> list[str]:
    return [path for path in relative_paths if (root / path).exists()]


def _missing_relative_paths(root: Path, relative_paths: tuple[str, ...]) -> list[str]:
    return [path for path in relative_paths if not (root / path).exists()]


def check_evaluation_readiness(
    repo_root: Path,
    humanml_root: Path,
    evaluator_root: Path | None = None,
) -> dict[str, object]:
    evaluator_root = evaluator_root or repo_root
    evaluator_sources = _exists_any(
        humanml_root,
        (
            "*eval*.py",
            "*evaluator*.py",
            "*Evaluator*.py",
        ),
    )
    evaluator_checkpoints = _exists_any(
        humanml_root,
        (
            "*.pth",
            "*.pt",
            "*.tar",
            "*.ckpt",
        ),
    )
    detected_t2m_sources = _existing_relative_paths(evaluator_root, T2M_EVALUATOR_SOURCE_HINTS)
    missing_t2m_sources = _missing_relative_paths(evaluator_root, T2M_EVALUATOR_SOURCE_HINTS)
    detected_t2m_assets = _existing_relative_paths(evaluator_root, T2M_REQUIRED_ASSETS)
    missing_t2m_assets = _missing_relative_paths(evaluator_root, T2M_REQUIRED_ASSETS)
    generated_metric_script = repo_root / "Script/stage1/evaluate_bvh_metrics.py"
    comparison_script = repo_root / "Script/stage1/run_text_gpt_comparison.py"
    token_script = repo_root / "Script/stage1/diagnose_token_distribution.py"
    observation_script = repo_root / "Script/stage1/diagnose_observation_distribution.py"

    t2m_ready = not missing_t2m_sources and not missing_t2m_assets
    paper_ready = bool(t2m_ready or (evaluator_sources and evaluator_checkpoints))
    missing = []
    if not evaluator_sources and missing_t2m_sources:
        missing.append("HumanML3D text-motion evaluator source files")
    if not evaluator_checkpoints and missing_t2m_assets:
        missing.append("pretrained HumanML3D evaluator / motion-feature extractor checkpoints")

    return {
        "paper_metrics": list(PAPER_METRICS),
        "paper_metrics_ready": paper_ready,
        "paper_metrics_missing": missing,
        "humanml_root": str(humanml_root),
        "evaluator_root": str(evaluator_root),
        "detected_evaluator_sources": evaluator_sources[:50],
        "detected_evaluator_checkpoints": evaluator_checkpoints[:50],
        "t2m_evaluator": {
            "source": "T2M-GPT/text-to-motion compatible evaluator wrapper",
            "source_hint": "T2M-GPT uses the same extractors provided by EricGuo5513/text-to-motion.",
            "expected_source_files": list(T2M_EVALUATOR_SOURCE_HINTS),
            "expected_assets": list(T2M_REQUIRED_ASSETS),
            "detected_source_files": detected_t2m_sources,
            "missing_source_files": missing_t2m_sources,
            "detected_assets": detected_t2m_assets,
            "missing_assets": missing_t2m_assets,
            "ready": t2m_ready,
            "remaining_adapter_gap": (
                "Even with evaluator assets, generated MoConVQ BVH/character motion must be converted "
                "to HumanML3D 263-d motion features before FID/R-precision are comparable."
            ),
        },
        "engineering_metrics": list(ENGINEERING_METRICS),
        "engineering_tools": {
            "bvh_metrics": {
                "path": str(generated_metric_script),
                "exists": generated_metric_script.exists(),
            },
            "baseline_vs_finetuned_comparison": {
                "path": str(comparison_script),
                "exists": comparison_script.exists(),
            },
            "token_distribution": {
                "path": str(token_script),
                "exists": token_script.exists(),
            },
            "observation_distribution": {
                "path": str(observation_script),
                "exists": observation_script.exists(),
            },
        },
        "recommendation": (
            "Use engineering diagnostics only as intermediate checks; do not claim paper-level "
            "improvement over baseline until FID/R-precision evaluator assets are available."
            if not paper_ready
            else "Paper-level FID/R-precision assets appear available; integrate them before final comparison."
        ),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--humanml-root", default="../HumanML3D")
    parser.add_argument("--evaluator-root", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    payload = check_evaluation_readiness(
        repo_root=Path(args.repo_root),
        humanml_root=Path(args.humanml_root),
        evaluator_root=Path(args.evaluator_root) if args.evaluator_root else None,
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
