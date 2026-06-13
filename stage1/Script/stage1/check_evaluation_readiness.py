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


def _exists_any(root: Path, patterns: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    if not root.exists():
        return matches
    for pattern in patterns:
        matches.extend(str(path) for path in root.rglob(pattern))
    return sorted(set(matches))


def check_evaluation_readiness(
    repo_root: Path,
    humanml_root: Path,
) -> dict[str, object]:
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
    generated_metric_script = repo_root / "Script/stage1/evaluate_bvh_metrics.py"
    comparison_script = repo_root / "Script/stage1/run_text_gpt_comparison.py"
    token_script = repo_root / "Script/stage1/diagnose_token_distribution.py"
    observation_script = repo_root / "Script/stage1/diagnose_observation_distribution.py"

    paper_ready = bool(evaluator_sources and evaluator_checkpoints)
    missing = []
    if not evaluator_sources:
        missing.append("HumanML3D text-motion evaluator source files")
    if not evaluator_checkpoints:
        missing.append("pretrained HumanML3D evaluator / motion-feature extractor checkpoints")

    return {
        "paper_metrics": list(PAPER_METRICS),
        "paper_metrics_ready": paper_ready,
        "paper_metrics_missing": missing,
        "humanml_root": str(humanml_root),
        "detected_evaluator_sources": evaluator_sources[:50],
        "detected_evaluator_checkpoints": evaluator_checkpoints[:50],
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
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    payload = check_evaluation_readiness(
        repo_root=Path(args.repo_root),
        humanml_root=Path(args.humanml_root),
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
