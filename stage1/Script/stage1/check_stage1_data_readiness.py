from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import csv
import json


CANONICAL_SPLITS = ("train", "val", "test", "train_val")
AMASS_REQUIRED_KEYS = frozenset(("poses", "trans"))
AMASS_RATE_KEYS = frozenset(("mocap_framerate", "mocap_frame_rate"))


def _read_nonempty_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_humanml_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "all.txt").exists():
        return path
    nested = path / "HumanML3D"
    if (nested / "all.txt").exists():
        return nested.resolve()
    return path


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def _missing_ids(ids: list[str], folder: Path, suffix: str, limit: int = 20) -> list[str]:
    missing: list[str] = []
    for sample_id in ids:
        if not (folder / f"{sample_id}{suffix}").exists():
            missing.append(sample_id)
            if len(missing) >= limit:
                break
    return missing


def _processed_corpus_status(humanml_root: Path) -> dict[str, object]:
    all_ids = _read_nonempty_lines(humanml_root / "all.txt")
    split_counts = {split: len(_read_nonempty_lines(humanml_root / f"{split}.txt")) for split in CANONICAL_SPLITS}
    texts = humanml_root / "texts"
    new_joints = humanml_root / "new_joints"
    new_joint_vecs = humanml_root / "new_joint_vecs"
    counts = {
        "all_txt": len(all_ids),
        "texts": _count_files(texts, "*.txt"),
        "new_joints": _count_files(new_joints, "*.npy"),
        "new_joint_vecs": _count_files(new_joint_vecs, "*.npy"),
    }
    missing = {
        "texts": _missing_ids(all_ids, texts, ".txt"),
        "new_joints": _missing_ids(all_ids, new_joints, ".npy"),
        "new_joint_vecs": _missing_ids(all_ids, new_joint_vecs, ".npy"),
    }
    required_files = {
        "all_txt": (humanml_root / "all.txt").exists(),
        "mean": (humanml_root / "Mean.npy").exists(),
        "std": (humanml_root / "Std.npy").exists(),
    }
    count_matches = all(
        counts[key] == counts["all_txt"]
        for key in ("texts", "new_joints", "new_joint_vecs")
    )
    indexed_payload_complete = bool(counts["all_txt"] and all(required_files.values()) and not any(missing.values()))
    return {
        "ready": indexed_payload_complete,
        "root": str(humanml_root),
        "counts": counts,
        "split_counts": split_counts,
        "required_files": required_files,
        "count_matches_all_txt": count_matches,
        "indexed_payload_complete": indexed_payload_complete,
        "sample_missing_ids": missing,
    }


def _read_index_rows(index_csv: Path) -> list[dict[str, str]]:
    if not index_csv.exists():
        return []
    with index_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _index_source_status(humanml_root: Path) -> dict[str, object]:
    index_csv = humanml_root.parent / "index.csv"
    rows = _read_index_rows(index_csv)
    existing = 0
    sample_missing: list[str] = []
    sample_existing: list[str] = []
    for row in rows:
        raw_source = row.get("source_path") or ""
        source_path = (humanml_root.parent / raw_source).resolve()
        if source_path.exists():
            existing += 1
            if len(sample_existing) < 10:
                sample_existing.append(str(source_path))
        elif len(sample_missing) < 10:
            sample_missing.append(str(source_path))
    return {
        "index_csv": str(index_csv),
        "exists": index_csv.exists(),
        "rows": len(rows),
        "existing_source_files": existing,
        "missing_source_files": max(len(rows) - existing, 0),
        "sample_existing_source_files": sample_existing,
        "sample_missing_source_files": sample_missing,
    }


def _looks_like_amass_motion_npz(path: Path) -> bool:
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
    except Exception:
        return False
    return AMASS_REQUIRED_KEYS.issubset(keys) and bool(keys & AMASS_RATE_KEYS)


def _scan_source_roots(source_roots: list[Path], max_npz_inspect: int) -> dict[str, object]:
    bvh_files: list[Path] = []
    npz_files: list[Path] = []
    for root in source_roots:
        if not root.exists():
            continue
        bvh_files.extend(root.rglob("*.bvh"))
        npz_files.extend(root.rglob("*.npz"))

    standard_amass: list[str] = []
    inspected = 0
    for path in sorted(npz_files):
        if inspected >= max_npz_inspect:
            break
        inspected += 1
        if _looks_like_amass_motion_npz(path):
            standard_amass.append(str(path))
            if len(standard_amass) >= 50:
                break

    return {
        "source_roots": [str(path) for path in source_roots],
        "bvh_count": len(bvh_files),
        "sample_bvh_files": [str(path) for path in sorted(bvh_files)[:20]],
        "npz_count": len(npz_files),
        "npz_inspected": inspected,
        "standard_amass_motion_npz_count": len(standard_amass),
        "sample_standard_amass_motion_npz": standard_amass[:20],
    }


def check_stage1_data_readiness(
    repo_root: Path,
    humanml_root: Path,
    source_roots: list[Path] | None = None,
    max_npz_inspect: int = 2000,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    humanml_root = resolve_humanml_root(humanml_root)
    if source_roots is None:
        source_roots = [humanml_root.parent]
    source_roots = [path.resolve() for path in source_roots]

    processed = _processed_corpus_status(humanml_root)
    index_status = _index_source_status(humanml_root)
    source_scan = _scan_source_roots(source_roots, max_npz_inspect=max_npz_inspect)

    bvh_cache_script = repo_root / "Script/stage1/build_bvh_character_gpt_cache.py"
    bvh_diagnostic_script = repo_root / "Script/stage1/diagnose_bvh_character_retarget.py"
    has_bvh = int(source_scan["bvh_count"]) > 0
    has_amass_motion = int(source_scan["standard_amass_motion_npz_count"]) > 0
    has_index_sources = int(index_status["existing_source_files"]) > 0

    native_bvh_cache_ready = has_bvh and bvh_cache_script.exists() and bvh_diagnostic_script.exists()
    source_motion_available_for_export = has_bvh or has_amass_motion or has_index_sources
    missing: list[str] = []
    if not processed["ready"]:
        missing.append("complete canonical HumanML3D processed corpus")
    if not source_motion_available_for_export:
        missing.append("HumanML3D/AMASS source motion files or BVH exports")
    if not has_bvh:
        missing.append("BVH files for MoConVQ MotionDataSet.add_bvh_with_character()")
    if not bvh_cache_script.exists():
        missing.append("BVH-to-character GPT cache builder")

    return {
        "humanml_root": str(humanml_root),
        "repo_root": str(repo_root),
        "processed_corpus": processed,
        "index_sources": index_status,
        "source_scan": source_scan,
        "tools": {
            "bvh_character_cache": {
                "path": str(bvh_cache_script),
                "exists": bvh_cache_script.exists(),
            },
            "bvh_character_diagnostic": {
                "path": str(bvh_diagnostic_script),
                "exists": bvh_diagnostic_script.exists(),
            },
        },
        "stage1_mainline_ready": native_bvh_cache_ready and bool(processed["ready"]),
        "source_motion_available_for_export": source_motion_available_for_export,
        "native_bvh_cache_ready": native_bvh_cache_ready,
        "missing": missing,
        "recommendation": _recommend_data_action(bool(processed["ready"]), native_bvh_cache_ready),
    }


def _recommend_data_action(processed_ready: bool, native_bvh_cache_ready: bool) -> str:
    if not processed_ready:
        return "Repair the canonical HumanML3D processed corpus before using it for Stage1 training."
    if not native_bvh_cache_ready:
        return (
            "Processed HumanML3D is available, but native MoConVQ BVH retarget still needs "
            "restored AMASS/HumanML3D source motion or exported BVH files."
        )
    return "Proceed with BVH-to-character cache construction and GPT fine-tuning."


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--humanml-root", default="../HumanML3D")
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help="Root to scan for BVH or standard AMASS motion NPZ files. Defaults to the HumanML3D parent.",
    )
    parser.add_argument("--max-npz-inspect", type=int, default=2000)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    humanml_root = Path(args.humanml_root)
    resolved_humanml = resolve_humanml_root(humanml_root)
    source_roots = [Path(path) for path in args.source_root] if args.source_root else [resolved_humanml.parent]
    payload = check_stage1_data_readiness(
        repo_root=Path(args.repo_root),
        humanml_root=resolved_humanml,
        source_roots=source_roots,
        max_npz_inspect=args.max_npz_inspect,
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
