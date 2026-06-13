from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import shutil
import zipfile


T2M_EXTRACTOR_URL = "https://drive.google.com/file/d/1FIiqtkt4F-GVWmnBgtZnv9W3cPWS-oM-/view"
KIT_EXTRACTOR_URL = "https://drive.google.com/file/d/1KNU8CsMAnxFrwopKBBkC8jEULGLPBHQp/view"
GLOVE_URL = "https://drive.google.com/file/d/1bCeS6Sh_mLVTebxIgiUHgdPrroW06mb6/view?usp=sharing"

REQUIRED_T2M_ASSETS = (
    "checkpoints/t2m/text_mot_match/model/finest.tar",
    "checkpoints/t2m/text_mot_match/opt.txt",
    "glove/our_vab_data.npy",
    "glove/our_vab_words.pkl",
)

T2M_SOURCE_HINTS = (
    "models/evaluator_wrapper.py",
    "utils/eval_trans.py",
    "options/get_eval_option.py",
)


def relative_status(root: Path, paths: Iterable[str]) -> dict[str, bool]:
    return {path: (root / path).exists() for path in paths}


def check_t2m_assets(root: Path) -> dict[str, object]:
    asset_status = relative_status(root, REQUIRED_T2M_ASSETS)
    source_status = relative_status(root, T2M_SOURCE_HINTS)
    missing_assets = [path for path, exists in asset_status.items() if not exists]
    missing_sources = [path for path, exists in source_status.items() if not exists]
    return {
        "root": str(root),
        "assets_ready": not missing_assets,
        "sources_ready": not missing_sources,
        "ready": not missing_assets and not missing_sources,
        "assets": asset_status,
        "sources": source_status,
        "missing_assets": missing_assets,
        "missing_sources": missing_sources,
    }


def _safe_extract(zip_path: Path, output_root: Path) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = output_root / member.filename
            resolved = target.resolve()
            if not str(resolved).startswith(str(output_root.resolve())):
                raise ValueError(f"refusing to extract unsafe path {member.filename!r} from {zip_path}")
            archive.extract(member, output_root)
            extracted.append(member.filename)
    return extracted


def unpack_archives(output_root: Path, t2m_zip: Path | None = None, glove_zip: Path | None = None) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    unpacked: dict[str, list[str]] = {}
    if t2m_zip is not None:
        unpacked[str(t2m_zip)] = _safe_extract(t2m_zip, output_root)
    if glove_zip is not None:
        unpacked[str(glove_zip)] = _safe_extract(glove_zip, output_root)

    nested_checkpoints = output_root / "checkpoints"
    if (output_root / "t2m").is_dir() and not (nested_checkpoints / "t2m").exists():
        nested_checkpoints.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output_root / "t2m"), str(nested_checkpoints / "t2m"))
    return {
        "output_root": str(output_root),
        "unpacked": {key: len(value) for key, value in unpacked.items()},
        "status": check_t2m_assets(output_root),
    }


def copy_evaluator_sources(source_root: Path, output_root: Path) -> dict[str, object]:
    copied: list[str] = []
    missing: list[str] = []
    for relative in T2M_SOURCE_HINTS:
        src = source_root / relative
        dst = output_root / relative
        if not src.exists():
            missing.append(relative)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(relative)
    return {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "copied_sources": copied,
        "missing_sources": missing,
        "status": check_t2m_assets(output_root),
    }


def download_commands(output_root: Path, python: str = "/home/chenjie/miniconda3/envs/moconvq/bin/python") -> list[str]:
    downloads = output_root / "downloads"
    return [
        f"mkdir -p {downloads}",
        (
            f'export http_proxy="http://127.0.0.1:7898" https_proxy="http://127.0.0.1:7898" && '
            f"{python} -m gdown --fuzzy {T2M_EXTRACTOR_URL} -O {downloads / 't2m.zip'}"
        ),
        (
            f'export http_proxy="http://127.0.0.1:7898" https_proxy="http://127.0.0.1:7898" && '
            f"{python} -m gdown --fuzzy {GLOVE_URL} -O {downloads / 'glove.zip'}"
        ),
        (
            f"{python} Script/stage1/prepare_t2m_evaluator_assets.py "
            f"--root {output_root} --t2m-zip {downloads / 't2m.zip'} --glove-zip {downloads / 'glove.zip'} --unpack"
        ),
        (
            f"{python} Script/stage1/prepare_t2m_evaluator_assets.py "
            f"--root {output_root} --source-root /tmp/T2M-GPT-stage1-inspect --copy-sources"
        ),
        (
            f"{python} Script/stage1/check_evaluation_readiness.py "
            f"--repo-root . --humanml-root /home/chenjie/cc/robotics/HumanML3D --evaluator-root {output_root}"
        ),
    ]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Evaluator asset root to check or populate.")
    parser.add_argument("--t2m-zip", default="", help="Optional downloaded t2m.zip to unpack.")
    parser.add_argument("--glove-zip", default="", help="Optional downloaded glove.zip to unpack.")
    parser.add_argument("--source-root", default="", help="Optional T2M-GPT/text-to-motion source root.")
    parser.add_argument("--unpack", action="store_true", help="Unpack provided zip files before checking.")
    parser.add_argument("--copy-sources", action="store_true", help="Copy required evaluator source files from --source-root.")
    parser.add_argument("--print-download-commands", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if args.print_download_commands:
        print("\n".join(download_commands(root)))
        return

    if args.unpack:
        payload = unpack_archives(
            root,
            t2m_zip=Path(args.t2m_zip) if args.t2m_zip else None,
            glove_zip=Path(args.glove_zip) if args.glove_zip else None,
        )
    elif args.copy_sources:
        if not args.source_root:
            raise SystemExit("--copy-sources requires --source-root")
        payload = copy_evaluator_sources(Path(args.source_root), root)
    else:
        payload = check_t2m_assets(root)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
