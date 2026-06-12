from __future__ import annotations

from pathlib import Path
import argparse

from Script.stage1.humanml3d import build_long_horizon_manifest, load_humanml3d_catalog, write_manifest_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml-root", default="../HumanML3D/HumanML3D")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--min-clips", type=int, default=2)
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    catalog = load_humanml3d_catalog(Path(args.humanml_root))
    manifest = build_long_horizon_manifest(
        catalog,
        split=args.split,
        num_sequences=args.num_sequences,
        min_clips=args.min_clips,
        max_clips=args.max_clips,
        seed=args.seed,
    )
    write_manifest_jsonl(manifest, Path(args.output))


if __name__ == "__main__":
    main()
