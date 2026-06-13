from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import json
import math
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _ensure_own_repo_root_on_path(package: str | None = __package__) -> None:
    if package not in {None, ""}:
        return
    repo_root = str(Path(__file__).resolve().parents[2])
    if not sys.path or sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)


_ensure_own_repo_root_on_path()

from Script.stage1.render_bvh_to_mp4 import frame_positions, parse_bvh, root_motion_display_positions


def select_quality_rows(
    quality_payload: dict[str, object],
    *,
    selection: str,
    limit_per_class: int,
) -> list[dict[str, object]]:
    rows = list(quality_payload.get("rows", []))  # type: ignore[arg-type]
    accepted = [row for row in rows if row.get("accepted")]
    rejected = [row for row in rows if not row.get("accepted")]
    if limit_per_class > 0:
        accepted = accepted[:limit_per_class]
        rejected = rejected[:limit_per_class]
    if selection == "accepted":
        return accepted
    if selection == "rejected":
        return rejected
    if selection == "both":
        return accepted + rejected
    if selection == "all":
        return accepted + rejected
    raise ValueError(f"unknown selection: {selection}")


def _rows_from_paths(paths: Iterable[str]) -> list[dict[str, object]]:
    return [
        {
            "path": str(path),
            "label": Path(path).stem,
            "accepted": True,
            "reject_reasons": [],
            "caption": "",
        }
        for path in paths
    ]


def _sample_indices(num_frames: int, frames_per_motion: int) -> list[int]:
    if num_frames <= 0:
        return []
    if frames_per_motion <= 1:
        return [num_frames // 2]
    return [int(round(x)) for x in np.linspace(0, num_frames - 1, frames_per_motion)]


def _view_matrix() -> np.ndarray:
    yaw = math.radians(-35.0)
    pitch = math.radians(12.0)
    ry = np.asarray([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    rx = np.asarray([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    return rx @ ry


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    width_chars: int,
    fill: tuple[int, int, int],
    font: ImageFont.ImageFont,
    line_height: int,
    max_lines: int,
) -> None:
    lines = textwrap.wrap(text, width=width_chars) or [""]
    for idx, line in enumerate(lines[:max_lines]):
        draw.text((xy[0], xy[1] + idx * line_height), line, fill=fill, font=font)


def _row_label(row: dict[str, object]) -> str:
    label = str(row.get("label") or Path(str(row.get("path", ""))).stem)
    caption = str(row.get("caption") or "")
    if row.get("accepted"):
        status = "ACCEPT"
    else:
        reasons = row.get("reject_reasons") or []
        status = "REJECT " + ",".join(str(reason) for reason in reasons)
    if caption:
        return f"{status}\n{label}\n{caption}"
    return f"{status}\n{label}"


def _draw_motion_row(
    image: Image.Image,
    *,
    row_index: int,
    row: dict[str, object],
    frames_per_motion: int,
    label_width: int,
    cell_width: int,
    cell_height: int,
    keep_root_motion: bool,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(image)
    top = row_index * cell_height
    path = Path(str(row["path"]))
    nodes, data, frame_time = parse_bvh(path)
    sample_ids = _sample_indices(len(data), frames_per_motion)
    sampled = np.stack([frame_positions(nodes, data[idx]) for idx in sample_ids], axis=0)
    relative = root_motion_display_positions(sampled, keep_root_motion=keep_root_motion)
    projected = relative @ _view_matrix().T

    flat_xy = projected[:, :, [0, 1]].reshape(-1, 2)
    min_xy = np.percentile(flat_xy, 1, axis=0)
    max_xy = np.percentile(flat_xy, 99, axis=0)
    span = np.maximum(max_xy - min_xy, 0.5)
    center = (min_xy + max_xy) / 2.0
    margin = 20
    scale = min((cell_width - 2 * margin) / span[0], (cell_height - 2 * margin) / span[1])
    edges = [(node_id, node.parent) for node_id, node in enumerate(nodes) if node.parent is not None]

    bg = (240, 248, 241) if row.get("accepted") else (250, 240, 238)
    draw.rectangle((0, top, image.width, top + cell_height), fill=bg)
    draw.line((0, top, image.width, top), fill=(210, 210, 210), width=1)
    _draw_wrapped_text(
        draw,
        (12, top + 12),
        _row_label(row),
        width_chars=max(20, label_width // 8),
        fill=(20, 20, 20),
        font=font,
        line_height=14,
        max_lines=max(1, (cell_height - 24) // 14),
    )

    for col, frame_id in enumerate(sample_ids):
        left = label_width + col * cell_width
        draw.rectangle((left, top, left + cell_width, top + cell_height), outline=(220, 220, 220))
        points = projected[col]
        xy = np.empty((len(nodes), 2), dtype=np.float64)
        xy[:, 0] = (points[:, 0] - center[0]) * scale + left + cell_width / 2
        xy[:, 1] = top + cell_height - ((points[:, 1] - center[1]) * scale + cell_height / 2)
        for node_id, parent_id in edges:
            assert parent_id is not None
            x1, y1 = xy[parent_id]
            x2, y2 = xy[node_id]
            color = (35, 70, 140) if "_end_" not in nodes[node_id].name else (90, 90, 90)
            draw.line((x1, y1, x2, y2), fill=color, width=3)
        for x, y in xy:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(170, 45, 45))
        seconds = frame_id * frame_time
        draw.text((left + 8, top + 8), f"{frame_id + 1}/{len(data)} {seconds:0.2f}s", fill=(35, 35, 35), font=font)


def make_contact_sheet(
    rows: list[dict[str, object]],
    output: Path,
    *,
    frames_per_motion: int = 6,
    label_width: int = 320,
    cell_width: int = 220,
    cell_height: int = 180,
    keep_root_motion: bool = False,
) -> None:
    if not rows:
        raise ValueError("no BVH rows selected for contact sheet")
    font = ImageFont.load_default()
    width = label_width + frames_per_motion * cell_width
    height = len(rows) * cell_height
    image = Image.new("RGB", (width, height), (250, 250, 248))
    for row_index, row in enumerate(rows):
        _draw_motion_row(
            image,
            row_index=row_index,
            row=row,
            frames_per_motion=frames_per_motion,
            label_width=label_width,
            cell_width=cell_width,
            cell_height=cell_height,
            keep_root_motion=keep_root_motion,
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bvh", nargs="*", help="BVH files to include when --quality-summary is omitted")
    parser.add_argument("--quality-summary", default="")
    parser.add_argument("--selection", choices=("accepted", "rejected", "both", "all"), default="both")
    parser.add_argument("--limit-per-class", type=int, default=10)
    parser.add_argument("--frames-per-motion", type=int, default=6)
    parser.add_argument("--label-width", type=int, default=320)
    parser.add_argument("--cell-width", type=int, default=220)
    parser.add_argument("--cell-height", type=int, default=180)
    parser.add_argument("--keep-root-motion", "--world-space", dest="keep_root_motion", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.quality_summary:
        payload = json.loads(Path(args.quality_summary).read_text(encoding="utf-8"))
        rows = select_quality_rows(payload, selection=args.selection, limit_per_class=args.limit_per_class)
    else:
        rows = _rows_from_paths(args.bvh)
    make_contact_sheet(
        rows,
        Path(args.output),
        frames_per_motion=args.frames_per_motion,
        label_width=args.label_width,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        keep_root_motion=args.keep_root_motion,
    )
    print(json.dumps({"output": args.output, "rows": len(rows), "frames_per_motion": args.frames_per_motion}, indent=2))


if __name__ == "__main__":
    main()
