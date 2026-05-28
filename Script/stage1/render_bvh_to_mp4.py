from __future__ import annotations

from pathlib import Path
import argparse
import math
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Node:
    def __init__(self, name: str, parent: int | None):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3, dtype=np.float64)
        self.channels: list[str] = []


def parse_bvh(path: Path) -> tuple[list[Node], np.ndarray, float]:
    lines = path.read_text(errors="replace").splitlines()
    nodes: list[Node] = []
    stack: list[int] = []
    pending: int | None = None
    motion_idx: int | None = None
    end_count = 0

    for line_no, raw in enumerate(lines):
        line = raw.strip()
        if line == "MOTION":
            motion_idx = line_no
            break
        if line.startswith("ROOT ") or line.startswith("JOINT "):
            name = line.split(maxsplit=1)[1]
            parent = stack[-1] if stack else None
            nodes.append(Node(name, parent))
            pending = len(nodes) - 1
        elif line.startswith("End Site"):
            parent = stack[-1] if stack else None
            parent_name = nodes[parent].name if parent is not None else "root"
            nodes.append(Node(f"{parent_name}_end_{end_count}", parent))
            end_count += 1
            pending = len(nodes) - 1
        elif line == "{":
            if pending is not None:
                stack.append(pending)
                pending = None
        elif line == "}":
            if stack:
                stack.pop()
        elif line.startswith("OFFSET"):
            nodes[stack[-1]].offset = np.asarray([float(x) for x in line.split()[1:4]], dtype=np.float64)
        elif line.startswith("CHANNELS"):
            parts = line.split()
            nodes[stack[-1]].channels = parts[2:]

    if motion_idx is None:
        raise ValueError(f"BVH has no MOTION section: {path}")
    frames = int(lines[motion_idx + 1].split(":", 1)[1].strip())
    frame_time = float(lines[motion_idx + 2].split(":", 1)[1].strip())
    values = np.asarray(
        [[float(x) for x in row.split()] for row in lines[motion_idx + 3 : motion_idx + 3 + frames]],
        dtype=np.float64,
    )
    return nodes, values, frame_time


def rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    if axis == "X":
        return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "Y":
        return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if axis == "Z":
        return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"unknown rotation axis: {axis}")


def frame_positions(nodes: list[Node], values: np.ndarray) -> np.ndarray:
    positions: list[np.ndarray | None] = [None] * len(nodes)
    rotations: list[np.ndarray | None] = [None] * len(nodes)
    cursor = 0

    for node_id, node in enumerate(nodes):
        local_pos = node.offset.copy()
        local_rot = np.eye(3, dtype=np.float64)
        for channel in node.channels:
            value = values[cursor]
            cursor += 1
            axis = channel[0].upper()
            if channel.endswith("position"):
                if axis == "X":
                    local_pos[0] += value
                elif axis == "Y":
                    local_pos[1] += value
                elif axis == "Z":
                    local_pos[2] += value
            elif channel.endswith("rotation"):
                local_rot = local_rot @ rotation_matrix(axis, value)

        if node.parent is None:
            positions[node_id] = local_pos
            rotations[node_id] = local_rot
        else:
            parent_pos = positions[node.parent]
            parent_rot = rotations[node.parent]
            if parent_pos is None or parent_rot is None:
                raise ValueError("invalid BVH hierarchy order")
            positions[node_id] = parent_pos + parent_rot @ local_pos
            rotations[node_id] = parent_rot @ local_rot

    return np.stack([pos for pos in positions if pos is not None], axis=0)


def render_bvh_to_mp4(
    src: Path,
    dst: Path,
    ffmpeg: str,
    out_fps: int,
    width: int,
    height: int,
    max_video_frames: int | None,
) -> None:
    nodes, data, frame_time = parse_bvh(src)
    src_fps = 1.0 / frame_time if frame_time > 0 else 120.0
    step = max(1, int(round(src_fps / float(out_fps))))
    frame_indices = list(range(0, len(data), step))
    if max_video_frames is not None:
        frame_indices = frame_indices[:max_video_frames]

    sampled = np.stack([frame_positions(nodes, data[idx]) for idx in frame_indices], axis=0)
    root_xz = sampled[:, :1, [0, 2]]
    relative = sampled.copy()
    relative[:, :, 0] -= root_xz[:, :, 0]
    relative[:, :, 2] -= root_xz[:, :, 1]

    yaw = math.radians(-35.0)
    pitch = math.radians(12.0)
    ry = np.asarray([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    rx = np.asarray([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    view = rx @ ry
    projected = relative @ view.T
    flat_xy = projected[:, :, [0, 1]].reshape(-1, 2)
    min_xy = np.percentile(flat_xy, 1, axis=0)
    max_xy = np.percentile(flat_xy, 99, axis=0)
    span = np.maximum(max_xy - min_xy, 0.5)

    margin = 80
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    center = (min_xy + max_xy) / 2.0
    edges = [(node_id, node.parent) for node_id, node in enumerate(nodes) if node.parent is not None]
    font = ImageFont.load_default()

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-r",
        str(out_fps),
        "-i",
        "-",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("failed to open ffmpeg stdin")

    for rendered_id, bvh_frame_id in enumerate(frame_indices):
        points = relative[rendered_id] @ view.T
        xy = np.empty((len(nodes), 2), dtype=np.float64)
        xy[:, 0] = (points[:, 0] - center[0]) * scale + width / 2
        xy[:, 1] = height - ((points[:, 1] - center[1]) * scale + height / 2)

        image = Image.new("RGB", (width, height), (250, 250, 248))
        draw = ImageDraw.Draw(image)
        draw.line((40, height - 70, width - 40, height - 70), fill=(210, 210, 210), width=2)

        for node_id, parent_id in edges:
            assert parent_id is not None
            x1, y1 = xy[parent_id]
            x2, y2 = xy[node_id]
            color = (35, 70, 140) if "_end_" not in nodes[node_id].name else (90, 90, 90)
            draw.line((x1, y1, x2, y2), fill=color, width=4)
        for x, y in xy:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(170, 45, 45))

        seconds = bvh_frame_id * frame_time
        draw.text((20, 18), src.name, fill=(20, 20, 20), font=font)
        draw.text((20, 38), f"frame {bvh_frame_id + 1}/{len(data)}  time {seconds:0.2f}s", fill=(20, 20, 20), font=font)
        image.save(proc.stdin, format="PNG")

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}: {dst}")


def output_path_for(src: Path, input_root: Path, output_root: Path) -> Path:
    rel = src.relative_to(input_root)
    return output_root / rel.with_suffix(".mp4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="BVH file or directory containing BVH files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-video-frames", type=int, default=None)
    args = parser.parse_args()

    source = Path(args.input)
    output_root = Path(args.output_dir)
    if source.is_file():
        sources = [source]
        input_root = source.parent
    else:
        sources = sorted(source.rglob("*.bvh"))
        input_root = source
    if not sources:
        raise SystemExit(f"no BVH files found under {source}")

    for src in sources:
        dst = output_path_for(src, input_root=input_root, output_root=output_root)
        print(f"render {src} -> {dst}")
        render_bvh_to_mp4(
            src=src,
            dst=dst,
            ffmpeg=args.ffmpeg,
            out_fps=args.fps,
            width=args.width,
            height=args.height,
            max_video_frames=args.max_video_frames,
        )


if __name__ == "__main__":
    main()
