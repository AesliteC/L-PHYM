# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record teacher policy rollouts to MP4."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import pprint
import re

import cv2
import joblib
import numpy as np
import torch

from teacher_policy_cfg import TeacherPolicyCfg

from isaaclab.app import AppLauncher

from utils import get_player_args

parser = get_player_args(description="Records a teacher policy rollout to MP4 in Isaac Lab.")
parser.add_argument(
    "--video_folder",
    type=str,
    default=None,
    help="Directory for recorded MP4 files. Defaults to videos/<timestamp>_<motion-pkl-name>.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=0,
    help="Maximum recorded length per motion in policy steps. Use 0 to record the full reference motion.",
)
parser.add_argument("--fps", type=int, default=50, help="FPS metadata used for MP4 writing.")
parser.add_argument("--width", type=int, default=1280, help="Rendered video width in pixels.")
parser.add_argument("--height", type=int, default=720, help="Rendered video height in pixels.")
parser.add_argument("--camera_eye", type=float, nargs=3, default=None, help="Camera eye position or offset.")
parser.add_argument("--camera_target", type=float, nargs=3, default=None, help="Camera look-at target or offset.")
parser.add_argument(
    "--follow_robot_camera",
    action="store_true",
    help="Track the robot root with the recording camera so moving demos stay centered.",
)
parser.add_argument(
    "--no_stop_on_done",
    action="store_true",
    help="Keep recording after an environment reset. By default recording stops at the first done signal.",
)
parser.add_argument(
    "--antialiasing_mode",
    type=str,
    default="TAA",
    choices=["Off", "FXAA", "DLSS", "TAA", "DLAA"],
    help="RTX antialiasing mode for recording.",
)
parser.add_argument(
    "--hide_ref_markers",
    action="store_true",
    help="Hide reference motion markers from the recorded video.",
)
parser.add_argument(
    "--recording_scene",
    action="store_true",
    help="Use a cleaner recording scene with solid ground and studio lighting.",
)
parser.add_argument(
    "--robot_color",
    type=float,
    nargs=3,
    default=None,
    help="Override the robot visual material diffuse RGB color for recording.",
)
parser.add_argument("--robot_metallic", type=float, default=0.15, help="Robot material metallic value.")
parser.add_argument("--robot_roughness", type=float, default=0.35, help="Robot material roughness value.")
parser.add_argument(
    "--render_samples_per_pixel",
    type=int,
    default=2,
    help="Direct lighting samples per pixel for recording.",
)
parser.add_argument("--motion_start_idx", type=int, default=0, help="First motion index in the pkl to record.")
parser.add_argument(
    "--motion_count",
    type=int,
    default=None,
    help="Number of motions to record. Defaults to all motions from motion_start_idx.",
)
TeacherPolicyCfg.add_args_to_parser(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.camera_eye is None:
    args_cli.camera_eye = [3.0, -4.0, 1.2] if args_cli.follow_robot_camera else [3.0, -4.0, 2.2]
if args_cli.camera_target is None:
    args_cli.camera_target = [0.0, 0.0, -0.1] if args_cli.follow_robot_camera else [0.0, 0.0, 0.9]

args_cli.headless = True
args_cli.enable_cameras = True
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from vecenv_wrapper import RslRlNeuralWBCVecEnvWrapper  # noqa: E402
from utils import get_ppo_runner_and_checkpoint_path  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

from neural_wbc.core.modes import NeuralWBCModes  # noqa: E402
from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv  # noqa: E402
from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1  # noqa: E402
from neural_wbc.isaac_lab_wrapper.terrain import recording_flat_terrain  # noqa: E402


def sanitize_filename(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return (value or "motion")[:max_len]


def get_motion_dataset_keys(motion_path: str) -> list[str]:
    data = joblib.load(motion_path)
    if not isinstance(data, dict):
        raise TypeError(
            f"Reference motion file must be a dict motion dataset, got {type(data).__name__}: {motion_path}"
        )
    return [str(key) for key in data.keys()]


def get_default_video_folder(reference_motion_path: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    motion_name = sanitize_filename(Path(reference_motion_path).stem)
    return Path.cwd() / "videos" / f"{timestamp}_{motion_name}"


def normalize_render_frame(frame) -> np.ndarray:
    if isinstance(frame, list):
        if not frame:
            raise ValueError("Render returned an empty frame list")
        frame = frame[-1]

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Expected render frame with shape HxWx3 or HxWx4, got {frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.max(initial=0.0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


class Mp4Writer:
    def __init__(self, path: Path, fps: int):
        self.path = path
        self.fps = fps
        self._writer = None
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def write(self, frame) -> None:
        frame = normalize_render_frame(frame)
        height, width = frame.shape[:2]
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(f"Failed to open MP4 writer: {self.path}")

        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr_frame)
        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def update_env_cfg(env_cfg: NeuralWBCEnvCfgH1, custom_config: dict):
    for key, value in custom_config.items():
        obj = env_cfg
        attrs = key.split(".")
        try:
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            setattr(obj, attrs[-1], value)
        except AttributeError as exc:
            raise AttributeError(f"[ERROR]: {key} is not a valid configuration key.") from exc


def make_env_cfg() -> NeuralWBCEnvCfgH1:
    if args_cli.robot == "h1":
        env_cfg = NeuralWBCEnvCfgH1(mode=NeuralWBCModes.TEST)
    elif args_cli.robot == "gr1":
        raise ValueError("GR1 is not yet implemented")
    else:
        raise ValueError(f"Unsupported robot: {args_cli.robot}")

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = args_cli.env_spacing
    if args_cli.recording_scene:
        env_cfg.terrain = recording_flat_terrain.copy()
    env_cfg.terrain.env_spacing = args_cli.env_spacing
    if args_cli.reference_motion_path:
        env_cfg.reference_motion_manager.motion_path = args_cli.reference_motion_path
    if args_cli.hide_ref_markers:
        env_cfg.visualize_ref_motion = False
    if args_cli.recording_scene:
        env_cfg.recording_lighting = True
    if args_cli.robot_color is not None:
        env_cfg.robot = env_cfg.robot.copy()
        env_cfg.robot.spawn.visual_material_path = "/World/Looks/recording_robot_material"
        env_cfg.robot.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(args_cli.robot_color),
            metallic=args_cli.robot_metallic,
            roughness=args_cli.robot_roughness,
        )
    if args_cli.follow_robot_camera:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0

    update_env_cfg(
        env_cfg,
        {
            "viewer.resolution": (args_cli.width, args_cli.height),
            "viewer.eye": tuple(args_cli.camera_eye),
            "viewer.lookat": tuple(args_cli.camera_target),
            "sim.render.antialiasing_mode": args_cli.antialiasing_mode,
            "sim.render.enable_shadows": True,
            "sim.render.enable_ambient_occlusion": args_cli.recording_scene,
            "sim.render.samples_per_pixel": args_cli.render_samples_per_pixel,
        },
    )
    return env_cfg


def main():
    motion_keys = get_motion_dataset_keys(args_cli.reference_motion_path)
    if args_cli.motion_start_idx < 0 or args_cli.motion_start_idx >= len(motion_keys):
        raise ValueError(
            f"motion_start_idx={args_cli.motion_start_idx} is outside dataset with {len(motion_keys)} motions"
        )

    motion_end_idx = len(motion_keys)
    if args_cli.motion_count is not None:
        if args_cli.motion_count <= 0:
            raise ValueError("motion_count must be positive when specified")
        motion_end_idx = min(args_cli.motion_start_idx + args_cli.motion_count, len(motion_keys))

    video_folder = Path(args_cli.video_folder) if args_cli.video_folder else get_default_video_folder(args_cli.reference_motion_path)
    video_folder.mkdir(parents=True, exist_ok=True)

    env_cfg = make_env_cfg()
    teacher_policy_cfg = TeacherPolicyCfg.from_argparse_args(args_cli)

    env = NeuralWBCEnv(cfg=env_cfg, render_mode="rgb_array")
    env.metadata["render_fps"] = args_cli.fps

    wrapped_env = RslRlNeuralWBCVecEnvWrapper(env)
    ppo_runner, checkpoint_path = get_ppo_runner_and_checkpoint_path(
        teacher_policy_cfg=teacher_policy_cfg,
        wrapped_env=wrapped_env,
        device=wrapped_env.unwrapped.device,
    )
    ppo_runner.load(checkpoint_path)
    print(f"[INFO]: Loaded model checkpoint from: {checkpoint_path}")

    policy = ppo_runner.get_inference_policy(device=wrapped_env.unwrapped.device)

    run_kwargs = {
        "video_folder": str(video_folder),
        "motion_start_idx": args_cli.motion_start_idx,
        "motion_end_idx": motion_end_idx,
        "num_motions": motion_end_idx - args_cli.motion_start_idx,
        "video_length": args_cli.video_length,
        "fps": args_cli.fps,
    }
    print("[INFO] Recording teacher MP4s.")
    pprint.pprint(run_kwargs)

    try:
        for motion_idx in range(args_cli.motion_start_idx, motion_end_idx):
            if not simulation_app.is_running():
                break

            motion_key = motion_keys[motion_idx]
            output_path = video_folder / f"{motion_idx:04d}_{sanitize_filename(motion_key)}.mp4"

            print(f"[INFO] Loading motion {motion_idx + 1}/{len(motion_keys)}: {motion_key}")
            wrapped_env.unwrapped.reference_motion_manager.load_motions(random_sample=False, start_idx=motion_idx)
            obs, _ = wrapped_env.reset()

            motion_steps = int(wrapped_env.unwrapped.reference_motion_manager.get_motion_num_steps()[0].item())
            max_steps = motion_steps if args_cli.video_length <= 0 else min(args_cli.video_length, motion_steps)
            print(f"[INFO] Recording {max_steps} policy steps to: {output_path}")

            writer = Mp4Writer(output_path, fps=args_cli.fps)
            try:
                writer.write(env.render())
                for timestep in range(max_steps):
                    if not simulation_app.is_running():
                        break
                    with torch.inference_mode():
                        actions = policy(obs)
                    obs, _, _, dones, extras = wrapped_env.step(actions)
                    writer.write(env.render())

                    if not args_cli.no_stop_on_done and torch.any(dones):
                        conditions = extras.get("termination_conditions", {})
                        active_conditions = []
                        for name, value in conditions.items():
                            if torch.any(value):
                                active_conditions.append(name)
                        print(f"[INFO] Stopping {motion_key} at step {timestep + 1} before another episode starts.")
                        if active_conditions:
                            print(f"[INFO] Active termination conditions: {active_conditions}")
                        break
            finally:
                writer.close()
            print(f"[INFO] Saved {writer.frame_count} frames: {output_path}")
    finally:
        wrapped_env.close()

    print(f"[INFO] Finished recording. MP4 files are under: {video_folder.resolve()}")


if __name__ == "__main__":
    main()
    simulation_app.close()
