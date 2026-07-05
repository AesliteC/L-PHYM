# L-PHYM

Project page: https://aeslitec.github.io/L-PHYM/

Report: [Report.pdf](Report.pdf)

L-PHYM is a two-stage course project for long-horizon language-driven humanoid
motion generation and physics-based motion tracking.

```text
Stage 1: text / HumanML3D -> MoConVQ motion tokens -> BVH motion
Stage 2: reference motion -> neural whole-body controller -> simulated / H1 tracking
```

This README focuses on how to run the two stages.  Detailed method notes,
experiment logs, and result summaries live in the stage-specific directories.

## Repository Layout

```text
stage1/   MoConVQ-based long-horizon text-to-motion generation.
stage2/   Neural WBC tracking, training, evaluation, and deployment code.
```

## Stage 1: Run Text-to-Motion Generation

Stage 1 is under `stage1/`.  Commands in this section run from the `stage1/`
directory.

### 1. Activate Environment

Use the MoConVQ environment:

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd stage1
```

Expected local resources:

```text
moconvq_base.data
text_generation_GPT.pth
../HumanML3D/HumanML3D/
```

See `stage1/README.md` and `stage1/TEXT_GPT_TRAINING.md` for the full resource
and cache requirements.

### 2. Build Long HumanML3D Sequences

```bash
python Script/stage1/synthesize_long_humanml3d.py \
  --humanml-root ../HumanML3D/HumanML3D \
  --split train \
  --num-sequences 1000 \
  --min-clips 2 \
  --max-clips 4 \
  --seed 0 \
  --candidate-pool 256 \
  --transition-max-score 0.35 \
  --blend-frames 5 \
  --caption-joiner " then " \
  --output-dir stage1_artifacts/long_humanml3d/train
```

For validation data, change `--split val`, `--seed 1`, and the output directory
to `stage1_artifacts/long_humanml3d/val`.

### 3. Build GPT Training Cache

```bash
python Script/stage1/build_real_moconvq_gpt_cache.py \
  --long-h5 stage1_artifacts/long_humanml3d/train/long_sequences.h5 \
  --manifest stage1_artifacts/long_humanml3d/train/manifest.jsonl \
  --base-data moconvq_base.data \
  --text-model t5-large \
  --window-size 50 \
  --window-stride 25 \
  --rvq-depth 4 \
  --caption-mode window \
  --window-policy clip \
  --sample-mode segment_prefix \
  --prefix-size 25 \
  --gpu 0 \
  --output stage1_artifacts/gpt_cache/train_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/train_failures.jsonl
```

Build the validation cache with the corresponding validation `long_sequences.h5`
and `manifest.jsonl`.

### 4. Fine-Tune Text2Motion GPT

Run a smoke test first:

```bash
python Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache/train_cache.pt \
  --val-cache stage1_artifacts/gpt_cache/val_cache.pt \
  --init-checkpoint text_generation_GPT.pth \
  --base-data moconvq_base.data \
  --output-dir stage1_artifacts/checkpoints/stage1_smoke \
  --epochs 1 \
  --batch-size 2 \
  --lr 1e-5 \
  --gpu 0 \
  --num-workers 0 \
  --smoke
```

Then run training:

```bash
python Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache/train_cache.pt \
  --val-cache stage1_artifacts/gpt_cache/val_cache.pt \
  --init-checkpoint text_generation_GPT.pth \
  --base-data moconvq_base.data \
  --output-dir stage1_artifacts/checkpoints/stage1_real \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --train-scope base_head \
  --depth-weights 1.0,0.7,0.4,0.2 \
  --gpu 0 \
  --seed 0 \
  --save-every 1 \
  --num-workers 4
```

### 5. Generate BVH From Text

```bash
python Script/stage1/generate_long_motion.py \
  --checkpoint stage1_artifacts/checkpoints/stage1_real/best_val.pth \
  --text "a person walks forward then turns around then crouches down" \
  --output-bvh stage1_artifacts/generated/demo.bvh \
  --base-data moconvq_base.data \
  --text-encoder t5 \
  --text-model t5-large \
  --generation-mode segmented \
  --segment-length 30 \
  --context-size 51 \
  --chunk-size 25 \
  --top-p 0.95 \
  --temperature 1.0 \
  --gpu 0 \
  --seed 0
```

### 6. Render BVH to MP4

```bash
python Script/stage1/render_bvh_to_mp4.py \
  --input stage1_artifacts/generated \
  --output-dir stage1_artifacts/generated/videos \
  --fps 30 \
  --width 960 \
  --height 720
```

### 7. Run Stage 1 Tests

```bash
python -m unittest discover -s tests -p "test_stage1*.py" -v
```

Useful Stage 1 docs:

```text
stage1/README.md
stage1/TEXT_GPT_TRAINING.md
stage1/STAGE1_README.md
stage1/STAGE1_FINAL_RESULT_SUMMARY.md
```

## Stage 2: Run Neural WBC Tracking

Stage 2 lives under `stage2/`.  It evaluates whether retargeted or generated
reference motions can be tracked by a physically simulated Unitree H1 humanoid
using the HOVER / neural WBC teacher policy.  Commands in this section run from
the `stage2/` directory:

```bash
cd stage2
export ISAACLAB_PATH=/path/to/IsaacLab4HOVER
```

Stage 2 uses Isaac Lab.  The install script checks for Isaac Lab `v2.0.0` when
the Isaac Lab checkout has git metadata.

### 1. Install Dependencies

```bash
bash run_scripts/install_deps.sh
```

This installs the local Stage 2 packages, Python requirements, and applies the
required third-party patches.

### 2. Prepare H1 Retargeted Motion Data

Training, evaluation, and video recording expect an H1 reference motion file in
`.pkl` format.  You can either pass an explicit path:

```bash
export REFERENCE_MOTION_PATH=/path/to/h1_motion_set.pkl
```

or place the file under `neural_wbc/data/data/motions/` and refer to it by name:

```bash
cp /path/to/h1_motion_set.pkl neural_wbc/data/data/motions/my_motion.pkl
export MOTION=my_motion
```

For AMASS-style data, place AMASS and SMPL resources in the paths expected by
`run_scripts/retarget_h1.sh`, then run:

```bash
bash run_scripts/retarget_h1.sh
```

To retarget only a selected motion list:

```bash
bash run_scripts/retarget_h1.sh \
  --motions-file punch.yaml \
  --refresh-filtered
```

The expected output is:

```text
third_party/human2humanoid/data/h1/amass_all.pkl
```

Use this `.pkl` path as `REFERENCE_MOTION_PATH` in the commands below.

### 3. Run Unit and End-to-End Tests

```bash
bash run_scripts/run_unit_tests.sh
bash run_scripts/run_e2e_tests.sh
```

### 4. Train Teacher Policy

For reproduction, prefer the explicit Isaac Lab command so the motion path and
training length are visible:

```bash
export REFERENCE_MOTION_PATH=/path/to/h1_motion_set.pkl

${ISAACLAB_PATH}/isaaclab.sh -p scripts/rsl_rl/train_teacher_policy.py \
  --num_envs 1024 \
  --reference_motion_path "${REFERENCE_MOTION_PATH}" \
  --robot h1 \
  --teacher_policy.max_iterations 150000 \
  --headless
```

Checkpoints are saved under `logs/teacher/<run_id>/` as
`model_<iteration>.pt`.  Reduce `--num_envs` for debugging if GPU memory is
limited.

To resume teacher training:

```bash
${ISAACLAB_PATH}/isaaclab.sh -p scripts/rsl_rl/train_teacher_policy.py \
  --num_envs 1024 \
  --reference_motion_path "${REFERENCE_MOTION_PATH}" \
  --robot h1 \
  --teacher_policy.resume \
  --teacher_policy.resume_path logs/teacher/<run_id> \
  --teacher_policy.checkpoint model_<iteration>.pt \
  --headless
```

### 5. Evaluate Teacher Policy

Evaluate the teacher checkpoint on a motion set:

```bash
mkdir -p outputs

${ISAACLAB_PATH}/isaaclab.sh -p scripts/rsl_rl/eval.py \
  --num_envs 10 \
  --reference_motion_path "${REFERENCE_MOTION_PATH}" \
  --robot h1 \
  --teacher_policy.resume_path logs/teacher/<run_id> \
  --teacher_policy.checkpoint model_<iteration>.pt \
  --metrics_path outputs/teacher_eval.json \
  --headless
```

The helper script uses the same arguments through environment variables:

```bash
REFERENCE_MOTION_PATH=/path/to/h1_motion_set.pkl \
TEACHER_RUN=logs/teacher/<run_id> \
TEACHER_CHECKPOINT=model_<iteration>.pt \
NUM_ENVS_EVAL=10 \
bash run_scripts/training/02_eval_teacher.sh \
  --metrics_path outputs/teacher_eval.json
```

The main Stage 2 diagnostic is teacher-level tracking success / completion
across motion sets.  MPJPE metrics are also reported for successful rollouts.

### 6. Record Teacher Demos

```bash
REFERENCE_MOTION_PATH=/path/to/h1_motion_set.pkl \
TEACHER_RUN=logs/teacher/<run_id> \
TEACHER_CHECKPOINT=model_<iteration>.pt \
VIDEO_FOLDER=videos/teacher_demo_1080p \
VIDEO_WIDTH=1920 \
VIDEO_HEIGHT=1080 \
VIDEO_FPS=50 \
MOTION_START_IDX=0 \
MOTION_COUNT=8 \
bash run_scripts/visualization/06_record_teacher_mp4.sh
```

Useful recording options:

```bash
STOP_ON_DONE=1            # stop when the reference ends or tracking fails
FOLLOW_ROBOT_CAMERA=1     # keep the robot centered
HIDE_REF_MARKERS=1        # cleaner qualitative demos
RECORDING_SCENE=1
```

Very short MP4 files usually indicate early termination, such as undesired
contact or excessive reference-motion distance.  Exclude those clips from
qualitative demos.

### 7. Optional Student and Visualization Scripts

Student distillation is not required for the teacher-level reproduction path,
but the scripts are kept for deployment-oriented experiments:

```bash
bash run_scripts/training/03_train_student.sh
bash run_scripts/training/04_resume_student.sh
bash run_scripts/training/05_eval_student.sh
bash run_scripts/visualization/01_play_teacher_livestream.sh
bash run_scripts/visualization/02_play_teacher_headless.sh
bash run_scripts/visualization/03_play_student_livestream.sh
bash run_scripts/visualization/04_play_teacher_gui.sh
bash run_scripts/visualization/05_export_teacher_onnx.sh
```

For a simple MuJoCo viewer:

```bash
${ISAACLAB_PATH}/isaaclab.sh -p neural_wbc/inference_env/scripts/mujoco_viewer_player.py
```

The viewer starts paused.  Press `SPACE` to start simulation or `RIGHT` to step
one frame.

### 8. Stage 2 Package Docs

```text
stage2/neural_wbc/core/README.md
stage2/neural_wbc/data/README.md
stage2/neural_wbc/mujoco_wrapper/README.md
stage2/neural_wbc/inference_env/inference_env/README.md
stage2/neural_wbc/student_policy/README.md
stage2/neural_wbc/hw_wrappers/README.md
```

## References

- MoConVQ: Unified Physics-Based Motion Control via Scalable Discrete
  Representations
- HumanML3D: 3D Human Motion-Language Dataset
- Isaac Lab
- Unitree H1
