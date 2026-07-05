# L-PHYM: Long-Horizon Language-Driven Physics-Based Motion Control

L-PHYM is a course project for long-horizon, language-driven humanoid motion
generation and physics-based deployment.  The project connects a language-to-
motion generation stage with a neural whole-body-control stage so that compound
natural-language instructions can first be converted into reference motions and
then tracked by a physically simulated humanoid.

The central challenge is that text-to-motion models usually work best on short,
single-skill clips.  With compound prompts such as walking, turning, crouching,
and continuing into another action, they may lose action order, repeat motions,
or stop early.  Kinematic outputs such as BVH files also are not automatically
executable by a simulated humanoid because they may violate balance, contact, or
torque constraints.

L-PHYM is organized as a two-stage system:

```text
Natural language instruction
  -> Stage 1: long-horizon text-to-motion generation
  -> BVH / reference motion output
  -> Stage 2: physics-consistent whole-body tracking and deployment
```

## Repository Layout

The `main` branch is the integration branch for the project.  It currently
contains the Stage 1 implementation and is expected to receive the Stage 2 code
from `origin/stage_2_xiyuan_dev` under `stage2/`.

```text
stage1/   Long-horizon motion generation with MoConVQ and HumanML3D.
stage2/   Neural WBC tracking, simulation, evaluation, and deployment code.
```

Stage-specific README files and experiment notes live inside each stage
directory.  The root README is only the project-level map.

## Stage 1: Long-Horizon Motion Generation

Stage 1 focuses on generating longer motion sequences from multi-step language
prompts.  It builds on MoConVQ's text-conditioned motion-token GPT and uses
HumanML3D as the language-motion data source.

The current Stage 1 route is:

```text
HumanML3D short motion clips
  -> synthesized long motion-language sequences
  -> BVH export
  -> MoConVQ native character retargeting
  -> simulator observations
  -> RVQ motion-token cache with segment metadata
  -> fine-tuned MoConVQ Text2Motion Transformer
  -> explicit-segment long-text inference
  -> BVH generation, rendering, and evaluation
```

Important Stage 1 components include:

- HumanML3D catalog and split loading;
- transition-aware long-sequence synthesis;
- HumanML3D long-motion export to BVH;
- MoConVQ-native BVH-to-character retargeting;
- RVQ index / latent cache construction with segment-aligned metadata;
- T5 text-feature extraction;
- conservative fine-tuning of MoConVQ `Text2Motion_Transformer`;
- BVH generation and MP4 rendering scripts;
- engineering metrics, contact sheets, and approximate T2M evaluator metrics.

The main Stage 1 finding is that naive hand-written
HumanML3D-to-MoConVQ state conversion produced unhealthy token distributions.
The current route instead exports HumanML3D motion to BVH and uses MoConVQ's
native character-retargeting path before building the GPT cache.  This gives a
working and reproducible pipeline, with partial but not complete improvement
over the original baseline on long multi-stage prompts.

Useful Stage 1 entry points:

```text
stage1/README.md
stage1/STAGE1_README.md
stage1/STAGE1_FINAL_RESULT_SUMMARY.md
stage1/STAGE1_METHOD_RESULTS_FOR_PRESENTATION.md
stage1/TEXT_GPT_TRAINING.md
```

## Stage 2: Physics-Consistent Whole-Body Control

Stage 2 focuses on tracking generated reference motions with a physically
simulated humanoid.  Its code is developed in `origin/stage_2_xiyuan_dev` and is
intended to be synchronized into `main` under `stage2/`.

Stage 2 is built around a neural whole-body-control stack for the Unitree H1
humanoid.  It includes reusable core data structures, Isaac Lab and MuJoCo
simulation wrappers, policy training code, evaluation tools, and deployment
interfaces.

The Stage 2 pipeline is:

```text
Reference motion / BVH-derived motion
  -> motion dataset and reference-motion manager
  -> Isaac Lab teacher-policy training
  -> student policy distillation
  -> MuJoCo sim-to-sim validation
  -> inference environment
  -> optional Unitree H1 deployment wrapper
```

Important Stage 2 components include:

- `neural_wbc/core`: body-state data structures, reference-motion management,
  observation utilities, termination logic, environment wrapper, and evaluator;
- `neural_wbc/isaac_lab_wrapper`: Isaac Lab environment, observations, rewards,
  terrain, events, visualization, and control utilities;
- `neural_wbc/student_policy`: student policy, storage, teacher-policy wrapper,
  and training utilities;
- `neural_wbc/mujoco_wrapper`: MuJoCo robot/simulator wrappers for sim-to-sim
  validation and visualization;
- `neural_wbc/inference_env`: high-level inference, evaluation, and deployment
  player scripts;
- `neural_wbc/hw_wrappers`: Unitree H1 hardware wrapper for real-robot
  deployment experiments;
- `run_scripts/`: install, retargeting, unit-test, end-to-end-test, training,
  evaluation, visualization, ONNX export, and MP4 recording scripts;
- `third_party/`: bundled or patched dependencies including Human2Humanoid,
  RSL-RL, and a modified MuJoCo viewer.

Stage 2 uses reinforcement learning and policy distillation to make generated
or dataset reference motions trackable under physical constraints.  Its metrics
include tracking success rate, global/local/procrustes-aligned MPJPE, velocity
and acceleration errors, root-orientation errors, root-velocity error, and
root-height error.

## References

This project builds on:

- MoConVQ: Unified Physics-Based Motion Control via Scalable Discrete
  Representations
- HumanML3D: 3D Human Motion-Language Dataset
- Neural whole-body control workflows for humanoid motion tracking
- Unitree H1 simulation and deployment tooling
