# L-PHYM: Long-Horizon Language-Driven Physics-Based Motion Control

L-PHYM is a course project for long-horizon, language-driven humanoid motion generation and physics-based deployment. The project builds on MoConVQ and extends it toward multi-step natural-language instructions, longer motion sequences, and physically plausible execution in simulation.

The central problem is that current text-to-motion systems usually perform best on short, single-skill clips. When given compound instructions such as walking, turning, crouching, and continuing with another action, they often lose action order, repeat motions, or stop early. In addition, kinematic motion outputs such as BVH files are not automatically executable by a physically simulated humanoid, because they may violate balance, contact, or torque constraints.

L-PHYM addresses these issues with a two-stage framework:

```text
Natural language instruction
  -> long-horizon motion generation
  -> BVH motion output
  -> physics-consistent humanoid deployment
```

## Project Goals

The project aims to deliver:

- a long-horizon text-to-motion generation pipeline based on MoConVQ;
- synthesized multi-step text-motion training data from HumanML3D;
- a fine-tuned MoConVQ text-conditioned GPT model for longer motion generation;
- BVH generation and visual evaluation for compound prompts;
- a planned physics-based deployment stage using reinforcement learning in simulation.

## Stage 1: Long-Horizon Motion Generation

Stage 1 focuses on improving the text-to-motion generation side. It constructs longer paired motion-language samples from HumanML3D short clips, converts them into MoConVQ-compatible observations and RVQ motion tokens, and fine-tunes MoConVQ's `Text2Motion_Transformer`.

The Stage 1 pipeline is:

```text
HumanML3D short motion clips
  -> synthesized long motion-language sequences
  -> MoConVQ-compatible motion observations
  -> RVQ motion-token training cache
  -> fine-tuned MoConVQ Text2Motion Transformer
  -> BVH generation and rendering
```

The current Stage 1 implementation includes:

- HumanML3D catalog and split loading;
- transition-aware long-sequence synthesis;
- HumanML3D-to-MoConVQ motion conversion;
- MoConVQ latent and RVQ-index cache construction;
- T5 text-feature extraction;
- GPT fine-tuning;
- BVH generation and MP4 rendering scripts;
- diagnostic tests and experiment documentation.

See the `stage1` branch for the full Stage 1 implementation and running instructions.

## Stage 2: Physics-Consistent Deployment

Stage 2 will focus on deploying generated motions in a physics simulator. The planned direction is to train a reinforcement learning controller that can track generated reference motions while satisfying physical constraints such as balance, stable foot contact, and torque limits.

The planned Stage 2 components include:

- humanoid tracking of generated BVH/reference motions;
- masked error correction for dynamically infeasible motion parts;
- PPO-based policy training in simulation;
- privileged distillation for robust deployment;
- qualitative video demonstration of multi-step instruction execution.

## Repository Branches

This repository is organized by project stage:

```text
main    Project overview and high-level documentation
stage1  Stage 1 implementation: HumanML3D synthesis and MoConVQ-GPT fine-tuning
```

A future `stage2` branch may be added for the physics-consistent deployment stage before the final integration into `main`.

## Current Status

Stage 1 has an end-to-end experimental pipeline: data synthesis, MoConVQ cache construction, GPT fine-tuning, BVH generation, and rendering. The current results show that the engineering pipeline is functional, but qualitative long-horizon generation still needs improvement. In particular, future work should improve synthesized data quality, motion-window/text alignment, long-horizon generation strategy, and evaluation metrics beyond token-level loss.

Stage 2 is planned as the next major development stage.

## References

This project is based on:

- MoConVQ: Unified Physics-Based Motion Control via Scalable Discrete Representations
- HumanML3D: 3D Human Motion-Language Dataset

The project proposal follows a two-stage design: long-horizon kinematic motion synthesis first, followed by physics-consistent deployment in simulation.
