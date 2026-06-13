# Stage1 Experiment Log

This file records Stage1 checks and experiment outcomes.  It is intentionally
separate from the user-facing README so that failed or diagnostic runs do not get
mixed with installation and usage instructions.

## 2026-06-12: Top-p generation path and current Stage1 audit

### Purpose

The immediate goal was to stop treating the previous greedy/fixed-top-k videos as
the current conclusion, add configurable nucleus sampling for inference, and
audit the current HumanML3D synthesis, MoConVQ cache, and GPT fine-tuning path.

### Code changes

- Added nucleus sampling support to `Text2Motion_Transformer.sample()`:
  - `top_p`
  - `top_k`
  - `temperature`
- Kept backward compatibility:
  - Existing calls without these arguments still work.
  - Greedy mode still uses argmax.
- Updated Stage1 generation wrappers to expose and pass the new sampling args:
  - `Script/stage1/generate_long_motion.py`
  - `Script/stage1/export_baseline_intermediate.py`
- Added tests for:
  - top-p/top-k filtering behavior;
  - rolling generation passing sampling args through to `model.sample()`.

### Verification

Commands run:

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd /home/chenjie/cc/robotics/MoConVQ

python -m py_compile \
  MoConVQCore/Model/cross_trans_ori_fixsum.py \
  Script/stage1/generate_long_motion.py \
  Script/stage1/export_baseline_intermediate.py \
  tests/test_stage1_gpt.py \
  tests/test_stage1_real_generate.py

python -m unittest \
  tests.test_stage1_gpt \
  tests.test_stage1_real_generate \
  -v

python -m unittest \
  tests.test_stage1_real_synthesis \
  tests.test_stage1_real_cache \
  tests.test_stage1_real_train \
  tests.test_stage1_gpt \
  tests.test_stage1_real_generate \
  -v
```

Results:

- `py_compile`: passed.
- GPT and generation tests: 15 tests passed.
- Full Stage1 synthesis/cache/train/generation subset: 40 tests passed.

One attempted test run in the base environment failed because `torch` was not
installed there.  The same tests passed after activating the `moconvq`
environment.

### Current data and cache state

Current fixed synthesis artifacts:

- Train synthesis:
  - path: `stage1_artifacts/long_humanml3d_fixed/train`
  - sequences: 1000
  - transitions: 1945
  - average clips per sequence: 2.945
  - average frames per synthesized sequence: 416.593
  - forced transitions: 0
  - duplicate sequences: 0
  - failed/retried sequences: 3
  - invalid clips filtered: 4
- Val synthesis:
  - path: `stage1_artifacts/long_humanml3d_fixed/val`
  - sequences: 200
  - transitions: 398
  - average clips per sequence: 2.99
  - average frames per synthesized sequence: 410.2
  - forced transitions: 0
  - duplicate sequences: 0
  - failed/retried sequences: 1

Current fixed cache artifacts:

- Train cache:
  - path: `stage1_artifacts/gpt_cache_fixed/train_cache.pt`
  - `latents`: `(2958, 50, 768)`
  - `indices`: `(2958, 50, 4)`
  - `text_features`: `(2958, 256, 1024)`
  - `text_masks`: `(2958, 256)`
  - unique sequences: 1000
  - unique captions: 2571
  - valid RVQ target tokens: 417512
  - padding tokens: 174088
- Val cache:
  - path: `stage1_artifacts/gpt_cache_fixed/val_cache.pt`
  - `latents`: `(598, 50, 768)`
  - `indices`: `(598, 50, 4)`
  - `text_features`: `(598, 256, 1024)`
  - `text_masks`: `(598, 256)`
  - unique sequences: 200
  - unique captions: 413
  - valid RVQ target tokens: 81716
  - padding tokens: 37884

Cache configuration:

- `window_size=50`
- `window_stride=25`
- `rvq_depth=4`
- `caption_mode=window`
- `window_policy=clip`
- `forced_transition_margin=2`
- `max_text_length=256`

Interpretation:

- The current cache already avoids training most windows across synthetic clip
  boundaries by defaulting to clip-aligned windows.
- Window-level captions are used, so the model is not always trained with the
  full long prompt for every short motion window.
- This is better than the first long-prompt cache, but it still depends on the
  quality of the HumanML3D-to-MoConVQ retarget path.

### Current training run status

Most relevant current checkpoint:

- run id: `kl_depth_stage1_20260529_215046`
- checkpoint dir: `stage1_artifacts/checkpoints/kl_depth_stage1_20260529_215046`
- checkpoint used for current top-p comparison:
  `stage1_artifacts/checkpoints/kl_depth_stage1_20260529_215046/best_val.pth`

Training setup recorded in `config.json` and logs:

- epochs: 20
- batch size: 4
- train scope: `base_head`
- learning rate: `1e-5`
- depth weights: `1.0,0.7,0.4,0.2`
- baseline KL weight: `0.05`
- KL temperature: `2.0`
- end-token auxiliary loss weight: `0.01`

Final/best epoch in the recorded 20-epoch run:

- epoch: 19
- train loss: 2.6072
- train CE loss: 2.1554
- train token accuracy: 0.4042
- train depth accuracy: `[0.4377, 0.5289, 0.3824, 0.2677]`
- val loss: 2.6920
- val CE loss: 2.2766
- val token accuracy: 0.3921
- val depth accuracy: `[0.3803, 0.5027, 0.3864, 0.2988]`

Interpretation:

- Token-level validation metrics improve and are internally consistent.
- These metrics alone do not prove long-text generation quality.
- In particular, predicting RVQ indices in short 50-token windows does not
  guarantee correct progress through a multi-step long prompt.

### Current top-p comparison

Important: older videos under previous run directories were produced before the
current top-p change and must be treated as historical diagnostics only.  They
are not the current conclusion.

Current conclusion below is based on the fresh top-p run only.  Any earlier
greedy/fixed-top-k comparison can be used to explain why the decoding path was
changed, but not to evaluate the current fine-tuned model.

New top-p comparison:

- run id: `top_p_stage1_20260612_105644`
- BVH dir:
  `stage1_artifacts/generated_bvh_compare/top_p_stage1_20260612_105644`
- MP4 dir:
  `stage1_artifacts/generated_video_compare/top_p_stage1_20260612_105644`
- summary:
  `stage1_artifacts/generated_video_compare/top_p_stage1_20260612_105644/summary.json`
- metric summary:
  `stage1_artifacts/generated_bvh_compare/top_p_stage1_20260612_105644/summary_metrics.json`
- screenshot sanity checks:
  `stage1_artifacts/generated_video_compare/top_p_stage1_20260612_105644/screenshots`

Generation parameters:

- `top_p=0.95`
- `top_k=0`
- `temperature=1.0`
- `seed=123`
- `max_length=75`
- `generation_mode=auto`
- `context_size=30`
- `chunk_size=20`

Prompts:

- `walk_turn_wave`: `a person walks forward then turns around then waves both arms`
- `circle_crouch_stand`: `a person walks in a circle then crouches down then stands up`

Generated BVH lengths:

| Prompt | Model | Frames at 120 Hz | Duration |
|---|---:|---:|---:|
| `walk_turn_wave` | baseline top-p | 696 | 5.80 s |
| `walk_turn_wave` | finetuned top-p | 1656 | 13.80 s |
| `circle_crouch_stand` | baseline top-p | 720 | 6.00 s |
| `circle_crouch_stand` | finetuned top-p | 1656 | 13.80 s |

Rough motion statistics:

| Prompt/model | Root path | Root displacement | Pose velocity | Lag-30 cosine |
|---|---:|---:|---:|---:|
| `circle_crouch_stand` baseline top-p | 3.935 | 1.127 | 0.1469 | 0.9222 |
| `circle_crouch_stand` finetuned top-p | 4.521 | 1.025 | 0.1015 | 0.9787 |
| `walk_turn_wave` baseline top-p | 1.264 | 1.166 | 0.0597 | 0.9937 |
| `walk_turn_wave` finetuned top-p | 3.256 | 2.286 | 0.0619 | 0.9951 |

Interpretation:

- The current top-p finetuned model generates longer sequences than baseline.
- The rough lag-similarity metric is still high for finetuned outputs, so there
  is still a repetition risk.
- These numbers are not semantic correctness metrics.  The MP4s should be
  inspected before claiming that finetuning improves long-text behavior.
- Screenshot extraction at 2s and 6s succeeded for both side-by-side videos.
  Image statistics were non-degenerate, so the rendered comparison videos are
  not blank.
- Current evidence supports the narrower statement: top-p generation now runs
  end-to-end and produces fresh baseline-vs-finetuned artifacts, but the
  long-text quality problem is not solved yet.

### Code audit: HumanML3D to MoConVQ cache

Current implemented route:

```text
HumanML3D new_joints
  -> hard-coded 22-joint to 20-body mapping
  -> estimated body quaternions/velocities
  -> MoConVQ state: (T, 20, 13)
  -> state2ob(): (T, 323)
  -> agent.encode_seq_all()
  -> latent_vq + RVQ indices
  -> 50-token GPT windows
```

What is covered:

- Shape and finite-value checks are covered by unit tests.
- Left/right mapping order is covered by unit tests.
- `state2ob()` output dimension is covered by tests.
- cache window size is guarded against exceeding GPT temporal context.
- clip-aligned windows and window captions are covered by tests.

Remaining risk:

- The hard-coded HumanML3D-to-MoConVQ mapping is a heuristic retarget, not the
  original MoConVQ BVH-to-character retarget path.
- Quaternions are estimated from joint positions, so local body orientations may
  be less accurate than BVH rotations loaded through the original simulator
  character pipeline.
- Because the finetuned model still tends to repeat, the next priority is not
  only decoding strategy; it is to compare this heuristic retarget path against
  the original MoConVQ `MotionDataSet.add_bvh_with_character()` path whenever a
  reliable HumanML3D/AMASS-to-BVH source is available.

Additional distribution diagnostic:

- script:
  `Script/stage1/diagnose_observation_distribution.py`
- output:
  `stage1_artifacts/diagnostics/humanml3d_to_moconvq_observation_train20.json`
- sample: first 20 sequences from
  `stage1_artifacts/long_humanml3d_fixed/train/long_sequences.h5`

Observed normalized-observation statistics against MoConVQ's own
`obs_mean/obs_std`:

- aggregate mean `|z|`: 1.3532
- aggregate p90 `|z|`: 2.9944
- aggregate p95 `|z|`: 5.6526
- aggregate p99 `|z|`: 19.3859
- maximum `|z|`: 85.6293
- fraction `|z| > 3`: 9.98%
- fraction `|z| > 5`: 6.72%
- fraction `|z| > 10`: 2.49%

Worst normalized dimensions by p99 `|z|`:

| Dim | Section | Body | Component | p99 `|z|` | MoConVQ std |
|---:|---|---|---:|---:|---:|
| 84 | `local_rot_6d` | `lUpperLeg` | 0 | 19.6686 | 0.1000 |
| 78 | `local_rot_6d` | `rUpperLeg` | 0 | 19.6682 | 0.1000 |
| 90 | `local_rot_6d` | `rLowerLeg` | 0 | 19.6030 | 0.1000 |
| 96 | `local_rot_6d` | `lLowerLeg` | 0 | 19.5879 | 0.1000 |
| 102 | `local_rot_6d` | `rFoot` | 0 | 18.3136 | 0.1058 |
| 114 | `local_rot_6d` | `rToes` | 0 | 18.2745 | 0.1060 |
| 108 | `local_rot_6d` | `lFoot` | 0 | 17.9754 | 0.1082 |
| 120 | `local_rot_6d` | `lToes` | 0 | 17.4927 | 0.1110 |
| 293 | `local_avel` | `lLowerArm` | z | 14.3561 | 1.1471 |
| 290 | `local_avel` | `rLowerArm` | z | 11.5216 | 1.2017 |

Interpretation:

- This is a stronger warning than a shape test.  The heuristic joint-to-state
  path produces many observations outside the range implied by the original
  MoConVQ normalization statistics.
- The worst dimensions are mostly leg/foot `local_rot_6d` channels.  These are
  low-variance observation channels where MoConVQ's stored std is around `0.1`,
  so small orientation-coordinate mistakes become very large normalized errors.
- This supports the hypothesis that the dataset conversion/retarget path is a
  major contributor to poor finetuned generation quality.

### Next experiment

Recommended next run:

1. Keep the top-p inference path.
2. Build a small BVH-based cache through MoConVQ's original
   `MotionDataSet.add_bvh_with_character()` path, if reliable BVH sources are
   available.
3. Compare reconstruction and generation before another full fine-tune:
   - heuristic HumanML3D joint-to-state cache;
   - original BVH-to-character cache.
4. Evaluate with fresh top-p artifacts only, not old greedy/fixed-top-k videos.

## 2026-06-12: Rest-pose rotation calibration for HumanML3D retarget

### Purpose

The previous diagnostic showed that the largest HumanML3D-to-MoConVQ
observation outliers were leg/foot `local_rot_6d` channels.  This suggested a
static body-frame mismatch: HumanML3D `new_joints` contains joint positions, but
MoConVQ's `state2ob()` expects simulator rigid-body quaternions in the body
frames defined by `Data/Misc/world.json`.

### Code changes

- Added `rotation_calibration` support to the HumanML3D retarget path:
  - `none`: previous heuristic bone-axis quaternion path.
  - `rest`: remove the static offset between the heuristic rest-pose
    quaternions and MoConVQ's world-json rest-pose body quaternions.
- The default is now `rotation_calibration=rest` for:
  - `humanml3d_joints_to_moconvq_state()`
  - `build_real_moconvq_gpt_cache.py`
  - `convert_humanml3d_to_moconvq_observation.py`
  - `diagnose_observation_distribution.py`
- Cache config now records:
  - `rotation_calibration`
  - `world_json`
- Added tests that verify:
  - the rest calibration aligns the heuristic rest pose with the MoConVQ
    world-json body quaternions;
  - cache metadata records the calibration mode.

### Verification

Commands run:

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd /home/chenjie/cc/robotics/MoConVQ

python -m py_compile \
  Script/stage1/real_moconvq_cache.py \
  Script/stage1/diagnose_observation_distribution.py \
  Script/stage1/convert_humanml3d_to_moconvq_observation.py \
  tests/test_stage1_real_cache.py \
  tests/test_stage1_observation_diagnostics.py

python -m unittest \
  tests.test_stage1_real_cache \
  tests.test_stage1_observation_diagnostics \
  -v
```

Results:

- `py_compile`: passed.
- `tests.test_stage1_real_cache`: 11 tests passed.
- `tests.test_stage1_observation_diagnostics`: 1 test passed.

### Diagnostic comparison

Data:

- `stage1_artifacts/long_humanml3d_fixed/train/long_sequences.h5`
- first 20 sequences
- MoConVQ normalization statistics from `moconvq_base.data`

Outputs:

- `stage1_artifacts/diagnostics/humanml3d_to_moconvq_observation_train20_calib_none.json`
- `stage1_artifacts/diagnostics/humanml3d_to_moconvq_observation_train20_calib_rest.json`

Aggregate normalized-observation statistics:

| Calibration | mean `|z|` | p90 `|z|` | p95 `|z|` | p99 `|z|` | frac `|z|>3` | frac `|z|>5` | frac `|z|>10` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | 1.3532 | 2.9944 | 5.6526 | 19.3859 | 9.98% | 6.72% | 2.49% |
| `rest` | 0.7095 | 1.7033 | 2.6180 | 5.0565 | 4.08% | 1.07% | 0.09% |

Interpretation:

- Rest-pose calibration substantially reduces the observation distribution
  shift caused by static body-frame mismatch.
- The remaining worst dimensions still include lower-leg/foot rotation and
  angular-velocity channels, so this is not a full retarget solution.
- The next cache should be rebuilt with `--rotation-calibration rest`; the old
  cache was built with the uncalibrated path and should not be used for the next
  fine-tuning conclusion.
- This result is consistent with the MoConVQ paper's warning that retargeting
  quality can significantly affect text-to-motion metrics.  The paper evaluates
  Text2Motion with FID and R-precision on HumanML3D and notes that retargeting
  between the simulated character and SMPL lowers R-precision.

### Evaluation metric plan

The next baseline-vs-finetuned comparison should include both rendered videos
and quantitative metrics:

- Paper-aligned metrics:
  - FID on generated HumanML3D/SMPL-style motion features.
  - R-precision for text-motion semantic matching.
- Stage1 engineering diagnostics:
  - generated duration / early-stop rate;
  - lagged pose similarity or repeated RVQ tuple rate;
  - root path length and root displacement;
  - observation z-score distribution against MoConVQ `obs_mean/obs_std`;
  - if practical, foot sliding/contact artifacts from decoded BVH.

The FID/R-precision implementation still needs a compatible HumanML3D
evaluator/feature extractor or retarget-to-SMPL path.  Until that is available,
video inspection and engineering diagnostics are useful but not sufficient to
claim improvement over baseline.

## 2026-06-12: BVH engineering metrics script

### Purpose

The updated project goal requires quantitative comparison in addition to video
rendering.  The MoConVQ paper reports Text2Motion FID and R-precision on
HumanML3D, but this repository does not currently include the compatible
HumanML3D evaluator / SMPL motion feature extractor needed for those metrics.
As an intermediate diagnostic, Stage1 now has a BVH metric script for generated
baseline-vs-finetuned artifacts.

### Code changes

- Added `Script/stage1/evaluate_bvh_metrics.py`.
- Added `tests/test_stage1_bvh_metrics.py`.
- The script reports:
  - frame count, FPS, duration;
  - early-stop flag if `--expected-min-frames` is provided;
  - root path length and root displacement;
  - root path/displacement ratio;
  - pose velocity and pose variance;
  - lagged centered-pose cosine and high-similarity repeat fraction.

These are engineering diagnostics, not a replacement for FID/R-precision.

### Verification

Commands run:

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd /home/chenjie/cc/robotics/MoConVQ

python -m py_compile \
  Script/stage1/evaluate_bvh_metrics.py \
  tests/test_stage1_bvh_metrics.py

python -m unittest tests.test_stage1_bvh_metrics -v
```

Results:

- `py_compile`: passed.
- `tests.test_stage1_bvh_metrics`: 1 test passed.

### Current top-p artifact metrics

Command:

```bash
python Script/stage1/evaluate_bvh_metrics.py \
  'stage1_artifacts/generated_bvh_compare/top_p_stage1_20260612_105644/*.bvh' \
  --sample-stride 6 \
  --lags 5,10,20,30 \
  --expected-min-frames 1200 \
  --output stage1_artifacts/generated_bvh_compare/top_p_stage1_20260612_105644/summary_metrics_script.json
```

Selected results:

| Prompt/model | Frames | Duration | Early stop | Root path | Root disp. | Lag-5 repeat >0.995 |
|---|---:|---:|---|---:|---:|---:|
| `circle_crouch_stand` baseline top-p | 720 | 6.00 s | yes | 3.935 | 1.127 | 0.87% |
| `circle_crouch_stand` finetuned top-p | 1656 | 13.80 s | no | 4.521 | 1.025 | 15.13% |
| `walk_turn_wave` baseline top-p | 696 | 5.80 s | yes | 1.264 | 1.166 | 0.00% |
| `walk_turn_wave` finetuned top-p | 1656 | 13.80 s | no | 3.256 | 2.286 | 0.00% |

Interpretation:

- Under the current top-p run, baseline still ends early on both prompts when
  using a 1200-frame threshold.
- The fine-tuned checkpoint avoids early stop and produces longer BVH files.
- Longer is not automatically better: `circle_crouch_stand` finetuned has a
  high short-lag repeat fraction, so video inspection and a future
  FID/R-precision implementation are still needed before claiming it beats
  baseline.
- The next post-calibration retraining run should use this script by default
  for every generated comparison directory.

## 2026-06-12: Calibrated cache construction smoke

### Purpose

Before rebuilding the full training cache, verify that the new
`rotation_calibration=rest` path works end-to-end through:

```text
HumanML3D joints -> calibrated MoConVQ state -> state2ob -> encode_seq_all
-> RVQ indices/latents -> T5 features -> training cache
```

### Command

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd /home/chenjie/cc/robotics/MoConVQ
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python Script/stage1/build_real_moconvq_gpt_cache.py \
  --long-h5 stage1_artifacts/long_humanml3d_fixed/val/long_sequences.h5 \
  --manifest stage1_artifacts/long_humanml3d_fixed/val/manifest.jsonl \
  --base-data moconvq_base.data \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --window-size 50 \
  --window-stride 25 \
  --rvq-depth 4 \
  --caption-mode window \
  --window-policy clip \
  --forced-transition-margin 2 \
  --rotation-calibration rest \
  --gpu 0 \
  --max-failure-rate 0.5 \
  --output /tmp/stage1_rest_cache_smoke/val_cache.pt \
  --failure-log /tmp/stage1_rest_cache_smoke/val_failures.jsonl
```

### Result

- windows: 598
- failed sequences: 0
- failure rate: 0.0
- index range: 0 to 511
- `latents`: `(598, 50, 768)`
- `indices`: `(598, 50, 4)`
- `text_features`: `(598, 256, 1024)`
- `text_masks`: `(598, 256)`
- `config.rotation_calibration`: `rest`
- `config.window_policy`: `clip`
- `config.caption_mode`: `window`

Interpretation:

- The calibrated retarget path is compatible with MoConVQ encoder and T5 cache
  construction.
- The next full experiment should rebuild both train and val caches into a new
  directory, for example `stage1_artifacts/gpt_cache_rest/`, rather than
  overwriting or reusing older uncalibrated cache files.

## 2026-06-12: Clean retraining guardrail

### Purpose

While launching the calibrated-cache retraining run, two tmux sessions
accidentally wrote into the same checkpoint directory:

```text
stage1_artifacts/checkpoints/rest_stage1_20260612_115855
```

The resulting `train_log.jsonl` contains duplicated epoch ids, for example a
0--13 sequence followed by another epoch 8 from the second writer.  Therefore
this run is treated as contaminated and must not be used as the official loss
curve, best checkpoint, or baseline comparison.

### Code fix

`Script/stage1/train_real_text_gpt.py` now adds two protections:

- a `.train.lock` file in the output directory so a second training process
  cannot write the same checkpoint directory concurrently;
- a non-empty output-directory check for clean runs, so accidental overwrites
  fail fast unless `--append-log` is explicitly used for an intentional resume.

### Verification

Commands run:

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd /home/chenjie/cc/robotics/MoConVQ

python -m py_compile Script/stage1/train_real_text_gpt.py tests/test_stage1_real_train.py
python -m unittest tests.test_stage1_real_train -v
```

Results:

- `py_compile`: passed.
- `tests.test_stage1_real_train`: 11 tests passed.

### New clean run

Started a new protected training run:

```text
run id: rest_locked_stage1_20260612_122510
checkpoint dir: stage1_artifacts/checkpoints/rest_locked_stage1_20260612_122510
log: stage1_artifacts/logs/rest_locked_stage1_20260612_122510.log
cache: stage1_artifacts/gpt_cache_rest/{train_cache.pt,val_cache.pt}
```

Training setup:

- `rotation_calibration=rest` cache;
- `train_scope=base_head`;
- `epochs=20`;
- `batch_size=8`;
- `lr=1e-5`;
- `depth_weights=1.0,0.7,0.4,0.2`;
- `baseline_kl_weight=0.05`;
- `kl_temperature=2.0`;
- `end_token_weight=0.01`;
- `teacher_checkpoint=text_generation_GPT.pth`.

Initial runtime check:

- exactly one training process was visible on GPU 0;
- the output directory contained `.train.lock`;
- `train_log.jsonl` was empty at launch, confirming this is a fresh run.

This is the run that should be used for the next official top-p generation and
BVH metric comparison after training completes.

## 2026-06-12: Protected retraining result and top-p comparison

### Training result

Clean run:

```text
run id: rest_locked_stage1_20260612_122510
checkpoint dir: stage1_artifacts/checkpoints/rest_locked_stage1_20260612_122510
figure dir: stage1_artifacts/figures/rest_locked_stage1_20260612_122510
```

The run completed 20 epochs with a monotonic epoch log from 0 to 19.  The
training lock was released after completion, and no duplicate epoch ids were
found by `plot_train_curves.py`.

Final/best metrics:

- best validation epoch: 19
- train loss: 3.6098
- train CE loss: 3.1149
- train token accuracy: 0.2692
- val loss: 3.6440
- val CE loss: 3.1756
- val token accuracy: 0.2587

Artifacts:

- `stage1_artifacts/figures/rest_locked_stage1_20260612_122510/loss_accuracy_curve.png`
- `stage1_artifacts/figures/rest_locked_stage1_20260612_122510/loss_accuracy_curve.pdf`
- `stage1_artifacts/figures/rest_locked_stage1_20260612_122510/loss_accuracy_curve_data.csv`
- `stage1_artifacts/figures/rest_locked_stage1_20260612_122510/curve_summary.json`

Interpretation:

- The supervised RVQ token objective is being optimized: training and validation
  losses decrease, and token accuracy increases.
- This still does not prove long-text generation quality.  Following the MoConVQ
  paper, a paper-level comparison should eventually use HumanML3D-style
  FID/R-precision with a compatible pretrained motion-text evaluator.  This
  repository currently records BVH engineering diagnostics as an interim check.

### Top-p generation comparison

Comparison run:

```text
run id: rest_locked_stage1_20260612_122510_top_p
BVH dir: stage1_artifacts/generated_bvh_compare/rest_locked_stage1_20260612_122510_top_p
video dir: stage1_artifacts/generated_video_compare/rest_locked_stage1_20260612_122510_top_p
metrics: stage1_artifacts/generated_bvh_compare/rest_locked_stage1_20260612_122510_top_p/summary_metrics_script.json
summary: stage1_artifacts/generated_video_compare/rest_locked_stage1_20260612_122510_top_p/summary.json
```

Generation parameters:

- `top_p=0.95`
- `top_k=0`
- `temperature=1.0`
- `seed=123`
- `max_length=75`
- `generation_mode=auto`
- `context_size=30`
- `chunk_size=20`
- `allow_early_stop=true`

Frame-level outcome at 120 Hz:

| Prompt | Baseline frames | Finetuned frames | Baseline early stop | Finetuned early stop |
|---|---:|---:|---|---|
| `walk_turn_wave` | 696 | 408 | yes | yes |
| `circle_crouch_stand` | 720 | 1656 | yes | no |
| `walk_jump_dance` | 792 | 408 | yes | yes |
| `sidestep_kick_turn` | 696 | 1656 | yes | no |

Selected engineering metrics:

| Prompt/model | Duration | Root path | Pose velocity mean | Lag-5 repeat >0.995 |
|---|---:|---:|---:|---:|
| `circle_crouch_stand` baseline | 6.00 s | 3.935 | 33.337 | 0.87% |
| `circle_crouch_stand` finetuned | 13.80 s | 4.443 | 14.636 | 0.00% |
| `sidestep_kick_turn` baseline | 5.80 s | 0.740 | 3.370 | 0.00% |
| `sidestep_kick_turn` finetuned | 13.80 s | 4.307 | 16.017 | 0.37% |
| `walk_jump_dance` baseline | 6.60 s | 1.334 | 5.749 | 4.72% |
| `walk_jump_dance` finetuned | 3.40 s | 1.007 | 13.675 | 0.00% |
| `walk_turn_wave` baseline | 5.80 s | 1.264 | 5.689 | 0.00% |
| `walk_turn_wave` finetuned | 3.40 s | 1.007 | 13.675 | 0.00% |

Video artifacts:

- `stage1_artifacts/generated_video_compare/rest_locked_stage1_20260612_122510_top_p/walk_turn_wave__baseline_top_p_vs_finetuned_top_p.mp4`
- `stage1_artifacts/generated_video_compare/rest_locked_stage1_20260612_122510_top_p/circle_crouch_stand__baseline_top_p_vs_finetuned_top_p.mp4`
- `stage1_artifacts/generated_video_compare/rest_locked_stage1_20260612_122510_top_p/walk_jump_dance__baseline_top_p_vs_finetuned_top_p.mp4`
- `stage1_artifacts/generated_video_compare/rest_locked_stage1_20260612_122510_top_p/sidestep_kick_turn__baseline_top_p_vs_finetuned_top_p.mp4`

Interpretation:

- The protected fine-tuned model does not yet demonstrate a stable advantage
  over baseline on long-text generation.
- It avoids early stopping for two prompts (`circle_crouch_stand` and
  `sidestep_kick_turn`) and produces longer motions there.
- It stops earlier than baseline for two prompts (`walk_turn_wave` and
  `walk_jump_dance`), producing only 408 frames.
- This mixed result suggests that the current cache/training setup is still not
  enough to make the model robustly track long multi-clause prompts.  The next
  fix should focus on the long-horizon formulation itself: segment-progress
  conditioning, curriculum on compound prompts, better HumanML3D-to-MoConVQ
  retarget validation, and eventually paper-level HumanML3D FID/R-precision.

## 2026-06-12: Segment-progress conditioning and segment-prefix training fix

### Purpose

The previous top-p comparison showed that the fine-tuned model did not reliably
track multi-clause prompts. Two formulation issues were identified:

- segmented inference could treat a segment-level early stop as a global prompt
  stop, so later text segments were skipped;
- training examples were still primarily clip/window examples, so the model did
  not explicitly learn `previous motion prefix + current segment text -> next
  segment tokens`.

This change is a code-level fix and preparation for the next real experiment. It
is not a claim that Stage1 now beats baseline. A new cache and new training run
are required before making any model-quality conclusion.

### Code changes

- Added `Script/stage1/segment_conditioning.py`.
  - Builds deterministic 512-d segment/progress features.
  - Injects them through MoConVQ GPT's existing `clip_feature` pathway, so the
    transformer architecture and pretrained checkpoint format remain compatible.
- Updated cache construction in `Script/stage1/real_moconvq_cache.py`.
  - Added `--sample-mode segment_prefix`.
  - Added `--prefix-size`.
  - Cache now records `target_masks`, `segment_idxs`, `num_segments`,
    `segment_progress`, `prefix_ranges`, `target_ranges`, `segment_ranges`, `prefix_lengths`, and
    `end_masks`.
  - In `segment_prefix` mode, each sample contains a previous motion prefix plus
    current segment target tokens. Prefix tokens are context only.
- Updated training loss in `Script/stage1/train_real_text_gpt.py`.
  - CE, KL, token accuracy and per-depth accuracy now respect `target_mask`.
  - Prefix tokens are ignored by the supervised objective.
  - End-token auxiliary loss is applied at the first padding step after the
    supervised target region.
  - Added `--progress-conditioning`, `--progress-scale`, and `--context-size`.
  - Added `--train-scope temporal_base_head`, because progress conditioning
    enters through the temporal/text condition pathway; freezing that pathway
    would prevent the model from learning the new condition signal.
- Updated generation.
  - `Script/stage1/generate_long_motion.py` passes segment progress during
    segmented generation.
  - Segment-level early stop no longer breaks the whole multi-clause prompt.
  - `Script/stage1/export_baseline_intermediate.py` and
    `Script/stage1/run_text_gpt_comparison.py` expose the same progress
    conditioning parameters.
- Updated HumanML3D synthesis and diagnostics.
  - `synthesize_long_humanml3d.py` adds `--drop-overlap-frames`, defaulting to
    1, to avoid duplicating the aligned boundary frame.
  - Added `Script/stage1/diagnose_long_humanml3d_quality.py` to report root,
    velocity, yaw and foot discontinuities at synthetic clip boundaries.

### Verification

Commands run:

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd /home/chenjie/cc/robotics/MoConVQ

python -m py_compile \
  Script/stage1/segment_conditioning.py \
  Script/stage1/real_moconvq_cache.py \
  Script/stage1/train_real_text_gpt.py \
  Script/stage1/generate_long_motion.py \
  Script/stage1/export_baseline_intermediate.py \
  Script/stage1/run_text_gpt_comparison.py \
  Script/stage1/synthesize_long_humanml3d.py \
  Script/stage1/diagnose_long_humanml3d_quality.py

python -m unittest \
  tests.test_stage1_real_cache \
  tests.test_stage1_real_train \
  tests.test_stage1_real_generate \
  tests.test_stage1_real_synthesis \
  tests.test_stage1_gpt \
  tests.test_stage1_intermediate_export \
  tests.test_stage1_text_gpt_comparison \
  -v
```

Result:

- `py_compile`: passed.
- Relevant Stage1 tests: 56 tests passed.

### Next experiment

The next real run should rebuild data/cache and retrain, rather than reusing the
old `rest_locked_stage1_20260612_122510` cache/checkpoint as the main result.
Recommended flow:

```bash
python Script/stage1/synthesize_long_humanml3d.py ... --drop-overlap-frames 1

python Script/stage1/diagnose_long_humanml3d_quality.py \
  --long-h5 <split>/long_sequences.h5 \
  --manifest <split>/manifest.jsonl \
  --output-json <split>/dataset_quality.json \
  --transition-jsonl <split>/transition_quality.jsonl

python Script/stage1/build_real_moconvq_gpt_cache.py \
  ... \
  --caption-mode window \
  --window-policy clip \
  --sample-mode segment_prefix \
  --prefix-size 25

python Script/stage1/train_real_text_gpt.py \
  ... \
  --train-scope temporal_base_head \
  --progress-conditioning auto \
  --progress-scale 1.0
```

Only after this new run should baseline-vs-finetuned top-p BVH/MP4 comparisons
be treated as the current Stage1 result.

## 2026-06-12: Loss/protocol guard before segment-progress retraining

### Purpose

Before launching the next long run, the Stage1 loss and comparison protocol were
checked again.  One issue was found in the KL distillation path: when progress
conditioning was enabled, the baseline teacher received the same progress
feature as the student.  The teacher checkpoint was never trained with this
extra condition vector, so this made the KL term constrain the student toward a
baseline distribution under an out-of-distribution condition.  That is not the
intended role of the KL term.

The intended setup is:

- student: receives segment-progress conditioning and learns the new long-prompt
  control signal;
- teacher/baseline KL: receives the original zero `clip_feature`, so it only
  regularizes toward the pretrained text-to-motion prior;
- baseline comparison: uses the same top-p sampling settings, but no segment
  progress feature unless explicitly requested.

### Code changes

- `train_real_text_gpt.py`
  - Added `--teacher-progress-conditioning`, defaulting to `none`.
  - The student still uses `--progress-conditioning auto` for segment-progress
    training.
  - The teacher used by `--baseline-kl-weight` now defaults to progress-free
    conditioning.
- `run_text_gpt_comparison.py`
  - Added `--baseline-progress-conditioning`, defaulting to `none`.
  - The finetuned model keeps `--progress-conditioning auto`.
- `export_baseline_intermediate.py`
  - Changed the default progress conditioning to `none`, matching the baseline
    checkpoint's original condition distribution.
- Added a regression test proving that, with KL enabled, the student receives a
  nonzero progress feature while the teacher receives a zero feature by default.

### Verification

Commands run:

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd /home/chenjie/cc/robotics/MoConVQ

python -m py_compile \
  Script/stage1/train_real_text_gpt.py \
  Script/stage1/run_text_gpt_comparison.py \
  tests/test_stage1_real_train.py

python -m unittest \
  tests.test_stage1_real_train \
  tests.test_stage1_text_gpt_comparison \
  -v
```

Result:

- `py_compile`: passed.
- Relevant tests: 18 tests passed.

### Status

This is still a code-level fix, not a model-quality result.  The next step is to
rebuild the segment-prefix cache and retrain with the corrected KL/comparison
protocol.

## 2026-06-12: Segment-prefix retraining result and length-120 top-p comparison

### Dataset and cache

New run:

```text
run id: segment_progress_stage1_20260612_135307
long data root: stage1_artifacts/long_humanml3d_segment_progress_segment_progress_stage1_20260612_135307
cache root: stage1_artifacts/gpt_cache_segment_progress_segment_progress_stage1_20260612_135307
checkpoint dir: stage1_artifacts/checkpoints/segment_progress_stage1_20260612_135307
figure dir: stage1_artifacts/figures/segment_progress_stage1_20260612_135307
```

Synthesis settings:

- `num_sequences=1000` train and `200` val;
- `min_clips=2`, `max_clips=4`;
- `candidate_pool=256`;
- `transition_max_score=0.35`;
- `drop_overlap_frames=1`;
- forced transitions disabled.

Synthesis and quality summary:

| Split | Sequences | Avg clips | Avg frames | Avg duration | Forced transitions | Bad transition rate |
|---|---:|---:|---:|---:|---:|---:|
| train | 1000 | 2.945 | 414.648 | 20.73 s | 0 | 1.49% |
| val | 200 | 2.990 | 408.210 | 20.41 s | 0 | 1.76% |

Cache summary:

| Split | Windows | Failed sequences | Index range | Sample mode |
|---|---:|---:|---|---|
| train | 6604 | 0 | 0-511 | `segment_prefix` |
| val | 1321 | 0 | 0-511 | `segment_prefix` |

### Training

Training settings:

- initialized from `text_generation_GPT.pth`;
- teacher checkpoint: `text_generation_GPT.pth`;
- `train_scope=temporal_base_head`;
- `batch_size=12`;
- `epochs=20`;
- `lr=1e-5`;
- `depth_weights=1.0,0.7,0.4,0.2`;
- `baseline_kl_weight=0.05`;
- `kl_temperature=2.0`;
- `end_token_weight=0.01`;
- student progress conditioning: `auto`;
- teacher progress conditioning: `none`.

Training completed 20 epochs.  The curve was generated with:

```bash
python Script/stage1/plot_train_curves.py \
  --train-log stage1_artifacts/checkpoints/segment_progress_stage1_20260612_135307/train_log.jsonl \
  --output-dir stage1_artifacts/figures/segment_progress_stage1_20260612_135307
```

Training curve artifacts:

- `stage1_artifacts/figures/segment_progress_stage1_20260612_135307/loss_accuracy_curve.png`
- `stage1_artifacts/figures/segment_progress_stage1_20260612_135307/loss_accuracy_curve.pdf`
- `stage1_artifacts/figures/segment_progress_stage1_20260612_135307/loss_accuracy_curve_data.csv`
- `stage1_artifacts/figures/segment_progress_stage1_20260612_135307/curve_summary.json`

Key metrics:

| Metric | Value |
|---|---:|
| best val epoch | 9 |
| best val loss | 2.9766 |
| best val token accuracy | 0.3603 |
| best val-accuracy epoch | 11 |
| best val token accuracy | 0.3627 |
| last train loss | 2.0503 |
| last train token accuracy | 0.5218 |
| last val loss | 3.1209 |
| last val token accuracy | 0.3545 |

Interpretation:

- The new segment-prefix objective trains successfully.
- Validation improves until around epoch 9-11, then degrades while train loss
  continues to fall.  This suggests overfitting or a mismatch between the
  supervised token objective and generation quality.
- The current default evaluation should use `best_val.pth`, not `last.pth`.

### Top-p comparison

Comparison run:

```text
run id: segment_progress_stage1_20260612_135307_top_p_len120
BVH dir: stage1_artifacts/generated_bvh_compare/segment_progress_stage1_20260612_135307_top_p_len120
video dir: stage1_artifacts/generated_video_compare/segment_progress_stage1_20260612_135307_top_p_len120
metrics: stage1_artifacts/generated_bvh_compare/segment_progress_stage1_20260612_135307_top_p_len120/summary_metrics_script.json
```

Generation settings:

- `max_length=120`;
- `generation_mode=auto`;
- `context_size=30`;
- `chunk_size=20`;
- `top_k=0`;
- `top_p=0.95`;
- `temperature=1.0`;
- `seed=123`;
- finetuned progress conditioning: `auto`;
- baseline progress conditioning: `none`.

Frame-level results at 120 Hz:

| Prompt | Baseline frames | Finetuned frames | Baseline duration | Finetuned duration |
|---|---:|---:|---:|---:|
| `walk_turn_wave` | 1272 | 2736 | 10.60 s | 22.80 s |
| `circle_crouch_stand` | 1464 | 2520 | 12.20 s | 21.00 s |
| `walk_jump_dance` | 1488 | 2736 | 12.40 s | 22.80 s |
| `sidestep_kick_turn` | 1776 | 2520 | 14.80 s | 21.00 s |
| average | 1500 | 2628 | 12.50 s | 21.90 s |

Selected engineering diagnostics:

| Model | Avg root path | Avg root displacement | Avg pose velocity | Avg pose variance | Avg lag-20 repeat >0.995 |
|---|---:|---:|---:|---:|---:|
| baseline | 4.024 | 1.416 | 17.781 | 152.256 | 0.00% |
| finetuned | 7.122 | 2.624 | 58.087 | 763.189 | 4.33% |

Video artifacts:

- `stage1_artifacts/generated_video_compare/segment_progress_stage1_20260612_135307_top_p_len120/walk_turn_wave__baseline_top_p_vs_finetuned_top_p.mp4`
- `stage1_artifacts/generated_video_compare/segment_progress_stage1_20260612_135307_top_p_len120/circle_crouch_stand__baseline_top_p_vs_finetuned_top_p.mp4`
- `stage1_artifacts/generated_video_compare/segment_progress_stage1_20260612_135307_top_p_len120/walk_jump_dance__baseline_top_p_vs_finetuned_top_p.mp4`
- `stage1_artifacts/generated_video_compare/segment_progress_stage1_20260612_135307_top_p_len120/sidestep_kick_turn__baseline_top_p_vs_finetuned_top_p.mp4`

Contact sheets:

- `stage1_artifacts/generated_video_compare/segment_progress_stage1_20260612_135307_top_p_len120/contact_sheets/`

Interpretation:

- The segment-progress model fixes a real failure mode from the previous run:
  it generates substantially longer motions for all four long prompts under the
  same top-p sampling setup.
- This is not yet a full win over baseline.  The finetuned outputs have much
  higher pose velocity and pose variance, and contact sheets suggest less stable
  motion quality on some prompts.
- The next correction should reduce the mismatch between token-level validation
  and rollout quality.  Immediate candidates are:
  - tune inference progress scale;
  - reduce train scope or add stronger KL/regularization;
  - filter or downweight bad synthetic boundaries;
  - add rollout-oriented diagnostics such as foot sliding and root jerk;
  - integrate paper-level HumanML3D evaluator metrics when practical.

### Inference progress-scale ablation

A lightweight inference-only ablation was run with the same checkpoint and
generation settings, changing only:

```text
progress_scale=0.5
run id: segment_progress_stage1_20260612_135307_top_p_len120_scale05
```

Frame-level result:

| Prompt | Baseline frames | Finetuned frames |
|---|---:|---:|
| `walk_turn_wave` | 1272 | 2304 |
| `circle_crouch_stand` | 1464 | 2304 |
| `walk_jump_dance` | 1488 | 2304 |
| `sidestep_kick_turn` | 1776 | 2304 |
| average | 1500 | 2304 |

Aggregate diagnostics:

| Setting | Avg finetuned frames | Avg pose velocity | Avg pose variance | Avg lag-20 repeat >0.995 |
|---|---:|---:|---:|---:|
| progress scale 1.0 | 2628 | 58.087 | 763.189 | 4.33% |
| progress scale 0.5 | 2304 | 53.314 | 539.704 | 0.34% |

Interpretation:

- Reducing progress scale keeps the main benefit over baseline, namely longer
  generation, while reducing pose variance and long-lag near-repetition.
- This does not solve all quality issues, but it is the better current inference
  default for the segment-progress checkpoint.
- Current recommended comparison command should use `--progress-scale 0.5` for
  the finetuned model and keep `--baseline-progress-conditioning none`.

## 2026-06-12: Loss audit and conservative retraining plan

### Purpose

The Stage1 training loss was inspected again before launching the next
experiment.  The main autoregressive objective is already aligned with the
MoConVQ GPT sampling path:

- motion history is fed as codebook-reconstructed RVQ latents rather than the
  encoder's cached `latent_vq`, matching rollout-time feedback;
- the temporal context is shifted by one step, so the current motion token is
  not visible while predicting itself;
- the first four RVQ depth logits supervise the four predicted RVQ code ids;
- `target_mask` excludes prefix/context-only tokens from CE, KL and accuracy;
- the baseline KL teacher receives the original progress-free condition vector.

Two smaller issues were fixed:

- legacy caches that do not contain explicit `end_masks` now infer the first
  padding step after the supervised target region;
- end-token auxiliary loss is now applied only once per motion timestep, at
  RVQ depth 0, rather than being repeated across all four RVQ depths.

### Verification

Commands run:

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd /home/chenjie/cc/robotics/MoConVQ

python -m py_compile \
  Script/stage1/train_real_text_gpt.py \
  tests/test_stage1_real_train.py

python -m unittest tests.test_stage1_real_train -v

python -m unittest \
  tests.test_stage1_gpt \
  tests.test_stage1_real_cache \
  tests.test_stage1_real_generate \
  tests.test_stage1_text_gpt_comparison \
  tests.test_stage1_real_train \
  -v
```

Result:

- `tests.test_stage1_real_train`: 18 tests passed.
- Related Stage1 GPT/cache/generation/comparison/training tests: 50 tests
  passed.
- A 1-batch smoke train completed forward/backward/checkpoint writing.

### Next experiment

The next run reuses the current segment-prefix cache, because the cache schema is
compatible with the loss fix.  The goal is to reduce the previous rollout
instability while keeping the segment-progress benefit:

```text
run id: lossfix_reg_stage1_20260612_155519
train cache: stage1_artifacts/gpt_cache_segment_progress_segment_progress_stage1_20260612_135307/train_cache.pt
val cache: stage1_artifacts/gpt_cache_segment_progress_segment_progress_stage1_20260612_135307/val_cache.pt
train_scope: temporal_base_head
epochs: 20
batch_size: 12
lr: 5e-6
depth_weights: 1.0,0.7,0.4,0.2
baseline_kl_weight: 0.1
kl_temperature: 2.0
end_token_weight: 0.01
student progress conditioning: auto
teacher progress conditioning: none
progress_scale: 0.5
```

Expected comparison after training:

- plot loss/accuracy curves;
- evaluate `best_val.pth` against `text_generation_GPT.pth` with top-p sampling;
- render side-by-side videos;
- report frame count, duration, root path/displacement, pose velocity, pose
  variance and long-lag repetition diagnostics.

If this run still improves length but not qualitative/metric quality, the next
stage will move from loss tuning to data/protocol diagnosis in this order:

1. Check synthetic HumanML3D boundary quality:
   - rerun transition diagnostics on train/val;
   - inspect the worst root/yaw/foot-velocity boundaries;
   - compare whether unstable BVH outputs correlate with high-boundary-error
     training samples.
2. Check retarget/observation distribution:
   - compute normalized MoConVQ observation z-scores against `moconvq_base.data`;
   - identify dimensions/bodies whose converted HumanML3D observations are far
     outside the MoConVQ training distribution.
3. Check caption-window alignment:
   - verify each cache window uses the local segment caption rather than the
     whole long caption when `segment_prefix` is enabled;
   - sample windows and inspect `prefix_range`, `target_range`, `segment_idx`,
     `caption` and `target_mask`.
4. Check rollout error accumulation:
   - compare teacher-forced validation loss with generated BVH metrics;
   - test shorter per-segment generation, smaller context, and lower progress
     scale to see whether instability grows with rollout length.
5. Check data coverage:
   - summarize action/caption diversity and segment-count distribution;
   - if coverage is weak, rebuild the synthetic dataset with stricter
     transition filtering or an LLM-assisted segment planner rather than simple
     sampled concatenation.

## 2026-06-12: Loss-fix retraining, top-p comparison, and data diagnosis

### Training outcome

The conservative loss-fix run completed:

```text
run id: lossfix_reg_stage1_20260612_155519
checkpoint dir: stage1_artifacts/checkpoints/lossfix_reg_stage1_20260612_155519
curve dir: stage1_artifacts/figures/lossfix_reg_stage1_20260612_155519
train cache: stage1_artifacts/gpt_cache_segment_progress_segment_progress_stage1_20260612_135307/train_cache.pt
val cache: stage1_artifacts/gpt_cache_segment_progress_segment_progress_stage1_20260612_135307/val_cache.pt
epochs: 20
batch size: 12
lr: 5e-6
train scope: temporal_base_head
depth weights: 1.0,0.7,0.4,0.2
baseline KL weight: 0.1
end-token weight: 0.01
progress scale: 0.5
```

Curve summary:

| Metric | Value |
|---|---:|
| best val epoch | 16 |
| best val loss | 3.5312 |
| last train loss | 2.8740 |
| last train CE loss | 2.0333 |
| last train token accuracy | 0.4348 |
| last val loss | 3.5501 |
| last val CE loss | 2.7310 |
| last val token accuracy | 0.3309 |

Interpretation:

- The run is not underfitting by epoch 20.  Validation loss reaches the best
  value at epoch 16 and then mildly worsens while train loss continues to fall.
- Evaluation should use `best_val.pth`, not `last.pth`.
- More epochs alone are unlikely to fix long-motion rollout quality.

### Top-p comparison against baseline

Comparison artifacts:

```text
run id: lossfix_reg_stage1_20260612_155519_top_p_len120_scale05
BVH dir: stage1_artifacts/generated_bvh_compare/lossfix_reg_stage1_20260612_155519_top_p_len120_scale05
video dir: stage1_artifacts/generated_video_compare/lossfix_reg_stage1_20260612_155519_top_p_len120_scale05
metrics: stage1_artifacts/generated_bvh_compare/lossfix_reg_stage1_20260612_155519_top_p_len120_scale05/summary_metrics_script.json
summary: stage1_artifacts/generated_bvh_compare/lossfix_reg_stage1_20260612_155519_top_p_len120_scale05/comparison_summary.md
```

Generation settings:

- `top_p=0.95`, `top_k=0`, `temperature=1.0`;
- `max_length=120`;
- `generation_mode=auto`;
- `context_size=30`, `chunk_size=20`;
- finetuned progress conditioning: `auto`;
- baseline progress conditioning: `none`;
- finetuned `progress_scale=0.5`;
- seed: `123`.

Aggregate diagnostics:

| Model | Avg frames | Avg duration | Avg root path | Avg root displacement | Avg pose velocity | Avg pose variance | Lag-20 repeat >0.995 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline top-p | 1500 | 12.50 s | 4.024 | 1.416 | 17.780 | 152.256 | 0.0000 |
| finetuned top-p | 2316 | 19.30 s | 5.873 | 1.836 | 23.936 | 358.819 | 0.0063 |

Per-prompt frame counts:

| Prompt | Baseline frames | Finetuned frames | Delta |
|---|---:|---:|---:|
| `walk_turn_wave` | 1272 | 2736 | +1464 |
| `circle_crouch_stand` | 1464 | 2280 | +816 |
| `walk_jump_dance` | 1488 | 2736 | +1248 |
| `sidestep_kick_turn` | 1776 | 1512 | -264 |

Interpretation:

- The loss-fix/regularized model is much less unstable than the previous
  segment-progress model: average pose velocity drops from about 53.3 to 23.9,
  and pose variance drops from about 539.7 to 358.8 under the same
  `progress_scale=0.5` comparison.
- It still does not beat baseline overall.  It usually generates longer clips,
  but pose variance is still more than 2x baseline and one prompt becomes
  shorter than baseline.
- This confirms that the next correction should not be "train more epochs" or
  "tune CE only"; it should diagnose the data and rollout protocol.

### Synthetic HumanML3D boundary quality

The current segment-progress synthesis data already contains boundary
diagnostics:

| Split | Sequences | Transitions | Avg clips | Avg frames | Forced transitions | Bad transition rate |
|---|---:|---:|---:|---:|---:|---:|
| train | 1000 | 1945 | 2.945 | 414.648 | 0 | 1.49% |
| val | 200 | 398 | 2.990 | 408.210 | 0 | 1.76% |

Boundary metrics are small:

| Split | root gap mean / p95 | yaw gap mean / p95 | foot velocity gap mean / p95 |
|---|---:|---:|---:|
| train | 0.0025 / 0.0133 | 0.0110 / 0.0532 | 0.0120 / 0.0554 |
| val | 0.0026 / 0.0141 | 0.0115 / 0.0565 | 0.0108 / 0.0537 |

Interpretation:

- The simple boundary checks do not support "bad clip stitching" as the main
  current failure source.  The dataset is still synthetic and semantically
  imperfect, but the measured root/yaw/foot boundary discontinuities are not
  large enough to explain the rollout instability by themselves.
- The next more suspicious part is the HumanML3D-to-MoConVQ retarget and
  observation distribution.

### Retarget / observation distribution

Observation diagnostics compare converted HumanML3D observations to the
MoConVQ encoder's `obs_mean/obs_std` from `moconvq_base.data`.

Artifacts:

```text
stage1_artifacts/diagnostics/lossfix_reg_stage1_20260612_155519/train_observation_distribution_100.json
stage1_artifacts/diagnostics/lossfix_reg_stage1_20260612_155519/val_observation_distribution_50.json
stage1_artifacts/diagnostics/lossfix_reg_stage1_20260612_155519/val_observation_distribution_50_none.json
```

Aggregate z-score diagnostics:

| Data | Calibration | mean abs z | p95 abs z | p99 abs z | frac >5 | frac >10 |
|---|---|---:|---:|---:|---:|---:|
| train 100 seq | rest | 0.7055 | 2.5861 | 5.0763 | 1.08% | 0.079% |
| val 50 seq | rest | 0.6948 | 2.5225 | 4.8894 | 0.90% | 0.074% |
| val 50 seq | none | 1.3413 | 5.5816 | 19.3859 | 6.60% | 2.481% |

Worst dimensions under `rest` calibration are consistently:

- `local_rot_6d` for `rLowerLeg`, `lLowerLeg`, `rFoot`, `lFoot`, `rToes`,
  `lToes`;
- `local_avel` for foot/toes and lower arms.

Interpretation:

- `rotation_calibration=rest` is necessary: removing it makes the converted
  distribution far worse.
- The remaining outliers point to the hand-written position-to-rigid-body
  retarget path.  HumanML3D `new_joints` provides joint positions, not the
  simulator body local frames used by MoConVQ.  Static rest-pose calibration
  fixes the largest global mismatch but cannot fully recover physically
  plausible lower-leg/foot rigid-body rotations and angular velocities.

### Code change from diagnosis

Added optional observation-quality filtering to the cache builder:

- `--max-observation-p99-abs-z`
- `--max-observation-frac-gt-5`
- `--max-observation-frac-gt-10`

Default behavior is unchanged.  When thresholds are provided, the builder
computes per-sequence observation z-score statistics before calling
`encode_seq_all()` and skips sequences that exceed the thresholds.  Skipped
sequences are stored in `cache["filtered_sequences"]`; true conversion failures
remain in the failure log.

Verification:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/real_moconvq_cache.py \
  tests/test_stage1_real_cache.py \
  Script/stage1/diagnose_observation_distribution.py \
  Script/stage1/summarize_bvh_comparison.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_real_cache \
  tests.test_stage1_observation_diagnostics \
  tests.test_stage1_bvh_comparison_summary \
  -v

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_real_train \
  tests.test_stage1_text_gpt_comparison \
  tests.test_stage1_real_generate \
  -v
```

Results:

- cache/observation/summary tests: 17 passed;
- train/comparison/generation tests: 30 passed.

### Next experiment

Build a new segment-prefix cache from the same synthesized data with moderate
observation filtering, then retrain the same conservative GPT setup:

```text
candidate filter:
  max_observation_p99_abs_z ~= 6.0
  max_observation_frac_gt_5 ~= 0.02
  max_observation_frac_gt_10 ~= 0.003
```

On the diagnostic subset this would remove approximately 6-11% of sequences,
depending on the exact threshold and split.  This is conservative enough to
keep most data while testing whether the lower-leg/foot retarget outliers are
polluting GPT supervision.

If filtered-cache training still generates longer but lower-quality motions,
the next priority should be a stronger retarget route through the original
MoConVQ BVH/character loader (`MotionDataSet.add_bvh_with_character()`) or a
more conservative data synthesis plan with LLM-assisted segment selection.

## 2026-06-12: Filtered-cache training and top-p rollout comparison

### Purpose

This run tested whether removing converted HumanML3D sequences with outlying
MoConVQ observation statistics improves Stage1 GPT fine-tuning.  The motivation
was the previous diagnosis that simple synthetic clip-boundary metrics looked
acceptable, while the converted observation distribution still had lower-leg,
foot, toe, and angular-velocity outliers.

### Cache

Run id:

```text
filtered_cache_stage1_20260612_174908
```

Cache path:

```text
stage1_artifacts/gpt_cache_filtered_cache_stage1_20260612_174908
```

Filtering thresholds:

```text
max_observation_p99_abs_z = 6.0
max_observation_frac_gt_5 = 0.02
max_observation_frac_gt_10 = 0.003
```

Cache summary:

| Split | Windows | Unique seqs kept | Seqs filtered | Failures | Latent shape | Text shape |
|---|---:|---:|---:|---:|---|---|
| train | 5767 | 877 | 123 / 1000 | 0 | `(5767, 50, 768)` | `(5767, 256, 1024)` |
| val | 1183 | 176 | 24 / 200 | 0 | `(1183, 50, 768)` | `(1183, 256, 1024)` |

Interpretation:

- The filter removed about 12% of synthesized sequences in both train and val.
- No sequence failed the conversion pipeline after filtering.
- This is a conservative test of whether observation outliers are a major
  source of bad GPT supervision.

### Training

Run id:

```text
filtered_stage1_20260612_181802
```

Checkpoint dir:

```text
stage1_artifacts/checkpoints/filtered_stage1_20260612_181802
```

Curve artifacts:

```text
stage1_artifacts/figures/filtered_stage1_20260612_181802/loss_accuracy_curve.png
stage1_artifacts/figures/filtered_stage1_20260612_181802/loss_accuracy_curve.pdf
stage1_artifacts/figures/filtered_stage1_20260612_181802/curve_summary.json
```

Training setup:

```text
epochs = 20
batch_size = 12
learning_rate = 5e-6
train_scope = temporal_base_head
depth_weights = 1.0,0.7,0.4,0.2
baseline_kl_weight = 0.1
kl_temperature = 2.0
end_token_weight = 0.01
progress_conditioning = auto
progress_scale = 0.5
teacher_progress_conditioning = none
context_size = 51
```

Curve summary:

| Metric | Value |
|---|---:|
| best val epoch | 17 |
| best val loss | 3.4974 |
| last train loss | 2.9000 |
| last val loss | 3.5221 |
| last train token accuracy | 0.4253 |
| last val token accuracy | 0.3291 |

Interpretation:

- Epochs 0-11 show healthy learning: train and val loss both decrease.
- Epochs 12-17 enter a validation plateau; epoch 17 is the best checkpoint.
- Epochs 18-19 show mild overfitting: train loss keeps dropping, while val loss
  rises from 3.4974 to 3.5221.
- The rollout comparison must therefore use `best_val.pth`, not `last.pth`.

### Top-p comparison

Run id:

```text
filtered_stage1_20260612_181802_top_p_len120_scale05
```

Artifacts:

```text
BVH:
stage1_artifacts/generated_bvh_compare/filtered_stage1_20260612_181802_top_p_len120_scale05

MP4:
stage1_artifacts/generated_video_compare/filtered_stage1_20260612_181802_top_p_len120_scale05

Summary:
stage1_artifacts/generated_bvh_compare/filtered_stage1_20260612_181802_top_p_len120_scale05/comparison_summary.md
stage1_artifacts/generated_bvh_compare/filtered_stage1_20260612_181802_top_p_len120_scale05/comparison_summary.json
```

Generation settings:

```text
top_p = 0.95
top_k = 0
temperature = 1.0
max_length = 120
generation_mode = auto
context_size = 30
chunk_size = 20
progress_scale = 0.5
baseline_progress_conditioning = none
finetuned_progress_conditioning = auto
seed = 123
```

Prompts:

- `walk_turn_wave`: `a person walks forward then turns around then waves both arms`
- `circle_crouch_stand`: `a person walks in a circle then crouches down then stands up`
- `walk_jump_dance`: `a person walks forward then jumps then dances`
- `sidestep_kick_turn`: `a person sidesteps to the left then kicks with the right foot then turns around`

Model averages:

| Model | Frames | Duration | Early stop | Root path | Root disp. | Pose velocity | Pose variance | Lag-20 repeat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline top-p | 1500.0 | 12.50 s | 0.00 | 4.024 | 1.416 | 17.780 | 152.256 | 0.000 |
| finetuned top-p | 1722.0 | 14.35 s | 0.25 | 4.844 | 1.558 | 25.065 | 307.950 | 0.000 |

Per-prompt outcome:

| Prompt | Baseline frames | Finetuned frames | Main observation |
|---|---:|---:|---|
| `walk_turn_wave` | 1272 | 1848 | Longer and lower pose variance than baseline. |
| `circle_crouch_stand` | 1464 | 1992 | Longer but much higher velocity/variance. |
| `walk_jump_dance` | 1488 | 2304 | Longer but much higher velocity/variance. |
| `sidestep_kick_turn` | 1776 | 744 | Finetuned early-stops below the 1200-frame threshold. |

Interpretation:

- Filtering observation outliers did not solve the Stage1 long-text generation
  problem.
- The fine-tuned model still tends to generate longer sequences than the
  baseline, but the engineering diagnostics do not show a quality win:
  average pose velocity and pose variance are substantially higher, and one
  prompt early-stops.
- The lag-20 repetition proxy is not the main signal in this run because both
  models have near-zero repeat fraction at the strict 0.995 threshold.  The
  stronger warning signs are instability-like pose statistics and early stop.
- This result argues against "just train longer" as the next fix.  The training
  curve already shows mild overfitting after epoch 17.

### Next diagnosis

The next priority is to inspect representation and supervision quality rather
than increase epochs:

1. Compare generated rollouts and cache targets at the MoConVQ observation /
   latent level to see whether the hand-written HumanML3D position-to-state
   retarget produces valid simulator-like body states.
2. Audit segment-prefix samples around clip transitions: verify that each
   target window uses the intended segment caption and that prefix/target masks
   match the rollout regime.
3. Test a stricter data recipe: shorter, single-transition sequences with only
   high-confidence HumanML3D clips and no aggressive semantic composition.
4. If quality still fails, prioritize the original MoConVQ BVH/character path
   (`MotionDataSet.add_bvh_with_character()`) or a backup data synthesis method
   using LLM-assisted action planning and retrieval, instead of relying on
   hand-written joint-position retarget alone.

### Follow-up: progress-feature inference alignment

After the filtered-cache rollout above, the generation script was audited for a
training/inference conditioning mismatch.  Training used segment-prefix cache
samples with `prefix_size=25` and `context_size=51`, but the comparison command
used rollout `context_size=30`.  Since segment progress conditioning includes a
prefix-length ratio, this made the inference feature for a full prefix closer to
`25/30` instead of the training value `25/51`.

Code fix:

- added `--progress-context-size` and `--progress-prefix-cap` to generation and
  comparison scripts;
- forwarded these parameters into the deterministic progress feature;
- added tests covering progress argument forwarding and the expected
  `25/51` prefix ratio.

Verification:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_real_generate \
  tests.test_stage1_text_gpt_comparison \
  tests.test_stage1_real_train \
  tests.test_stage1_bvh_comparison_summary \
  tests.test_stage1_plot_train_curves \
  -v
```

Result:

- 36 tests passed.

Aligned inference run:

```text
filtered_stage1_20260612_181802_top_p_len120_scale05_progress_aligned
```

Additional generation settings:

```text
progress_context_size = 51
progress_prefix_cap = 25
```

Model averages:

| Model | Frames | Duration | Early stop | Root path | Root disp. | Pose velocity | Pose variance | Lag-20 repeat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline top-p | 1500.0 | 12.50 s | 0.00 | 4.024 | 1.416 | 17.780 | 152.256 | 0.000 |
| finetuned top-p | 1716.0 | 14.30 s | 0.25 | 4.880 | 1.692 | 28.782 | 379.135 | 0.000 |

Interpretation:

- Aligning the progress prefix feature did not solve the rollout problem.
- The `sidestep_kick_turn` finetuned output still early-stops at 744 frames.
- Average pose velocity and pose variance became worse than the previous
  filtered run, so this is not the missing fix.
- The next likely source is not the scalar progress feature itself, but the
  supervision distribution: segment captions, target windows, HumanML3D
  semantic composition, and/or the hand-written HumanML3D-to-MoConVQ retarget.

### Follow-up: training-distribution rollout length

The cache audit showed two more possible mismatches:

- the most common training sample has a 25-token motion prefix and a 25-token
  supervised target;
- about 30% of per-segment HumanML3D captions already contain `then`, so
  splitting every user prompt by `then` can make inference segments finer than
  many training segments.

To isolate the length/context issue, an additional rollout was run near the
training distribution:

```text
filtered_stage1_20260612_181802_top_p_seg25_ctx25
```

Settings:

```text
max_length = 75
segment_length = 25
context_size = 25
chunk_size = 25
progress_context_size = 51
progress_prefix_cap = 25
top_p = 0.95
```

Model averages:

| Model | Frames | Duration | Early stop | Root path | Root disp. | Pose velocity | Pose variance | Lag-20 repeat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline top-p | 972.0 | 8.10 s | 0.00 | 2.522 | 1.087 | 15.840 | 155.313 | 0.000 |
| finetuned top-p | 1338.0 | 11.15 s | 0.00 | 3.425 | 1.387 | 29.389 | 306.626 | 0.001 |

Interpretation:

- Matching the rollout segment length and context to the dominant training
  pattern removes the early-stop case in this prompt set.
- It still does not make the fine-tuned model better than baseline: finetuned
  pose velocity and pose variance remain much higher.
- This suggests that long-context rollout mechanics are not the only problem.
  The fine-tuning supervision itself is likely pushing GPT toward a motion
  token distribution that decodes to less stable motions.

### Cache audit notes

The filtered cache has structurally valid segment-prefix samples:

- `target_masks` are contiguous and align with `target_ranges`;
- target-length and range checks pass;
- `segment_idxs`, `num_segments`, `segment_progress`, `prefix_ranges`, and
  `segment_ranges` are present.

However, caption granularity is suspicious:

| Pattern in per-window caption | Count | Share |
|---|---:|---:|
| contains `then` | 1723 / 5767 | 29.88% |
| contains `and then` | 686 / 5767 | 11.90% |
| contains `while` | 344 / 5767 | 5.96% |
| multi-clause heuristic | 1908 / 5767 | 33.08% |

Unique segment length distribution:

| Quantile | Tokens |
|---|---:|
| 25% | 22 |
| 50% | 37 |
| 75% | 49 |
| 90% | 50 |

Interpretation:

- The current cache treats each original HumanML3D clip caption as one segment,
  even if that caption already describes multiple sub-actions.
- Inference currently splits user text on `then`, which can produce shorter
  and more atomic segments than the model saw during fine-tuning.
- This supports the next data fix: normalize captions into a consistent
  segment granularity, either by avoiding `then`-splitting for captions whose
  motion is a single HumanML3D clip, or by rebuilding synthesis/cache with
  explicitly atomic sub-action segments and matching latent boundaries.

### Follow-up: RVQ token distribution shift

A token-distribution diagnostic was added:

```text
Script/stage1/diagnose_token_distribution.py
```

It compares RVQ code usage in Stage1 caches against a native MoConVQ motion
dataset observation sequence.  The native comparison uses:

```text
simple_motion_data.h5
observation key: walk1_subject5/observation
```

Output:

```text
stage1_artifacts/diagnostics/token_distribution_hml_vs_native.json
```

Result:

| Depth | HML entropy | Native entropy | HML unique | Native unique | JS divergence |
|---|---:|---:|---:|---:|---:|
| 0 | 5.385 | 5.982 | 311 | 139 | 0.928 |
| 1 | 7.530 | 8.072 | 484 | 366 | 0.483 |
| 2 | 8.114 | 8.378 | 506 | 412 | 0.330 |
| 3 | 7.074 | 8.374 | 476 | 418 | 0.587 |

Top-token concentration is especially different at RVQ depth 0:

```text
HumanML3D cache depth 0:
  token 492: 24.43%
  token 338: 12.04%

Native MoConVQ simple motion depth 0:
  token 414: 6.81%
  token 272: 6.66%
```

Interpretation:

- The HumanML3D-derived cache is not merely a noisy version of native MoConVQ
  data; its RVQ token distribution is substantially shifted.
- The strongest shift is in the first RVQ depth, which is also the most
  heavily weighted depth in the current loss.
- This supports the hypothesis that the hand-written HumanML3D
  joint-position-to-body-state retarget is producing a biased latent/token
  distribution.  Fine-tuning GPT on this distribution can improve
  teacher-forcing token loss while degrading rollout quality.
- The next engineering fix should prioritize data representation, not another
  20-epoch run on the same cache.

## 2026-06-12: MoConVQ native BVH-to-character retarget diagnostic

### Purpose

The filtered-cache experiment showed that HumanML3D-derived RVQ tokens are
substantially shifted from native MoConVQ tokens.  To separate a model-training
problem from a representation problem, this check probes the repository's
original BVH loading path:

```text
BVH file -> MotionDataSet.add_bvh_with_character()
         -> simulator character state
         -> state2ob()
         -> encode_seq_all()
         -> RVQ tokens
```

This is the path used by the original MoConVQ utilities and is therefore a
useful reference for whether our hand-written HumanML3D joint-position retarget
is producing simulator-like observations.

### Code changes

- Added `Script/stage1/diagnose_bvh_character_retarget.py`.
- Added `tests/test_stage1_bvh_character_retarget.py`.
- The diagnostic reports:
  - extracted `state` and `observation` shapes;
  - normalized observation z-scores against `moconvq_base.data`;
  - RVQ token distribution;
  - optional JS-divergence against a native H5 observation sequence.

### Verification

Commands run:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/diagnose_bvh_character_retarget.py \
  tests/test_stage1_bvh_character_retarget.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_bvh_character_retarget -v
```

Result:

- `py_compile`: passed.
- `tests.test_stage1_bvh_character_retarget`: 1 test passed.

### Real diagnostic

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/diagnose_bvh_character_retarget.py \
  base.bvh track.bvh \
  --base-data moconvq_base.data \
  --native-h5 simple_motion_data.h5 \
  --native-observation-key walk1_subject5/observation \
  --gpu 0 \
  --output-json stage1_artifacts/diagnostics/bvh_character_retarget_base_track_vs_native.json
```

The first sandboxed attempt failed because MoConVQ's backend initializes MPI and
needs to create a socket.  The same command succeeded outside the sandbox.

Observation statistics for `base.bvh + track.bvh` through the original
BVH-to-character path:

| Metric | Value |
|---|---:|
| state shape | `(503, 20, 13)` |
| observation shape | `(503, 323)` |
| latent token shape | `(125, 4)` |
| mean abs z | 0.5626 |
| p95 abs z | 1.8741 |
| p99 abs z | 2.8260 |
| frac abs z > 5 | 0.155% |
| frac abs z > 10 | 0.020% |

For comparison, the HumanML3D hand-written retarget path after rest-pose
calibration still had p99 abs z around 4.9--5.1 on sampled train/val subsets,
and the uncalibrated path had p99 abs z around 19.4.

RVQ JS divergence against `simple_motion_data.h5/walk1_subject5/observation`:

| Depth | JS divergence |
|---:|---:|
| 0 | 0.928 |
| 1 | 0.718 |
| 2 | 0.708 |
| 3 | 0.665 |

### Interpretation

- The original BVH-to-character path produces substantially healthier
  normalized observations than the hand-written HumanML3D position-to-state
  conversion.
- This supports the current diagnosis that HumanML3D retarget quality, rather
  than epoch count, is a major bottleneck.
- The RVQ JS-divergence numbers above should be interpreted cautiously:
  `base.bvh + track.bvh` produces only 125 latent tokens and is not the same
  motion distribution as `walk1_subject5` in `simple_motion_data.h5`.  The token
  comparison is therefore a smoke reference, not a final distribution-matching
  proof.
- The next data fix should either route HumanML3D through a BVH/character
  retarget path, or make the current joint-position route more conservative by
  filtering retarget outliers and normalizing segment caption granularity.

## 2026-06-12: HumanML3D caption granularity diagnosis and atomic-caption synthesis option

### Purpose

The filtered-cache audit showed that many training windows still use captions
that already contain multiple actions, while inference splits user prompts by
`then`.  This creates a granularity mismatch:

```text
training: one HumanML3D clip caption may describe multiple sub-actions
inference: one prompt is split into smaller then-separated segments
```

This check quantifies how severe that problem is and adds a data-synthesis
option to build cleaner segment-level supervision.

### Code changes

- Added caption-complexity helpers to `synthesize_long_humanml3d.py`:
  - detects `then`, `thens`, `while`, `before`, `after`, `followed by`,
    `subsequently`, `next`, and multiple-sentence captions;
  - optional word-count cap through `--max-caption-words`.
- Added synthesis arguments:
  - `--caption-filter-mode none`
  - `--caption-filter-mode prefer_atomic`
  - `--caption-filter-mode atomic`
  - `--max-caption-words`
- `prefer_atomic` keeps all samples but chooses the simplest available caption.
- `atomic` filters out samples for which none of the captions pass the
  atomic-caption heuristic.
- Manifest rows now record:
  - `clip_caption_complexity`
  - `clip_caption_is_atomic`
- Added `Script/stage1/diagnose_humanml3d_caption_granularity.py`.
- Added `tests/test_stage1_caption_granularity.py` and synthesis regression
  tests for caption filtering.

### Verification

Commands run:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/synthesize_long_humanml3d.py \
  Script/stage1/diagnose_humanml3d_caption_granularity.py \
  tests/test_stage1_real_synthesis.py \
  tests/test_stage1_caption_granularity.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_caption_granularity \
  tests.test_stage1_real_synthesis \
  tests.test_stage1_bvh_character_retarget \
  -v
```

Result:

- `py_compile`: passed.
- Caption/synthesis/BVH-retarget diagnostic tests: 11 tests passed in the
  combined run; synthesis-only rerun after the `then/thens` rule update also
  passed.

### Real HumanML3D caption statistics

Commands:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/diagnose_humanml3d_caption_granularity.py \
  --humanml-root ../HumanML3D/HumanML3D \
  --splits train,val,test \
  --output-json stage1_artifacts/diagnostics/humanml3d_caption_granularity.json

/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/diagnose_humanml3d_caption_granularity.py \
  --humanml-root ../HumanML3D/HumanML3D \
  --splits train,val,test \
  --max-caption-words 18 \
  --output-json stage1_artifacts/diagnostics/humanml3d_caption_granularity_max18.json
```

Without a word-count cap:

| Split | Samples | First caption non-atomic | `prefer_atomic` non-atomic | `atomic` keep rate |
|---|---:|---:|---:|---:|
| train | 23384 | 35.20% | 8.09% | 91.91% |
| val | 1460 | 34.38% | 7.53% | 92.47% |
| test | 4384 | 35.77% | 8.39% | 91.61% |

With `--max-caption-words 18`:

| Split | Samples | First caption non-atomic | `prefer_atomic` non-atomic | `atomic` keep rate |
|---|---:|---:|---:|---:|
| train | 23384 | 38.66% | 9.78% | 90.22% |
| val | 1460 | 37.67% | 9.86% | 90.14% |
| test | 4384 | 39.19% | 10.24% | 89.76% |

### Interpretation

- The previous default behavior, selecting the first caption, gives a
  multi-action caption for roughly one third of HumanML3D samples.
- `prefer_atomic` is a low-risk improvement: it keeps all samples while reducing
  non-atomic selected captions to about 8--10%.
- `atomic` with `--max-caption-words 18` is also feasible: it still keeps about
  90% of train/val clips.
- The next data experiment should build a smaller controlled dataset with:

```text
--caption-filter-mode atomic
--max-caption-words 18
--min-clips 2
--max-clips 3
```

Then it should run cache construction and token/observation diagnostics before
launching another full 20-epoch training run.  If the RVQ depth-0 distribution
remains strongly shifted after this cleanup, the limiting factor is more likely
the HumanML3D-to-MoConVQ retarget representation than caption granularity.

## 2026-06-12: Atomic-caption small data/cache diagnostic

### Purpose

This experiment tests whether fixing caption granularity alone improves the
Stage1 supervision distribution.  It intentionally uses a small dataset first,
so that we do not spend another full training run on a cache whose token
distribution is already known to be bad.

### Synthesis

Run id:

```text
atomic_caption_stage1_20260612_203030
```

Synthesis settings:

```text
caption_filter_mode = atomic
max_caption_words = 18
min_clips = 2
max_clips = 3
candidate_pool = 256
transition_max_score = 0.35
drop_overlap_frames = 1
allow_forced_transitions = false
```

Summary:

| Split | Sequences | Avg clips | Avg frames | Avg duration | Failed attempts | Forced transitions | Non-atomic clip captions |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 100 | 2.43 | 323.77 | 16.19 s | 1 | 0 | 0 |
| val | 30 | 2.57 | 368.87 | 18.44 s | 0 | 0 | 0 |

Boundary diagnostics:

| Split | Transitions | Bad transition rate | Root gap mean / p95 | Yaw gap mean / p95 | Foot velocity gap mean / p95 |
|---|---:|---:|---:|---:|---:|
| train | 143 | 1.40% | 0.0034 / 0.0143 | 0.0118 / 0.0443 | 0.0124 / 0.0519 |
| val | 47 | 4.26% | 0.0020 / 0.0123 | 0.0125 / 0.0651 | 0.0142 / 0.0511 |

Interpretation:

- Atomic caption filtering works as intended: all selected per-clip captions
  pass the heuristic.
- The boundary quality remains comparable to previous synthesis runs.  The val
  bad-transition rate is higher because the split is tiny and one short
  sequence contributes two warnings, but p95 root/yaw/foot metrics remain low.

### Observation and cache diagnostics

Observation diagnostic:

```text
stage1_artifacts/diagnostics/atomic_caption_stage1_20260612_203030/train_observation_distribution_100.json
```

Train 100-sequence observation z-scores after rest-pose calibration:

| Metric | Value |
|---|---:|
| mean abs z | 0.7451 |
| p95 abs z | 2.7831 |
| p99 abs z | 5.2472 |
| frac abs z > 5 | 1.27% |
| frac abs z > 10 | 0.119% |

Worst dimensions remain lower-leg/foot `local_rot_6d` and angular-velocity
channels.  This is the same pattern seen in the previous filtered-cache run.

Cache settings:

```text
sample_mode = segment_prefix
caption_mode = window
window_policy = clip
prefix_size = 25
forced_transition_margin = 2
rotation_calibration = rest
max_observation_p99_abs_z = 6.0
max_observation_frac_gt_5 = 0.02
max_observation_frac_gt_10 = 0.003
```

Cache summary:

| Split | Windows | Unique seqs kept | Filtered seqs | Failures | Target tokens | Non-atomic window captions |
|---|---:|---:|---:|---:|---:|---:|
| train | 403 | 78 | 22 / 100 | 0 | 9578 | 0 / 403 |
| val | 171 | 28 | 2 / 30 | 0 | 4164 | 0 / 171 |

### Token distribution

Token diagnostic:

```text
stage1_artifacts/diagnostics/atomic_caption_stage1_20260612_203030/token_distribution_atomic_val_vs_native.json
```

Comparison against `simple_motion_data.h5/walk1_subject5/observation`:

| Depth | Atomic-cache entropy | Native entropy | Atomic unique | Native unique | JS divergence |
|---|---:|---:|---:|---:|---:|
| 0 | 5.283 | 5.982 | 169 | 139 | 0.942 |
| 1 | 7.258 | 8.072 | 357 | 366 | 0.540 |
| 2 | 7.845 | 8.378 | 420 | 412 | 0.381 |
| 3 | 6.848 | 8.374 | 353 | 418 | 0.626 |

Depth-0 top-token concentration remains almost identical to the previous
HumanML3D-derived cache:

```text
Atomic HumanML3D cache depth 0:
  token 492: 22.69%
  token 338: 11.79%

Native MoConVQ depth 0:
  token 414: 6.81%
  token 272: 6.66%
```

### Interpretation

- Caption granularity is now clean in this small cache, so it is no longer the
  immediate explanation for token distribution shift.
- Atomic caption filtering does not fix the RVQ depth-0 distribution problem.
  The same token ids dominate, and JS divergence remains extremely high.
- Therefore, launching another full 20-epoch GPT training run on this cache is
  not the highest-value next step.  The next correction should target the
  HumanML3D-to-MoConVQ representation route:
  - replace or validate the hand-written joint-position-to-body-state retarget;
  - prefer a BVH/character loading path if HumanML3D/AMASS source motion can be
    converted to BVH;
  - or build a more conservative proof-of-concept from native MoConVQ
    observations before returning to HumanML3D.

### Source-motion availability check

`HumanML3D/index.csv` still records the original source paths, for example:

```text
./pose_data/KIT/3/kick_high_left02_poses.npy
./pose_data/CMU/80/80_63_poses.npy
```

However, those files are not present in the current workspace.  The available
HumanML3D files are the processed `new_joints`, `new_joint_vecs`, `texts`, and
SMPLH/DMPL body models.  Therefore, the original MoConVQ
`MotionDataSet.add_bvh_with_character()` route cannot currently be applied to
HumanML3D source motions without restoring/downloading the source pose data or
creating BVH files from `new_joints`.

Practical next options:

1. Restore HumanML3D/AMASS `pose_data`, convert source motions to BVH, and pass
   them through MoConVQ's native BVH-to-character path.
2. Keep `new_joints`, but replace the current heuristic rigid-body quaternion
   construction with a more principled IK/rest-pose fitting method before
   calling `state2ob()`.
3. For a proof-of-concept long-text pipeline, build synthetic long sequences
   from native MoConVQ observations/BVHs first, then return to HumanML3D after
   retarget quality is fixed.

## 2026-06-12: 20-epoch fit status and HumanML3D 6D rotation diagnostic

### 20-epoch fit status

Run:

```text
stage1_artifacts/checkpoints/filtered_stage1_20260612_181802/train_log.jsonl
```

The 20-epoch run does not look under-trained on the current cache.  Validation
loss reaches its best value around epoch 17, then slightly degrades while train
loss continues to decrease.

| Epoch | Train loss | Val loss | Train token acc. | Val token acc. |
|---:|---:|---:|---:|---:|
| 0 | 5.3542 | 4.5479 | 0.1238 | 0.1828 |
| 17 | 2.9831 | 3.4974 | 0.4108 | 0.3282 |
| 19 | 2.9000 | 3.5221 | 0.4253 | 0.3291 |

Interpretation:

- The model is not mainly limited by too few epochs.
- Epochs 18--19 show mild overfitting: train loss improves, but validation loss
  rises from the best epoch.
- `best_val.pth` is the correct checkpoint for comparison; `last.pth` should
  not be used as the representative model.
- Poor rollout quality is more likely caused by motion representation shift,
  data/segment alignment, and autoregressive error accumulation than by
  insufficient optimization.

### HumanML3D `new_joint_vecs` 6D rotation path

Motivation: current HumanML3D-to-MoConVQ conversion estimates MoConVQ body
rotations from `new_joints` positions.  Since HumanML3D `new_joint_vecs` stores
root yaw velocity and 21 local 6D joint rotations, we tested whether using this
rotation block can reduce MoConVQ observation outliers.

Implementation:

- Added `--rotation-source heuristic|humanml_vec6d`.
- `heuristic` preserves the previous default path.
- `humanml_vec6d` reconstructs HumanML3D global 22-joint rotations from the
  263-d representation and maps them to the 20 MoConVQ bodies.
- Unit tests verify that the 6D block can reconstruct processed
  `new_joints` with mean error below `1e-4`, so the HumanML3D layout parsing is
  correct.

Validation command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/diagnose_observation_distribution.py \
  --long-h5 stage1_artifacts/long_humanml3d_atomic_atomic_caption_stage1_20260612_203030/train/long_sequences.h5 \
  --base-data moconvq_base.data \
  --gpu 0 \
  --max-sequences 100 \
  --rotation-source humanml_vec6d \
  --rotation-calibration rest \
  --output stage1_artifacts/diagnostics/atomic_caption_stage1_20260612_203030/train_observation_distribution_humanml_vec6d_100.json
```

Observation z-score comparison on the same 100 atomic-caption train sequences:

| Rotation source | Mean abs-z | P95 abs-z | P99 abs-z | Frac > 5 | Frac > 10 | Worst p99 dim |
|---|---:|---:|---:|---:|---:|---|
| `heuristic` | 0.7451 | 2.7831 | 5.2472 | 1.267% | 0.119% | `lLowerLeg local_rot_6d[0]`, 19.01 |
| `humanml_vec6d` | 1.0776 | 4.0729 | 7.0407 | 2.939% | 0.570% | `lowerBack local_rot_6d[0]`, 19.80 |

Interpretation:

- The 263-d HumanML3D rotation block is parsed correctly, but directly mapping
  those rotations to MoConVQ rigid bodies makes the MoConVQ observation
  distribution worse.
- This means the immediate problem is not simply that the previous path ignored
  `new_joint_vecs` rotations.  The larger issue is cross-skeleton/rigid-body
  retargeting: HumanML3D joint rotations are not equivalent to MoConVQ's
  20-body simulator orientations.
- Do not rebuild the main GPT cache with `rotation_source=humanml_vec6d` unless
  a later calibration/IK step improves the observation and RVQ token
  distribution first.

### Native MoConVQ observation cache smoke

Purpose: separate GPT training-code compatibility from HumanML3D retarget
quality.  This smoke uses the original MoConVQ processed observation H5 directly
and therefore bypasses HumanML3D-to-MoConVQ state conversion.

Source:

```text
simple_motion_data.h5/walk1_subject5/observation
```

Cache command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_native_moconvq_gpt_cache.py \
  --native-h5 simple_motion_data.h5 \
  --motion 'walk1_subject5=a person walks forward' \
  --base-data moconvq_base.data \
  --text-model ../hf_models/t5-large \
  --gpu 0 \
  --window-size 50 \
  --window-stride 25 \
  --output stage1_artifacts/gpt_cache_native_smoke/train_cache.pt \
  --summary stage1_artifacts/gpt_cache_native_smoke/train_summary.json
```

Cache summary:

| Field | Value |
|---|---:|
| windows | 52 |
| latents shape | `(52, 50, 768)` |
| indices shape | `(52, 50, 4)` |
| text features shape | `(52, 256, 1024)` |
| valid tokens | 10400 |
| index range | 0..511 |
| unique source sequences | 1 |

Training smoke:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache_native_smoke/train_cache.pt \
  --val-cache stage1_artifacts/gpt_cache_native_smoke/train_cache.pt \
  --init-checkpoint text_generation_GPT.pth \
  --base-data moconvq_base.data \
  --output-dir stage1_artifacts/checkpoints/native_smoke_20260612 \
  --epochs 1 \
  --batch-size 4 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --train-scope head \
  --gpu 0 \
  --seed 0 \
  --save-every 1 \
  --num-workers 0 \
  --smoke
```

Result:

| Metric | Train | Val |
|---|---:|---:|
| loss | 8.6373 | 8.0467 |
| token accuracy | 0.0088 | 0.0075 |
| valid tokens | 800 | 800 |
| batches | 1 | 1 |

Interpretation:

- The native MoConVQ observation cache schema is compatible with the existing
  GPT training code and corrected autoregressive loss path.
- This is only a smoke test, not a quality result: it uses one walking motion
  and one batch.
- The next meaningful experiment is either:
  1. build a small native-observation proof-of-concept with multiple MoConVQ
     motions/captions to verify long-text segmentation and rollout without
     HumanML3D retarget noise; or
  2. restore HumanML3D/AMASS source `pose_data`/BVH and run MoConVQ's native
     `MotionDataSet.add_bvh_with_character()` path instead of hand-written
     position-to-state retargeting.

### BVH-to-character cache path added

To make option 2 concrete, added a cache builder that uses the original MoConVQ
BVH loading route:

```text
Script/stage1/build_bvh_character_gpt_cache.py
```

The route is:

```text
BVH file
  -> MotionDataSet.add_bvh_with_character()
  -> MoConVQ simulator character state/observation
  -> agent.encode_seq_all()
  -> latent_vq + RVQ indices + T5 caption feature cache
```

This is intentionally different from the current HumanML3D `new_joints` route:
it lets VclSimuBackend/MoConVQ perform the character retargeting instead of
hard-coding a HumanML3D joint-to-20-body mapping.  Once HumanML3D/AMASS source
motions can be restored or exported as BVH, this script is the preferred path
for rebuilding the Stage1 GPT cache.

Example command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --bvh 'path/to/motion.bvh=a person walks forward' \
  --base-data moconvq_base.data \
  --text-model ../hf_models/t5-large \
  --gpu 0 \
  --window-size 50 \
  --window-stride 25 \
  --output stage1_artifacts/gpt_cache_bvh_character/train_cache.pt \
  --observation-h5 stage1_artifacts/gpt_cache_bvh_character/source_observation.h5 \
  --summary stage1_artifacts/gpt_cache_bvh_character/train_summary.json
```

Tested:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_bvh_character_cache \
  tests.test_stage1_native_cache \
  tests.test_stage1_bvh_character_retarget \
  -v
```

Result: 6 tests passed.

## 2026-06-13: LLM in-context token planning backup path implemented

### Purpose

The current HumanML3D-derived GPT fine-tuning route has not yet produced a
clear rollout-quality win over the baseline.  Diagnostics point to a
representation problem: the hand-written HumanML3D-to-MoConVQ retarget path
produces shifted RVQ token distributions, especially at depth 0, and the
available local BVH files are too few for a full native BVH-to-character
retraining run.  To keep Stage1 moving toward a complete long-horizon
language-to-motion pipeline, this change starts engineering the MoConVQ paper's
LLM in-context integration backup route.

### Code changes

Added:

```text
Script/stage1/llm_token_planning.py
tests/test_stage1_llm_token_planning.py
tests/test_stage1_repository_hygiene.py
```

The new script provides one unified entry point with subcommands:

```text
export-bank      export caption -> 4-depth RVQ token examples from a GPT cache
retrieve         retrieve examples for a query segment
build-prompt     build a JSON-only in-context prompt for an external LLM
validate         parse and validate an LLM token response
retrieval-plan   deterministic retrieval-only token baseline, no LLM API needed
decode-bvh       decode validated RVQ tokens through MoConVQ into BVH
```

Repository hygiene:

- Added `.gitignore` patterns for `AGENT.md`, `AGENTS.md`, `CODEX.md`,
  `CLAUDE.md`, `.codex/`, and `.claude/`.
- Added a test that fails if agent/private assistant docs are tracked in the
  `MoConVQ` repository.

### Smoke run

Run id:

```text
llm_backup_smoke_20260613
```

Commands:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/llm_token_planning.py export-bank \
  --cache stage1_artifacts/gpt_cache_filtered_cache_stage1_20260612_174908/train_cache.pt \
  --output stage1_artifacts/llm_backup/example_bank_filtered_200.jsonl \
  --max-examples 200 \
  --max-tokens-per-example 32 \
  --min-tokens-per-example 8

/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/llm_token_planning.py build-prompt \
  --bank stage1_artifacts/llm_backup/example_bank_filtered_200.jsonl \
  --text 'a person walks forward then kicks with the right foot then dances' \
  --top-k 3 \
  --segment-token-count 12 \
  --max-tokens-per-example 12 \
  --output-prompt stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/prompt.txt \
  --output-json stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/retrieval.json

/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/llm_token_planning.py retrieval-plan \
  --bank stage1_artifacts/llm_backup/example_bank_filtered_200.jsonl \
  --text 'a person walks forward then kicks with the right foot then dances' \
  --top-k 3 \
  --segment-token-count 12 \
  --output-tokens stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/retrieval_tokens.json \
  --validation-json stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/retrieval_validation.json

/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/llm_token_planning.py decode-bvh \
  --tokens stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/retrieval_tokens.json \
  --base-data moconvq_base.data \
  --gpu 0 \
  --output-bvh stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/retrieval_output.bvh

/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/evaluate_bvh_metrics.py \
  'stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/retrieval_output.bvh' \
  --sample-stride 6 \
  --lags 5,10,20,30 \
  --expected-min-frames 600 \
  --output stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/retrieval_bvh_metrics.json
```

Result:

| Item | Value |
|---|---:|
| exported examples | 200 |
| retrieval-only RVQ tuples | 36 |
| token validation | passed |
| decoded BVH frames | 864 |
| duration | 7.20 s |
| early stop threshold | 600 frames |
| early stop | false |
| root path | 1.739 |
| root displacement | 0.279 |
| pose velocity mean | 11.362 |
| pose variance mean | 89.864 |
| lag-20 repeat > 0.995 | 0.00% |

Interpretation:

- The LLM backup route is now a working minimal engineering loop:

```text
GPT cache -> example bank -> retrieval/prompt -> validated RVQ tokens
-> MoConVQ decoder/controller -> BVH -> engineering metrics
```

- This is not yet an LLM semantic-quality result.  The smoke used
  `retrieval-plan`, which copies/repeats retrieved examples as a deterministic
  lower bound and API-free sanity check.
- The next backup experiment should take the generated `prompt.txt`, obtain an
  actual LLM JSON response, validate it, decode it to BVH, render side-by-side
  videos against baseline/finetuned GPT, and record the same engineering
  metrics.

### Verification

Commands run:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/llm_token_planning.py \
  tests/test_stage1_llm_token_planning.py \
  tests/test_stage1_repository_hygiene.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_llm_token_planning \
  tests.test_stage1_repository_hygiene \
  -v
```

Result:

- `py_compile`: passed.
- LLM token planning + repository hygiene tests: 7 tests passed.

### BVH-to-character cache smoke on existing BVH files

Purpose: verify that the new BVH-character cache path can run end-to-end on
actual local BVH files and produce a `train_real_text_gpt.py` compatible cache.

Commands:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --bvh 'base.bvh=a person walks forward' \
  --base-data moconvq_base.data \
  --text-model ../hf_models/t5-large \
  --gpu 0 \
  --window-size 50 \
  --window-stride 25 \
  --output stage1_artifacts/gpt_cache_bvh_smoke/train_cache.pt \
  --observation-h5 stage1_artifacts/gpt_cache_bvh_smoke/source_observation.h5 \
  --summary stage1_artifacts/gpt_cache_bvh_smoke/train_summary.json

/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --bvh 'track.bvh=a person walks and changes direction' \
  --base-data moconvq_base.data \
  --text-model ../hf_models/t5-large \
  --gpu 0 \
  --window-size 50 \
  --window-stride 25 \
  --output stage1_artifacts/gpt_cache_bvh_smoke/track_cache.pt \
  --observation-h5 stage1_artifacts/gpt_cache_bvh_smoke/track_observation.h5 \
  --summary stage1_artifacts/gpt_cache_bvh_smoke/track_summary.json
```

Cache summaries:

| Source BVH | Windows | Latents shape | Indices shape | Valid tokens | Index range |
|---|---:|---|---|---:|---|
| `base.bvh` | 1 | `(1, 50, 768)` | `(1, 50, 4)` | 16 | 27..489 |
| `track.bvh` | 4 | `(4, 50, 768)` | `(4, 50, 4)` | 800 | 3..511 |

Token distribution diagnostic:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/diagnose_token_distribution.py \
  --cache stage1_artifacts/gpt_cache_bvh_smoke/track_cache.pt \
  --native-h5 simple_motion_data.h5 \
  --native-observation-key walk1_subject5/observation \
  --base-data moconvq_base.data \
  --gpu 0 \
  --output-json stage1_artifacts/diagnostics/bvh_character_track_vs_native_tokens.json
```

Comparison against `simple_motion_data.h5/walk1_subject5/observation`:

| Depth | BVH cache entropy | Native entropy | BVH unique | Native unique | JS divergence |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.376 | 5.982 | 59 | 139 | 0.944 |
| 1 | 6.270 | 8.072 | 91 | 366 | 0.787 |
| 2 | 6.454 | 8.378 | 96 | 412 | 0.687 |
| 3 | 6.439 | 8.374 | 98 | 418 | 0.698 |

The JS divergence is still high, but this is expected for a 200-token cache
from a single `track.bvh` sample and should not be over-interpreted.  Unlike the
HumanML3D-derived cache, the top depth-0 token is not extremely dominant:

```text
track.bvh depth 0 top fractions:
  0.090, 0.075, 0.055, 0.055, 0.050
```

Training smoke:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache_bvh_smoke/track_cache.pt \
  --val-cache stage1_artifacts/gpt_cache_bvh_smoke/track_cache.pt \
  --init-checkpoint text_generation_GPT.pth \
  --base-data moconvq_base.data \
  --output-dir stage1_artifacts/checkpoints/bvh_smoke_20260612 \
  --epochs 1 \
  --batch-size 2 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --train-scope head \
  --gpu 0 \
  --seed 0 \
  --save-every 1 \
  --num-workers 0 \
  --smoke
```

Result:

| Metric | Train | Val |
|---|---:|---:|
| loss | 7.6435 | 7.8137 |
| token accuracy | 0.0050 | 0.0175 |
| valid tokens | 400 | 400 |
| batches | 1 | 1 |

Interpretation:

- The original MoConVQ BVH-to-character route can produce a GPT training cache
  and the current training code can consume it.
- The available local BVH files are too few for a meaningful fine-tuning
  experiment.  `base.bvh` is especially short; `track.bvh` gives only four
  windows.
- This validates the engineering path for the retarget fix, but the next
  required data step is still to restore/export a larger HumanML3D/AMASS BVH
  source set and rebuild cache through this route.

## 2026-06-12: Evaluation readiness audit

### Paper-level metrics from MoConVQ

The MoConVQ paper evaluates Text2Motion-MoConGPT on the HumanML3D test set
using FID and R-precision.  It explicitly states that the evaluation follows
the HumanML3D text-to-motion protocol and uses the same pretrained motion
feature extractor as prior HumanML3D methods.  The paper also notes that
retargeting between the simulated character and SMPL can reduce R-precision,
which is directly relevant to the current Stage1 retarget bottleneck.

Therefore, the final "better than baseline" claim should ideally include:

- FID on HumanML3D/SMPL-style motion features;
- R-precision for text-motion semantic retrieval;
- rendered videos and engineering diagnostics as supporting evidence.

### Current repository readiness

Added:

```text
Script/stage1/check_evaluation_readiness.py
tests/test_stage1_evaluation_readiness.py
```

Audit command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/check_evaluation_readiness.py \
  --repo-root . \
  --humanml-root ../HumanML3D \
  --output stage1_artifacts/diagnostics/evaluation_readiness_20260612.json
```

Result:

| Item | Status |
|---|---|
| HumanML3D evaluator source files | missing |
| Pretrained HumanML3D evaluator / motion-feature extractor checkpoints | missing |
| Paper-level FID/R-precision ready | no |
| BVH engineering metrics script | available |
| baseline-vs-finetuned comparison script | available |
| token distribution diagnostic | available |
| observation z-score diagnostic | available |

Current recommendation:

```text
Use engineering diagnostics only as intermediate checks; do not claim
paper-level improvement over baseline until FID/R-precision evaluator assets
are available.
```

Verification:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_evaluation_readiness \
  tests.test_stage1_bvh_metrics \
  tests.test_stage1_text_gpt_comparison \
  -v
```

Result: 6 tests passed.

## 2026-06-13: Stage1 mainline diagnostics and training/generation protocol synced

### Purpose

The previous GitHub `stage1` branch had the LLM token-planning backup path, but
many mainline Stage1 diagnostics and protocol fixes were still local.  This
sync packages the mainline capabilities needed to reproduce and explain the
current Stage1 diagnosis:

```text
HumanML3D synthesis/caption filtering
-> calibrated HumanML3D-to-MoConVQ cache construction
-> segment-prefix/progress-conditioned GPT training
-> top-p and segmented generation
-> BVH/native-observation retarget diagnostics
-> paper-metric readiness audit
-> engineering BVH metrics and comparison summaries
```

This is still not a claim that Stage1 is complete or that the fine-tuned model
beats baseline.  It makes the current evidence and reproduction tooling
available from the remote branch.

### Code included

Updated or added:

```text
MoConVQCore/Model/cross_trans_ori_fixsum.py
Script/stage1/real_moconvq_cache.py
Script/stage1/train_real_text_gpt.py
Script/stage1/generate_long_motion.py
Script/stage1/synthesize_long_humanml3d.py
Script/stage1/build_bvh_character_gpt_cache.py
Script/stage1/build_native_moconvq_gpt_cache.py
Script/stage1/diagnose_observation_distribution.py
Script/stage1/diagnose_token_distribution.py
Script/stage1/diagnose_bvh_character_retarget.py
Script/stage1/diagnose_humanml3d_caption_granularity.py
Script/stage1/diagnose_long_humanml3d_quality.py
Script/stage1/evaluate_bvh_metrics.py
Script/stage1/check_evaluation_readiness.py
Script/stage1/run_text_gpt_comparison.py
Script/stage1/summarize_bvh_comparison.py
Script/stage1/segment_conditioning.py
Script/stage1/plot_train_curves.py
Script/stage1/export_baseline_intermediate.py
Script/stage1/intermediate_motion_format.py
```

Key protocol fixes now present in the pushed code:

- GPT training reconstructs context latents from RVQ codebook indices, matching
  rollout-time feedback instead of using encoder `latent_vq` directly.
- `target_masks` exclude prefix/context-only tokens from CE, KL, and accuracy.
- End-token auxiliary loss is applied once at RVQ depth 0.
- Segment-progress conditioning enters through the existing `clip_feature`
  pathway and can be disabled for the baseline/teacher.
- HumanML3D retarget defaults to rest-pose rotation calibration and records
  retarget config in cache metadata.
- Cache construction supports observation z-score filtering.
- HumanML3D synthesis supports atomic caption filtering and overlap-frame
  dropping.
- Inference supports top-p/top-k/temperature and segmented generation without
  treating segment early-stop as whole-prompt early-stop.

### BVH metrics robustness fix

While smoke-testing `evaluate_bvh_metrics.py` on the repository BVHs, `base.bvh`
reported `Frames: 100` but contained 110 motion rows.  The metric tool now
accepts BVH files with extra rows by trimming to the header frame count, while
still rejecting files with fewer rows than declared.  This makes the engineering
metric script usable on the existing local BVH references.

Smoke command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/evaluate_bvh_metrics.py base.bvh track.bvh \
  --sample-stride 6 \
  --lags 5,10 \
  --expected-min-frames 600 \
  --output /tmp/stage1_bvh_metrics_sync_check.json
```

Selected result:

| BVH | Frames | Duration | Early stop | Root path | Pose velocity mean |
|---|---:|---:|---|---:|---:|
| `base.bvh` | 100 | 0.83 s | yes | 0.102 | 12.665 |
| `track.bvh` | 2904 | 24.20 s | no | 9.774 | 26.801 |

### Evaluation readiness smoke

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/check_evaluation_readiness.py \
  --repo-root . \
  --humanml-root ../HumanML3D \
  --output /tmp/stage1_eval_readiness_sync_check.json
```

Result:

```text
paper_metrics_ready = false
missing:
  - HumanML3D text-motion evaluator source files
  - pretrained HumanML3D evaluator / motion-feature extractor checkpoints
engineering tools available:
  - BVH metrics
  - baseline-vs-finetuned comparison
  - token distribution diagnostic
  - observation distribution diagnostic
```

### Verification

Commands run in a clean worktree based on `origin/stage1`:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  MoConVQCore/Model/cross_trans_ori_fixsum.py \
  Script/stage1/real_moconvq_cache.py \
  Script/stage1/train_real_text_gpt.py \
  Script/stage1/generate_long_motion.py \
  Script/stage1/synthesize_long_humanml3d.py \
  Script/stage1/evaluate_bvh_metrics.py \
  Script/stage1/check_evaluation_readiness.py \
  Script/stage1/llm_token_planning.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_gpt \
  tests.test_stage1_real_cache \
  tests.test_stage1_real_train \
  tests.test_stage1_real_generate \
  tests.test_stage1_real_synthesis \
  tests.test_stage1_caption_granularity \
  tests.test_stage1_observation_diagnostics \
  tests.test_stage1_bvh_metrics \
  tests.test_stage1_evaluation_readiness \
  tests.test_stage1_llm_token_planning \
  tests.test_stage1_repository_hygiene \
  tests.test_stage1_bvh_character_cache \
  tests.test_stage1_bvh_character_retarget \
  tests.test_stage1_bvh_comparison_summary \
  tests.test_stage1_intermediate_export \
  tests.test_stage1_native_cache \
  tests.test_stage1_plot_train_curves \
  tests.test_stage1_render_bvh \
  tests.test_stage1_text_gpt_comparison \
  -v
```

Result:

```text
py_compile passed
92 tests passed, 1 skipped
tracked agent/Codex private doc check: empty
```

### Next step

With both mainline diagnostics and the LLM backup tool now available from the
remote branch, the next Stage1 experiment should be a real comparison package:

1. Choose the current best mainline checkpoint and baseline checkpoint.
2. Generate the same prompt set with baseline, fine-tuned GPT, retrieval-only
   backup, and an actual LLM-token response if available.
3. Render side-by-side videos and run BVH engineering metrics.
4. Record a semantic checklist per prompt.
5. If HumanML3D evaluator assets become available, add FID/R-precision before
   making the final paper-level claim.

## 2026-06-13: Unified Stage1 model-suite comparison runner

### Motivation

The Stage1 goal now requires a final comparison package, not only isolated
single-route smoke tests.  Previous runs compared baseline-vs-finetuned GPT in
one script and tested the LLM-token backup in a separate script, which made it
easy to use different prompt sets, decoding settings, or metric commands.

This update adds a unified runner:

```text
Script/stage1/run_stage1_model_suite.py
tests/test_stage1_model_suite.py
```

The suite creates one `stage1_artifacts/model_suite/<run_id>/` directory with:

```text
prompts.tsv
bvh/<prompt>__baseline_top_p.bvh
bvh/<prompt>__finetuned_top_p.bvh
bvh/<prompt>__backup_retrieval.bvh
bvh/<prompt>__backup_llm.bvh        # optional, only when an LLM response is supplied
summary_metrics.json
suite_summary.json
llm_backup/<prompt>/prompt.txt
llm_backup/<prompt>/retrieval_tokens.json
llm_backup/<prompt>/retrieval_validation.json
```

Default prompts:

```text
walk_turn_wave        a person walks forward then turns around then waves both arms
circle_crouch_stand   a person walks in a circle then crouches down then stands up
walk_jump_dance       a person walks forward then jumps then dances
sidestep_kick_turn    a person sidesteps to the left then kicks with the right foot then turns around
```

The runner intentionally keeps the current metric scope explicit:

- `summary_metrics.json` is produced with `evaluate_bvh_metrics.py`.
- Metrics are BVH engineering diagnostics: frames, duration, early-stop flag,
  root path/displacement, pose velocity/variance, and lagged repeat proxies.
- These metrics still do not replace HumanML3D FID/R-precision.

### Command template for the next real suite

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python Script/stage1/run_stage1_model_suite.py \
  --run-id suite_filtered_stage1_20260613 \
  --finetuned-checkpoint stage1_artifacts/checkpoints/filtered_stage1_20260612_181802/best_val.pth \
  --backup-cache stage1_artifacts/gpt_cache_filtered_cache_stage1_20260612_174908/train_cache.pt \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --max-length 120 \
  --context-size 30 \
  --chunk-size 20 \
  --top-p 0.95 \
  --progress-scale 0.5 \
  --progress-context-size 51 \
  --progress-prefix-cap 25 \
  --expected-min-frames 1200 \
  --gpu 0
```

For an actual LLM backup result, save the external LLM response JSON and pass:

```bash
--llm-response-map stage1_artifacts/model_suite/<run_id>/llm_responses.json
```

where the map is a JSON object from prompt name to response file.  Relative
response paths are resolved relative to the map file.

### Implementation notes

- `--skip-backup` now fully skips backup setup, including example-bank export.
- `--llm-response-map` can be used without a retrieval bank, so actual LLM
  token responses can be validated and decoded independently.
- Retrieval-only backup plans now support `--trim-repeat-runs`.  This does not
  claim semantic improvement; it prevents copied retrieval snippets from
  producing invalid long runs of identical RVQ tuples before BVH decoding.
- `decode-bvh` now accepts `--motion-dataset`, and the suite records the
  resolved path in `suite_summary.json`.  This avoids depending on
  `./simple_motion_data.h5` relative to the caller's current directory.
- `llm_token_planning.py` now inserts its own repo root into `sys.path` when run
  as a script, preventing a temporary worktree from importing stale modules
  from another checkout.
- Default prompt generation, response-map parsing, backup command wiring,
  repeat-run trimming, and repository hygiene are covered by tests.

### Smoke: existing filtered BVH summary

The suite was first run in `--skip-generation --skip-backup` mode on an existing
baseline-vs-finetuned directory:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python Script/stage1/run_stage1_model_suite.py \
  --run-id suite_existing_bvh_smoke_20260613 \
  --suite-dir stage1_artifacts/model_suite/suite_existing_bvh_smoke_20260613 \
  --bvh-dir /home/chenjie/cc/robotics/MoConVQ/stage1_artifacts/generated_bvh_compare/filtered_stage1_20260612_181802_top_p_len120_scale05 \
  --finetuned-checkpoint stage1_artifacts/checkpoints/filtered_stage1_20260612_181802/best_val.pth \
  --skip-generation \
  --skip-backup \
  --expected-min-frames 1200
```

Result:

```text
baseline_top_p:
  avg_frames = 1500.0
  avg_duration_sec = 12.4995
  avg_root_path_length = 4.0237
  avg_root_displacement = 1.4157
  avg_pose_velocity_mean = 17.7805
  avg_pose_variance_mean = 152.2556
  avg_lag_20_repeat_fraction_0.995 = 0.0
  early_stop_rate = 0.0

finetuned_top_p:
  avg_frames = 1722.0
  avg_duration_sec = 14.3494
  avg_root_path_length = 4.8438
  avg_root_displacement = 1.5576
  avg_pose_velocity_mean = 25.0650
  avg_pose_variance_mean = 307.9503
  avg_lag_20_repeat_fraction_0.995 = 0.0
  early_stop_rate = 0.25
```

Interpretation:

- The suite can summarize an existing comparison directory without regenerating
  BVHs.
- This is not a success claim for the fine-tuned GPT.  The fine-tuned rollout is
  longer on average, but it also has higher pose velocity/variance and one of
  four prompts is still below the 1200-frame threshold.

### Smoke: retrieval-only backup suite

Initial failure:

```text
retrieval_validation.ok = false
repeat_violations:
  start=0, length=8
  start=39, length=9
```

This showed that deterministic retrieval-only token plans can copy a snippet
with long identical-tuple runs.  The suite now enables repeat-run trimming for
retrieval-only backup by default and records the repair count in each
`retrieval_validation.json`.

Second failure:

```text
FileNotFoundError: ./simple_motion_data.h5
```

This exposed an implicit current-directory dependency in the MoConVQ agent
builder.  The backup decoder now accepts `--motion-dataset`, and the suite
passes the resolved dataset path explicitly.

Third failure:

```text
TypeError: build_loaded_moconvq_agent() got an unexpected keyword argument 'motion_dataset'
```

This was caused by direct script execution from the temporary worktree importing
`Script.stage1.real_moconvq_cache` from another checkout.  The script now
preprends its own repo root to `sys.path`.

Final smoke command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python Script/stage1/run_stage1_model_suite.py \
  --run-id suite_backup_retrieval_smoke_20260613 \
  --suite-dir stage1_artifacts/model_suite/suite_backup_retrieval_smoke_20260613 \
  --skip-gpt \
  --finetuned-checkpoint stage1_artifacts/checkpoints/filtered_stage1_20260612_181802/best_val.pth \
  --backup-cache /home/chenjie/cc/robotics/MoConVQ/stage1_artifacts/gpt_cache_filtered_cache_stage1_20260612_174908/train_cache.pt \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --backup-max-examples 40 \
  --backup-max-tokens-per-example 24 \
  --backup-min-tokens-per-example 8 \
  --backup-top-k 3 \
  --backup-segment-token-count 18 \
  --expected-min-frames 1200 \
  --gpu 0
```

Result:

```text
example_bank examples_written = 40

walk_turn_wave:
  tokens = 47, repeat_repairs = 7, frames = 1128
circle_crouch_stand:
  tokens = 50, repeat_repairs = 4, frames = 1200
walk_jump_dance:
  tokens = 51, repeat_repairs = 3, frames = 1224
sidestep_kick_turn:
  tokens = 54, repeat_repairs = 0, frames = 1296

backup_retrieval model averages:
  avg_frames = 1212.0
  avg_duration_sec = 10.0996
  avg_root_path_length = 2.5058
  avg_root_displacement = 0.6933
  avg_pose_velocity_mean = 20.8206
  avg_pose_variance_mean = 192.8391
  avg_lag_20_repeat_fraction_0.995 = 0.0
  early_stop_rate = 0.25
```

Interpretation:

- The LLM-token backup engineering route can now enter the same suite format as
  GPT baseline and fine-tuned generation.
- This smoke used retrieval-only planning, not an actual external LLM response.
  It is therefore a deterministic lower bound and decoder validation, not a
  semantic-quality claim.
- The next useful backup experiment is to use the generated `prompt.txt` files
  with an actual LLM, validate the JSON responses through the suite, render the
  resulting BVHs, and compare semantic completion prompt by prompt.

### Verification

Commands run in this clean worktree:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/run_stage1_model_suite.py \
  Script/stage1/llm_token_planning.py \
  Script/stage1/real_moconvq_cache.py \
  tests/test_stage1_model_suite.py \
  tests/test_stage1_llm_token_planning.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_model_suite \
  tests.test_stage1_llm_token_planning \
  tests.test_stage1_text_gpt_comparison \
  tests.test_stage1_bvh_metrics \
  tests.test_stage1_repository_hygiene \
  tests.test_stage1_real_cache \
  -v
```

Result:

```text
py_compile passed
35 tests passed, 1 skipped
```

## 2026-06-13: Goal and repository-target correction

### Repository target correction

The Stage1 code must be pushed to the `stage1/` folder on the remote `main`
branch:

```text
origin/main:stage1/
```

It should not be treated as work whose final destination is the remote
`stage1` branch.  A previous sync mistakenly pushed the same Stage1 suite work
to `origin/stage1`; the corrected commit is now on `origin/main`:

```text
8f84781 Sync Stage1 pipeline tools into main stage1 folder
```

All future Stage1 commits should use the `main` branch worktree and keep changes
scoped to `stage1/`.

### LLM in-context status

No actual external LLM in-context experiment has been run yet.

Current implemented backup tooling:

```text
GPT cache -> caption/RVQ-token example bank
example bank -> in-context prompt.txt
external LLM JSON response -> validator -> RVQ tokens -> BVH
```

Current executed backup experiments:

```text
retrieval-only token planning lower bound
```

The retrieval-only route does not call ChatGPT, Claude, an OpenAI API, a local
LLM, or the coding assistant.  It only copies/recombines retrieved MoConVQ RVQ
token examples to verify the backup token-to-BVH and metric pipeline.  A real
LLM experiment must explicitly record:

```text
model name
prompt file
raw response
validated tokens
BVH output
metrics
manual semantic checklist
```

### HumanML3D mainline status

The HumanML3D data route has not been abandoned.  The current diagnosis is that
the hand-written HumanML3D-to-MoConVQ retarget/cache path remains unreliable for
final claims.  The mainline fix should continue through a more faithful
MoConVQ-native retarget path:

```text
HumanML3D / AMASS source motion
-> restore/export BVH or compatible source motion
-> MotionDataSet.add_bvh_with_character()
-> MoConVQ observation/latent/RVQ cache
-> text-conditioned MoConGPT fine-tuning
-> baseline-vs-finetuned comparison
```

The LLM in-context token-planning route is therefore a backup, not a replacement
for the HumanML3D/BVH retarget mainline.

## 2026-06-13: Full unified Stage1 suite smoke on main/stage1

### Direct script import fix

While running the full suite from the clean `main` worktree, direct execution of
`Script/stage1/generate_long_motion.py` could import `real_moconvq_cache.py`
from a different checkout when multiple MoConVQ worktrees were present on the
same machine.  This happened because Python put the script directory on
`sys.path`, but not the owning repository root.

Fix:

```text
generate_long_motion.py now inserts its own repo root at sys.path[0] when it is
executed as a script.
```

Regression coverage:

```text
tests.test_stage1_real_generate.Stage1RealGenerateTests
  .test_direct_script_execution_prefers_own_repo_root
```

The test simulates a conflicting checkout path and verifies that the script path
guard restores the owning repository root before stage1 imports resolve.

### Suite command

Executed from:

```text
/home/chenjie/cc/robotics/MoConVQ-main/stage1
```

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python Script/stage1/run_stage1_model_suite.py \
  --run-id suite_full_filtered_stage1_20260613_len75_main \
  --suite-dir stage1_artifacts/model_suite/suite_full_filtered_stage1_20260613_len75_main \
  --baseline-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --finetuned-checkpoint /home/chenjie/cc/robotics/MoConVQ/stage1_artifacts/checkpoints/filtered_stage1_20260612_181802/best_val.pth \
  --backup-cache /home/chenjie/cc/robotics/MoConVQ/stage1_artifacts/gpt_cache_filtered_cache_stage1_20260612_174908/train_cache.pt \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --max-length 75 \
  --context-size 30 \
  --chunk-size 20 \
  --top-p 0.95 \
  --progress-scale 0.5 \
  --progress-context-size 51 \
  --progress-prefix-cap 25 \
  --backup-max-examples 120 \
  --backup-max-tokens-per-example 24 \
  --backup-min-tokens-per-example 8 \
  --backup-top-k 3 \
  --backup-segment-token-count 18 \
  --expected-min-frames 1200 \
  --gpu 0
```

Output directory:

```text
stage1_artifacts/model_suite/suite_full_filtered_stage1_20260613_len75_main/
```

Important files:

```text
suite_summary.json
summary_metrics.json
bvh/*.bvh
llm_backup/*/prompt.txt
llm_backup/*/retrieval_tokens.json
llm_backup/*/retrieval_validation.json
```

The `stage1_artifacts/` outputs are ignored artifacts and are not intended to be
committed.

### Model averages

All metrics below are BVH engineering diagnostics, not paper-level
FID/R-precision.

| model | avg frames | avg sec | avg root path | avg root disp | avg pose vel | avg pose var | lag20 repeat >=0.995 | early stop rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_top_p | 1062.0 | 8.849646 | 3.471612 | 1.268157 | 14.052114 | 141.193729 | 0.000000 | 0.75 |
| finetuned_top_p | 1272.0 | 10.599576 | 3.518982 | 1.570334 | 30.893359 | 326.253117 | 0.000000 | 0.25 |
| backup_retrieval | 1188.0 | 9.899604 | 3.118687 | 1.200555 | 13.664251 | 148.864182 | 0.000000 | 0.50 |

### Per-prompt diagnostics

Expected minimum length was 1200 frames.

| prompt | model | frames | sec | root path | root disp | pose vel | pose var | lag20 repeat >=0.995 | early stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| circle_crouch_stand | baseline_top_p | 1176 | 9.799608 | 5.771557 | 1.235966 | 31.519118 | 417.674809 | 0.000000 | true |
| circle_crouch_stand | finetuned_top_p | 1296 | 10.799568 | 4.091763 | 2.813482 | 59.866482 | 737.274795 | 0.000000 | false |
| circle_crouch_stand | backup_retrieval | 1296 | 10.799568 | 4.307908 | 1.605424 | 18.514475 | 389.838804 | 0.000000 | false |
| sidestep_kick_turn | baseline_top_p | 864 | 7.199712 | 0.781742 | 0.561169 | 3.014137 | 5.735598 | 0.000000 | true |
| sidestep_kick_turn | finetuned_top_p | 744 | 6.199752 | 1.837311 | 1.248592 | 11.702774 | 33.771956 | 0.000000 | true |
| sidestep_kick_turn | backup_retrieval | 1296 | 10.799568 | 3.557276 | 1.227494 | 13.027709 | 44.542508 | 0.000000 | false |
| walk_jump_dance | baseline_top_p | 1392 | 11.599536 | 6.049767 | 2.113611 | 16.527376 | 129.777956 | 0.000000 | false |
| walk_jump_dance | finetuned_top_p | 1800 | 14.999400 | 5.463325 | 1.216706 | 39.959676 | 394.484026 | 0.000000 | false |
| walk_jump_dance | backup_retrieval | 1128 | 9.399624 | 2.043694 | 0.330492 | 10.483378 | 98.112158 | 0.000000 | true |
| walk_turn_wave | baseline_top_p | 816 | 6.799728 | 1.283382 | 1.161882 | 5.147825 | 11.586552 | 0.000000 | true |
| walk_turn_wave | finetuned_top_p | 1248 | 10.399584 | 2.683528 | 1.002555 | 12.044505 | 139.481690 | 0.000000 | false |
| walk_turn_wave | backup_retrieval | 1032 | 8.599656 | 2.565869 | 1.638812 | 12.631442 | 62.963258 | 0.000000 | true |

### Interpretation

- The fine-tuned checkpoint reduces the early-stop rate from `0.75` to `0.25`
  and increases average length from `1062.0` to `1272.0` frames.
- The fine-tuned checkpoint also increases average root displacement from
  `1.268157` to `1.570334`, which is a useful movement sanity check.
- The fine-tuned checkpoint has much higher pose velocity and pose variance than
  baseline.  This may indicate richer motion, instability, or both; video review
  is required before making a qualitative claim.
- The fine-tuned checkpoint is not uniformly better.  On `sidestep_kick_turn`,
  it stops earlier than baseline (`744` vs `864` frames), so this suite is a
  controlled smoke result, not a final success claim.
- `backup_retrieval` is still a retrieval-only lower bound.  It proves that the
  backup token-planning decode and metric path works, but it is not an actual
  LLM in-context planning result.
- Paper-level FID and R-precision remain unavailable until the HumanML3D
  evaluator source/checkpoints are restored.

### Next evidence needed

- Render the twelve BVH outputs to videos and complete a manual semantic
  checklist for each prompt/model pair.
- Continue the main HumanML3D/AMASS/BVH route through MoConVQ-native character
  retarget, because the current fine-tuned checkpoint was trained from the
  filtered stage1 cache rather than a fully repaired large HumanML3D native
  retarget cache.
- Run a real LLM in-context token-planning experiment only if the backup route
  is needed, and record the exact model, prompt, raw response, validation output,
  decoded BVH, metrics, and semantic checklist.

### Verification

Commands run with the `moconvq` conda environment:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/run_stage1_model_suite.py \
  Script/stage1/llm_token_planning.py \
  Script/stage1/real_moconvq_cache.py \
  Script/stage1/generate_long_motion.py \
  tests/test_stage1_model_suite.py \
  tests/test_stage1_llm_token_planning.py \
  tests/test_stage1_real_generate.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_model_suite \
  tests.test_stage1_llm_token_planning \
  tests.test_stage1_text_gpt_comparison \
  tests.test_stage1_bvh_metrics \
  tests.test_stage1_repository_hygiene \
  tests.test_stage1_real_cache \
  tests.test_stage1_real_generate \
  -v
```

Result:

```text
py_compile passed
47 tests passed, 1 skipped
```

## 2026-06-13: Stage1 HumanML3D/source-motion readiness audit

### Why this audit was added

The Stage1 mainline should remain:

```text
HumanML3D / AMASS source motion
-> restore/export BVH or compatible source motion
-> MotionDataSet.add_bvh_with_character()
-> MoConVQ observation/latent/RVQ cache
-> text-conditioned MoConGPT fine-tuning
```

To keep that distinction reproducible, I added a data-readiness checker:

```text
Script/stage1/check_stage1_data_readiness.py
tests/test_stage1_data_readiness.py
```

The checker separates three states:

```text
1. canonical processed HumanML3D payload is available;
2. original HumanML3D/AMASS source motion or BVH exports are available;
3. MoConVQ-native BVH-to-character cache tools are available.
```

This matters because the processed HumanML3D payload (`new_joints`,
`new_joint_vecs`, `texts`) can be complete while the source motion needed for
native BVH retarget is still missing.

### Command

Executed from:

```text
/home/chenjie/cc/robotics/MoConVQ-main/stage1
```

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/check_stage1_data_readiness.py \
  --repo-root . \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --output stage1_artifacts/diagnostics/data_readiness_20260613_main.json
```

### Result

Current local dataset state:

```text
processed_corpus.ready = true
processed_corpus.indexed_payload_complete = true
all.txt = 29228
texts = 29232
new_joints = 29228
new_joint_vecs = 29228
train = 23384
val = 1460
test = 4384
train_val = 24844
```

The `texts` directory has four extra text files, but every sample in
`all.txt` has a matching `texts/*.txt`, `new_joints/*.npy`, and
`new_joint_vecs/*.npy`.  This is therefore a complete canonical processed
HumanML3D payload under the `all.txt` index.

Current source-motion state:

```text
index.csv rows = 14616
existing index source files = 0
missing index source files = 14616
BVH files under /home/chenjie/cc/robotics/HumanML3D = 0
NPZ files under /home/chenjie/cc/robotics/HumanML3D = 6
standard AMASS motion NPZ files detected = 0
source_motion_available_for_export = false
native_bvh_cache_ready = false
stage1_mainline_ready = false
```

Representative missing source paths from `index.csv`:

```text
/home/chenjie/cc/robotics/HumanML3D/pose_data/KIT/3/kick_high_left02_poses.npy
/home/chenjie/cc/robotics/HumanML3D/pose_data/CMU/80/80_63_poses.npy
/home/chenjie/cc/robotics/HumanML3D/pose_data/Eyes_Japan_Dataset/hamada/pose-06-hangon-hamada_poses.npy
```

Available native-retarget tools:

```text
Script/stage1/build_bvh_character_gpt_cache.py exists
Script/stage1/diagnose_bvh_character_retarget.py exists
```

Interpretation:

- HumanML3D itself is not abandoned and the processed corpus is usable for
  cataloging, long-caption synthesis, and diagnostics.
- The current local checkout still lacks the source motion needed to run the
  preferred MoConVQ-native BVH retarget route at HumanML3D scale.
- The next mainline data step is to restore/export AMASS/HumanML3D source motion
  to BVH, or to implement a validated `new_joints` to MoConVQ-compatible BVH
  exporter and then run `build_bvh_character_gpt_cache.py`.
- The current hand-written HumanML3D-to-MoConVQ observation/cache route remains
  useful as a diagnostic baseline, but should not be treated as the final
  Stage1 data route unless its retarget quality is repaired and validated.

### Verification

Commands run with the `moconvq` conda environment:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/check_stage1_data_readiness.py \
  tests/test_stage1_data_readiness.py \
  tests/test_stage1_humanml3d.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_data_readiness \
  tests.test_stage1_evaluation_readiness \
  tests.test_stage1_humanml3d \
  -v
```

Result:

```text
py_compile passed
7 tests passed
```

## 2026-06-13: Processed HumanML3D to MoConVQ-template BVH export smoke

### Purpose

The previous readiness audit showed that the local HumanML3D processed corpus is
complete, but original `pose_data`/AMASS source motions and large BVH exports are
missing.  To keep moving on the main HumanML3D route without claiming that the
retarget problem is solved, I added a conservative bridge:

```text
Script/stage1/export_humanml3d_to_bvh.py
tests/test_stage1_humanml3d_bvh_export.py
```

The exporter:

```text
HumanML3D new_joints/new_joint_vecs
-> HumanML3D global 22-joint rotations from 6D representation
-> map to MoConVQ 20-body order
-> write MOTION rows under the existing base.bvh hierarchy
-> feed the resulting BVH into MotionDataSet.add_bvh_with_character()
```

This is a smoke bridge, not a final retarget-quality claim.  It deliberately
reuses `base.bvh`'s hierarchy and channel order so that MoConVQ's native BVH
loader sees a familiar skeleton.

### Code fixes while testing

The BVH diagnostic/cache scripts had the same multi-checkout direct-script import
hazard previously fixed in `generate_long_motion.py`: when run as
`python Script/stage1/*.py`, Python could import `Script.stage1.*` modules from
the older dirty checkout.  I added repo-root path guards and explicit
`--motion-dataset` forwarding to:

```text
Script/stage1/diagnose_bvh_character_retarget.py
Script/stage1/build_bvh_character_gpt_cache.py
Script/stage1/train_real_text_gpt.py
```

The `--motion-dataset` argument is important in the clean `main/stage1` worktree
because the large `simple_motion_data.h5` asset is stored in the older local
MoConVQ worktree and is not committed.

### BVH export command

Executed from:

```text
/home/chenjie/cc/robotics/MoConVQ-main/stage1
```

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/export_humanml3d_to_bvh.py \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --sample-id 000021 \
  --sample-id 012314 \
  --output-dir stage1_artifacts/humanml_bvh_export_smoke_20260613 \
  --summary stage1_artifacts/humanml_bvh_export_smoke_20260613/export_summary.json
```

Result:

```text
000021.bvh:
  frames = 179
  channels = 63
  frame_time = 0.05
  caption = person is walking normally in a circle

012314.bvh:
  frames = 170
  channels = 63
  frame_time = 0.05
  caption = a person appears to be playing tennis and shoots the ball with the racket in his right hand
```

Parser sanity check:

```text
000021.bvh nodes = 25, motion shape = (179, 63), first root = [0.0, 0.9741, 0.0, 0.0, 0.0, 0.0]
012314.bvh nodes = 25, motion shape = (170, 63), first root = [0.0, 0.8370, 0.0, 0.0, 0.0, 0.0]
```

### BVH engineering metrics

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/evaluate_bvh_metrics.py \
  stage1_artifacts/humanml_bvh_export_smoke_20260613/*.bvh \
  --expected-min-frames 120 \
  --output stage1_artifacts/humanml_bvh_export_smoke_20260613/export_bvh_metrics.json
```

Result:

| sample | frames | duration | root path | root displacement | pose velocity mean | pose variance mean | early stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 000021 | 179 | 8.95 | 5.385233 | 0.124187 | 198.533001 | 2387.514295 | false |
| 012314 | 170 | 8.50 | 5.523310 | 0.094050 | 195.905735 | 1744.774603 | false |

The high pose velocity/variance is a warning sign.  It may reflect Euler
discontinuities, imperfect local rotation mapping, or both.  These BVHs are
therefore only a path smoke until rendered and manually inspected.

### Native MoConVQ BVH retarget diagnostic

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/diagnose_bvh_character_retarget.py \
  stage1_artifacts/humanml_bvh_export_smoke_20260613/000021.bvh \
  stage1_artifacts/humanml_bvh_export_smoke_20260613/012314.bvh \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --native-h5 /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --native-observation-key walk1_subject5/observation \
  --fps 20 \
  --output-json stage1_artifacts/humanml_bvh_export_smoke_20260613/native_retarget_diagnostic.json
```

Result:

```text
state_shape = [349, 20, 13]
observation_shape = [349, 323]
RVQ token shape = [87, 4]
observation |z| mean = 1.4201
observation |z| p95 = 4.6358
observation |z| p99 = 12.0568
observation |z| max = 25.5605
observation frac_gt_5 = 0.0433
observation frac_gt_10 = 0.0139
```

Token distribution compared to native `simple_motion_data.h5`:

| depth | exported-BVH unique | native unique | JS divergence bits | exported entropy | native entropy | exported top frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 139 | 1.000000 | 3.201233 | 5.981861 | 0.333333 |
| 1 | 31 | 366 | 0.941444 | 4.150506 | 8.072053 | 0.264368 |
| 2 | 46 | 412 | 0.859005 | 5.081799 | 8.378403 | 0.103448 |
| 3 | 54 | 418 | 0.837896 | 5.521806 | 8.374331 | 0.057471 |

Interpretation:

- The exported processed-HumanML3D BVH can be consumed by MoConVQ's native
  `MotionDataSet.add_bvh_with_character()` path.
- Distribution quality is not yet good: depth0 JS is near the maximum and
  observation p99 `|z|` is high.
- This is still better as an engineering path than the previous hand-written
  cache in one important sense: it exercises MoConVQ's original BVH-to-character
  retarget code instead of directly fabricating simulator state.
- It is not ready for final fine-tuning claims without rendering, semantic
  inspection, and larger-sample diagnostics.

### GPT cache smoke

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --bvh "stage1_artifacts/humanml_bvh_export_smoke_20260613/000021.bvh=person is walking normally in a circle" \
  --bvh "stage1_artifacts/humanml_bvh_export_smoke_20260613/012314.bvh=a person appears to be playing tennis and shoots the ball with the racket in his right hand" \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --window-size 20 \
  --window-stride 10 \
  --fps 20 \
  --output stage1_artifacts/humanml_bvh_export_smoke_20260613/gpt_cache.pt \
  --observation-h5 stage1_artifacts/humanml_bvh_export_smoke_20260613/observations.h5 \
  --summary stage1_artifacts/humanml_bvh_export_smoke_20260613/gpt_cache_summary.json
```

Result:

```text
windows = 8
latents_shape = [8, 20, 768]
indices_shape = [8, 20, 4]
text_features_shape = [8, 256, 1024]
valid_tokens = 640
index_min = 0
index_max = 509
unique_sequences = 2
```

Cache token distribution:

| depth | tokens | unique | entropy bits | top frac |
| --- | ---: | ---: | ---: | ---: |
| 0 | 160 | 15 | 3.124159 | 0.343750 |
| 1 | 160 | 31 | 4.134691 | 0.250000 |
| 2 | 160 | 44 | 4.919886 | 0.137500 |
| 3 | 160 | 57 | 5.614343 | 0.050000 |

Training-data loader smoke:

```text
RealStage1CacheDataset length = 8
latent batch = (2, 20, 768)
indices batch = (2, 20, 4)
text_feature batch = (2, 256, 1024)
text_mask batch = (2, 256)
target_mask batch = (2, 20)
autoregressive pre_latent = (2, 19, 768)
targets = (2, 20, 4)
first caption = person is walking normally in a circle
first sequence_id = 0000_000021
```

Full `train_real_text_gpt.py --smoke` was attempted on CPU.  The environment
reported `GPU not detected`; both `train_scope=all` and `train_scope=head`
remained too slow for a quick smoke and were interrupted manually.  This does
not contradict cache readability; it means full forward/backward smoke should be
rerun on GPU before using this cache for any training claim.

### Verification

Commands run with the `moconvq` conda environment:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/export_humanml3d_to_bvh.py \
  Script/stage1/diagnose_bvh_character_retarget.py \
  Script/stage1/build_bvh_character_gpt_cache.py \
  Script/stage1/train_real_text_gpt.py \
  tests/test_stage1_humanml3d_bvh_export.py \
  tests/test_stage1_bvh_character_retarget.py \
  tests/test_stage1_bvh_character_cache.py \
  tests/test_stage1_real_train.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_humanml3d_bvh_export \
  tests.test_stage1_bvh_character_retarget \
  tests.test_stage1_bvh_character_cache \
  tests.test_stage1_real_train \
  tests.test_stage1_data_readiness \
  tests.test_stage1_repository_hygiene \
  -v
```

Result:

```text
py_compile passed
32 tests passed
```

### Next action

The exporter creates a concrete path to test the main HumanML3D route without
restored AMASS source files, but it needs quality work before scale-up:

```text
1. render 000021.bvh and 012314.bvh to MP4 and inspect for flips/twists;
2. reduce Euler discontinuity or switch to a more stable local rotation export;
3. run a larger sampled export diagnostic and compare observation z-scores/token
   distributions against native MoConVQ data;
4. only then build a larger GPT cache and train on GPU.
```

## 2026-06-13: Joints-IK HumanML3D BVH export improves the bridge path

### Purpose

The previous processed-HumanML3D BVH smoke used `new_joint_vecs` 6D rotations as
if they were directly compatible with the MoConVQ `base.bvh` rigid-body frame.
MP4 inspection showed that this assumption is too strong:

```text
stage1_artifacts/humanml_bvh_export_smoke_20260613/video/000021_contact.png
stage1_artifacts/humanml_bvh_export_smoke_20260613/video/012314_contact.png
```

Visual result:

- `000021` (`person is walking normally in a circle`) showed inverted/prone
  posture changes even though the source caption is a normal walk.
- `012314` (`playing tennis`) showed large limb flips and twisted body poses.

This made the failure concrete: the route could construct a cache, but the
exported BVH motion was not a credible training target.

### Code change

Updated:

```text
Script/stage1/export_humanml3d_to_bvh.py
tests/test_stage1_humanml3d_bvh_export.py
```

The exporter now supports two rotation sources:

```text
--rotation-source joints_ik   # default
--rotation-source vec6d       # old path, kept for reproducibility
```

`joints_ik` uses HumanML3D `new_joints` global joint positions to estimate BVH
local rotations by aligning each MoConVQ BVH node's child offsets to the
corresponding HumanML3D bone directions.  It then unwraps Euler angles over time
before writing the `base.bvh` MOTION rows.  The older `vec6d` path is still
available because it is useful as a negative baseline and for reproducing the
initial smoke.

### Export and visual inspection

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/export_humanml3d_to_bvh.py \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --sample-id 000021 \
  --sample-id 012314 \
  --output-dir stage1_artifacts/humanml_bvh_export_ik_smoke_20260613 \
  --summary stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/export_summary.json
```

Result:

```text
000021.bvh:
  frames = 179
  channels = 63
  frame_time = 0.05
  rotation_source = joints_ik
  unwrap_euler = true

012314.bvh:
  frames = 170
  channels = 63
  frame_time = 0.05
  rotation_source = joints_ik
  unwrap_euler = true
```

Rendered with explicit conda ffmpeg path because plain `ffmpeg` was not on
`PATH`:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/render_bvh_to_mp4.py \
  --input stage1_artifacts/humanml_bvh_export_ik_smoke_20260613 \
  --output-dir stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/video \
  --ffmpeg /home/chenjie/miniconda3/envs/moconvq/bin/ffmpeg \
  --fps 20 \
  --width 960 \
  --height 720 \
  --keep-root-motion
```

Visual result:

- `000021` no longer shows the inverted/prone posture seen in the `vec6d`
  export; the walk is coarse but readable.
- `012314` also improves substantially; complex tennis poses are still rough and
  sometimes over-bent, but the large full-body flips are gone in the contact
  sheet.

### BVH engineering metrics

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/evaluate_bvh_metrics.py \
  stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/*.bvh \
  --expected-min-frames 120 \
  --output stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/export_bvh_metrics.json
```

Result:

| sample | frames | duration | root path | root displacement | pose velocity mean | pose variance mean | early stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 000021 | 179 | 8.95 | 5.385233 | 0.124187 | 132.143545 | 6310.638096 | false |
| 012314 | 170 | 8.50 | 5.523310 | 0.094050 | 145.523080 | 4698.496920 | false |

The pose velocity is lower than the earlier `vec6d` export (`198.53` and
`195.91`), but pose variance remains high.  This matches the visual diagnosis:
IK fixes catastrophic orientation mismatch but is still a rough bridge.

### Native MoConVQ retarget diagnostic

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/diagnose_bvh_character_retarget.py \
  stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/000021.bvh \
  stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/012314.bvh \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --native-h5 /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --native-observation-key walk1_subject5/observation \
  --fps 20 \
  --output-json stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/native_retarget_diagnostic.json
```

Result:

```text
state_shape = [349, 20, 13]
observation_shape = [349, 323]
RVQ token shape = [87, 4]
observation |z| mean = 0.9015
observation |z| p95 = 3.1468
observation |z| p99 = 7.8950
observation |z| max = 59.4022
observation frac_gt_5 = 0.0218
observation frac_gt_10 = 0.0072
```

Token distribution compared to native `simple_motion_data.h5`:

| depth | exported-BVH unique | native unique | JS divergence bits | exported entropy | native entropy | exported top frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 51 | 139 | 0.851726 | 5.348020 | 5.981861 | 0.114943 |
| 1 | 68 | 366 | 0.790950 | 5.908123 | 8.072053 | 0.068966 |
| 2 | 61 | 412 | 0.809710 | 5.644944 | 8.378403 | 0.080460 |
| 3 | 68 | 418 | 0.741603 | 5.910017 | 8.374331 | 0.057471 |

Compared with the earlier `vec6d` smoke, depth0 top fraction improved from about
`0.333` to about `0.115`, and observation p99 `|z|` improved from about `12.06`
to about `7.90`.  The high max `|z|` means the route still needs filtering and
larger-sample diagnostics before formal training.

### GPT cache and training smoke

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --bvh "stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/000021.bvh=person is walking normally in a circle" \
  --bvh "stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/012314.bvh=a person appears to be playing tennis and shoots the ball with the racket in his right hand" \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --window-size 20 \
  --window-stride 10 \
  --fps 20 \
  --output stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/gpt_cache.pt \
  --observation-h5 stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/observations.h5 \
  --summary stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/gpt_cache_summary.json
```

Result:

```text
windows = 8
latents_shape = [8, 20, 768]
indices_shape = [8, 20, 4]
text_features_shape = [8, 256, 1024]
valid_tokens = 640
index_min = 5
index_max = 510
unique_sequences = 2
```

Cache token distribution:

| depth | tokens | unique | entropy bits | top frac |
| --- | ---: | ---: | ---: | ---: |
| 0 | 160 | 50 | 5.337615 | 0.068750 |
| 1 | 160 | 67 | 5.884463 | 0.050000 |
| 2 | 160 | 63 | 5.611659 | 0.112500 |
| 3 | 160 | 71 | 5.933638 | 0.062500 |

Training-data loader smoke:

```text
RealStage1CacheDataset length = 8
latent batch = (2, 20, 768)
indices batch = (2, 20, 4)
text_feature batch = (2, 256, 1024)
text_mask batch = (2, 256)
target_mask batch = (2, 20)
autoregressive pre_latent = (2, 19, 768)
targets = (2, 20, 4)
first caption = person is walking normally in a circle
first sequence_id = 0000_000021
```

Head-only forward/backward smoke:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/gpt_cache.pt \
  --val-cache stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/gpt_cache.pt \
  --init-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --output-dir stage1_artifacts/humanml_bvh_export_ik_smoke_20260613/train_smoke_head \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --train-scope head \
  --gpu 0 \
  --smoke
```

Result:

```text
GPU not detected. Defaulting to CPU.
train_scope=head trainable_parameters=7880448
epoch=0 train=9.5660/acc=0.0000 val=18.2526/acc=0.0500 elapsed=5.3s
```

This is a pipeline smoke only.  It proves that the IK cache can enter
`train_real_text_gpt.py` and execute one forward/backward/update step.  It is not
a quality result and should not be reported as a trained model improvement.

### Interpretation

- The previous `vec6d` exporter should not be used as the main training-data
  path except as a documented failure baseline.
- The new default `joints_ik` exporter is a better processed-HumanML3D bridge:
  it improves visual sanity, token diversity, and observation z-score
  diagnostics while still using MoConVQ's native BVH-to-character loader.
- It is still not final.  The high max observation z-score and rough complex
  poses require larger-sample export diagnostics, quality filtering, and more
  video inspection before building a real fine-tuning cache.

### Next action

Run a sampled HumanML3D export diagnostic, for example 20 to 100 samples:

```text
HumanML3D processed samples
-> export_humanml3d_to_bvh.py --rotation-source joints_ik
-> render a subset to MP4/contact sheets
-> diagnose_bvh_character_retarget.py
-> reject or filter sequences with extreme observation z-score/token collapse
-> build a larger GPT cache only from acceptable sequences
```

## 2026-06-13: Batch10 joints-IK export quality filtering and filtered-cache smoke

### Purpose

The two-sample IK smoke showed that the route is promising, but two examples are
too few to decide whether the processed-HumanML3D BVH bridge can feed training.
This run adds the missing reproducible batch controls and a preliminary
engineering filter:

```text
export_humanml3d_to_bvh.py --split/--limit/--seed
diagnose_bvh_character_retarget.py --per-file
summarize_bvh_retarget_quality.py
build_bvh_character_gpt_cache.py --quality-summary
```

The intent is to move from "the bridge can run" to "we can identify which
exported BVHs are plausible cache candidates."

### Code changes

Updated:

```text
Script/stage1/export_humanml3d_to_bvh.py
Script/stage1/diagnose_bvh_character_retarget.py
Script/stage1/build_bvh_character_gpt_cache.py
```

Added:

```text
Script/stage1/summarize_bvh_retarget_quality.py
tests/test_stage1_bvh_retarget_quality.py
```

Key behavior:

- exporter can now sample from a HumanML3D split with `--split`, `--limit`, and
  `--seed`;
- retarget diagnostic can emit per-file z-score/token summaries with
  `--per-file`;
- quality summarizer combines export summary, BVH metrics, and retarget
  diagnostics into accepted/rejected rows;
- BVH cache builder can consume accepted rows directly with `--quality-summary`.

### Batch10 export

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/export_humanml3d_to_bvh.py \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --split train \
  --limit 10 \
  --seed 13 \
  --output-dir stage1_artifacts/humanml_bvh_export_ik_batch10_20260613 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/export_summary.json
```

Selected sample ids:

```text
007115, 008684, 011137, M001207, M009990,
012887, M005884, M006200, 012245, M012785
```

The batch includes walk/run/jump/swing/cartwheel/near-static arm motions, so it
is a useful tiny stress test.  Export time for 10 samples was about 32 seconds.

### Aggregate retarget diagnostic

Commands:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/evaluate_bvh_metrics.py \
  stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/*.bvh \
  --expected-min-frames 120 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/export_bvh_metrics.json

/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/diagnose_bvh_character_retarget.py \
  stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/*.bvh \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --native-h5 /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --native-observation-key walk1_subject5/observation \
  --fps 20 \
  --per-file \
  --output-json stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/native_retarget_diagnostic.json
```

Aggregate result:

```text
state_shape = [1585, 20, 13]
observation_shape = [1585, 323]
RVQ token shape = [396, 4]
observation |z| mean = 0.7923
observation |z| p95 = 2.8537
observation |z| p99 = 6.3582
observation |z| max = 45.4098
observation frac_gt_5 = 0.0162
observation frac_gt_10 = 0.0028
```

Token distribution compared to native `simple_motion_data.h5`:

| depth | exported unique | native unique | JS divergence bits | exported entropy | native entropy | exported top frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 142 | 139 | 0.795736 | 6.275738 | 5.981861 | 0.113636 |
| 1 | 205 | 366 | 0.549635 | 7.060256 | 8.072053 | 0.113636 |
| 2 | 212 | 412 | 0.509432 | 7.272277 | 8.378403 | 0.060606 |
| 3 | 170 | 418 | 0.596195 | 6.537737 | 8.374331 | 0.095960 |

Compared with the two-sample IK smoke, larger batch diversity improves JS
divergence at depths 1-3 and increases unique token coverage.

### Preliminary quality filter

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/summarize_bvh_retarget_quality.py \
  --retarget-json stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/native_retarget_diagnostic.json \
  --bvh-metrics-json stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/export_bvh_metrics.json \
  --export-summary stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/export_summary.json \
  --output-json stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/quality_summary.json \
  --output-csv stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/quality_summary.csv
```

Default preliminary thresholds:

```text
min_frames = 120
min_tokens = 20
max_p99_abs_z = 8.0
max_max_abs_z = 50.0
max_frac_gt_5 = 0.05
max_depth0_top_frac = 0.25
min_depth0_unique = 16
```

Result:

```text
total = 10
accepted = 5
rejected = 5
```

Accepted:

| sample | frames | tokens | p99 | max | depth0 top | depth0 unique | caption |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 011137 | 128 | 32 | 6.8734 | 25.4324 | 0.0938 | 24 | run right, turn, run left, then middle |
| M006200 | 199 | 49 | 6.5083 | 43.5899 | 0.1020 | 29 | swinging a club or bat |
| 007115 | 199 | 49 | 5.7387 | 13.1242 | 0.1429 | 23 | walking in an s pattern |
| M009990 | 181 | 45 | 5.5772 | 10.0675 | 0.1333 | 23 | runs sideways repeatedly |
| 012887 | 148 | 37 | 4.8993 | 12.8066 | 0.1622 | 21 | jumping straight up |

Rejected:

| sample | reason |
| --- | --- |
| M005884 | high p99/frac_gt_5; cartwheel-like motion remains hard for the bridge |
| 008684 | short and token-collapsed |
| 012245 | short, too few tokens, token-collapsed |
| M012785 | low z-score but severe depth0 token collapse |
| M001207 | low z-score but depth0 token collapse |

This is an important distinction: low observation z-score is not sufficient.
Near-static or low-diversity samples can be physically normal but poor GPT
fine-tuning targets if their RVQ stream collapses to a few tokens.

### Filtered GPT cache smoke

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/quality_summary.json \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --window-size 20 \
  --window-stride 10 \
  --fps 20 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/gpt_cache_filtered.pt \
  --observation-h5 stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/observations_filtered.h5 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch10_20260613/gpt_cache_filtered_summary.json
```

Result:

```text
windows = 18
latents_shape = [18, 20, 768]
indices_shape = [18, 20, 4]
text_features_shape = [18, 256, 1024]
valid_tokens = 1440
index_min = 0
index_max = 510
unique_sequences = 5
```

Filtered cache token distribution:

| depth | tokens | unique | entropy bits | top frac |
| --- | ---: | ---: | ---: | ---: |
| 0 | 360 | 106 | 6.410223 | 0.038889 |
| 1 | 360 | 144 | 6.917245 | 0.027778 |
| 2 | 360 | 151 | 6.970901 | 0.027778 |
| 3 | 360 | 131 | 6.531636 | 0.080556 |

Training-data loader smoke:

```text
RealStage1CacheDataset length = 18
latent batch = (2, 20, 768)
indices batch = (2, 20, 4)
text_feature batch = (2, 256, 1024)
text_mask batch = (2, 256)
target_mask batch = (2, 20)
autoregressive pre_latent = (2, 19, 768)
targets = (2, 20, 4)
first caption = a person runs to the right, turns around and runs to the left, then runs towards the middle.
first sequence_id = 0000_011137
```

Head-only forward/backward smoke:

```text
GPU not detected. Defaulting to CPU.
train_scope=head trainable_parameters=7880448
epoch=0 train=7.3669/acc=0.0000 val=28.2333/acc=0.0875 elapsed=7.2s
```

This confirms the filtered cache can enter `train_real_text_gpt.py`, but it is
still a tiny smoke.  It should not be reported as model improvement.

### Interpretation

- The Stage1 main HumanML3D route now has a reproducible engineering loop:

```text
processed HumanML3D split sample
-> joints-IK BVH export
-> MoConVQ native BVH-to-character retarget
-> per-file quality filter
-> accepted-only GPT cache
-> training smoke
```

- The current bottleneck is no longer only code wiring.  It is dataset quality
  control and scale: thresholds need to be tested on at least 50-100 samples,
  with video inspection for accepted and rejected examples.
- Rejected examples are informative and should stay in the experiment record:
  they show that cartwheels/acrobatics and token-collapsed near-static clips are
  not safe to include blindly.

### Verification

Commands:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/export_humanml3d_to_bvh.py \
  Script/stage1/diagnose_bvh_character_retarget.py \
  Script/stage1/build_bvh_character_gpt_cache.py \
  Script/stage1/summarize_bvh_retarget_quality.py \
  tests/test_stage1_humanml3d_bvh_export.py \
  tests/test_stage1_bvh_character_retarget.py \
  tests/test_stage1_bvh_character_cache.py \
  tests/test_stage1_bvh_retarget_quality.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_humanml3d_bvh_export \
  tests.test_stage1_bvh_character_retarget \
  tests.test_stage1_bvh_character_cache \
  tests.test_stage1_bvh_retarget_quality \
  -v
```

Result:

```text
py_compile passed
18 tests passed
```

### Next action

Run the same loop on a larger sample:

```text
train split limit 50 or 100
-> render contact sheets for accepted/rejected examples
-> adjust thresholds if false positives/false negatives are obvious
-> build a larger accepted-only cache
-> run GPU training smoke/full fine-tune when GPU availability is confirmed
```

## 2026-06-13: Batch50 joints-IK export quality filtering and filtered-cache smoke

### Purpose

The batch10 experiment proved that the processed-HumanML3D to
MoConVQ-template BVH bridge can run through native
`MotionDataSet.add_bvh_with_character()` and produce a trainable GPT cache.
The next question was whether the same route remains useful on a larger random
train split sample, and whether the quality filter rejects plausible failure
modes instead of accepting every exported BVH.

This is still an engineering data-quality experiment, not a model-quality
claim.  No paper-level FID/R-precision evaluator was used.

### Code change

`export_humanml3d_to_bvh.py` now loads the HumanML3D catalog once in `main()`
and passes it into `write_humanml3d_bvh()`.  This avoids repeatedly rebuilding
the text/catalog index when exporting many samples in one run.  The exported
motion content is unchanged.

### Export and retarget artifacts

Artifacts are local and ignored by Git:

```text
stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/
```

The export used:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/export_humanml3d_to_bvh.py \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --split train \
  --limit 50 \
  --seed 29 \
  --output-dir stage1_artifacts/humanml_bvh_export_ik_batch50_20260613 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/export_summary.json
```

Then BVH engineering metrics and native retarget diagnostics were generated:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/evaluate_bvh_metrics.py \
  stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/*.bvh \
  --expected-min-frames 120 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/export_bvh_metrics.json

/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/diagnose_bvh_character_retarget.py \
  stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/*.bvh \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --native-h5 /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --native-observation-key walk1_subject5/observation \
  --fps 20 \
  --per-file \
  --output-json stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/native_retarget_diagnostic.json
```

One sandboxed diagnostic attempt hit an MPI/socket initialization error
(`PMPI_Init_thread ... unable to create a socket`).  The same command succeeded
after rerunning with escalated permissions.  This is an execution-environment
issue, not a retarget-quality result.

Aggregate native retarget summary:

```text
state_shape = [7355, 20, 13]
observation_shape = [7355, 323]
RVQ token shape = [1838, 4]
observation |z| mean = 0.723373
observation |z| p50 = 0.364388
observation |z| p90 = 1.585061
observation |z| p95 = 2.502536
observation |z| p99 = 6.161048
observation |z| max = 66.685669
observation frac_gt_3 = 0.036870
observation frac_gt_5 = 0.014325
observation frac_gt_10 = 0.004065
```

Token distribution compared to native `simple_motion_data.h5`:

| depth | exported unique | native unique | JS divergence bits | exported entropy | native entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 310 | 139 | 0.735875 | 7.290868 | 5.981861 |
| 1 | 435 | 366 | 0.367018 | 8.160762 | 8.072053 |
| 2 | 441 | 412 | 0.334422 | 8.248014 | 8.378403 |
| 3 | 416 | 418 | 0.345083 | 7.892674 | 8.374331 |

Compared with batch10, increasing sample diversity substantially improves
depths 1-3.  Depth0 still differs from the native reference, and the max
observation z-score is high, so per-file filtering remains necessary.

### Quality filtering

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/summarize_bvh_retarget_quality.py \
  --retarget-json stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/native_retarget_diagnostic.json \
  --bvh-metrics-json stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/export_bvh_metrics.json \
  --export-summary stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/export_summary.json \
  --output-json stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/quality_summary.json \
  --output-csv stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/quality_summary.csv
```

Result:

```text
total = 50
accepted = 10
rejected = 40
```

Accepted samples:

| sample | frames | tokens | p99 | max | depth0 top | depth0 unique | caption |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M004417 | 178 | 44 | 7.1374 | 32.5422 | 0.1364 | 23 | throwing or hitting with left arm twice |
| 003985 | 199 | 49 | 6.5829 | 16.5302 | 0.1429 | 28 | walks around carefully looking for something |
| M009321 | 199 | 49 | 6.1444 | 19.8138 | 0.1633 | 26 | walks around in a wavy/s pattern |
| M012210 | 199 | 49 | 5.8896 | 14.4010 | 0.1020 | 24 | walking around in a squiggly line |
| 009799 | 199 | 49 | 4.7375 | 14.1391 | 0.1020 | 31 | jumps rope |
| 007392 | 168 | 42 | 3.2787 | 7.3688 | 0.0952 | 23 | shuffles right, left, then right |
| M013714 | 199 | 49 | 2.9965 | 13.8271 | 0.1224 | 23 | chicken dance and claps hands |
| 004249 | 195 | 48 | 2.8696 | 4.2709 | 0.1250 | 18 | overhand swimming motions to stretch |
| 006268 | 140 | 35 | 2.3208 | 5.1012 | 0.1143 | 24 | steps forward, loses balance, resumes walk |
| M013604 | 193 | 48 | 2.3162 | 4.9922 | 0.1667 | 25 | walks tentatively through slippery or muddy ground |

Reject-reason counts:

| reason | count |
| --- | ---: |
| depth0_unique<16 | 27 |
| depth0_top_frac>0.25 | 24 |
| frames<120 | 19 |
| p99_abs_z>8 | 9 |
| tokens<20 | 8 |
| max_abs_z>50 | 7 |
| frac_gt_5>0.05 | 2 |

The dominant rejection mode is not only high z-score.  Many low-z-score samples
are still rejected because depth0 collapses to a few tokens or the sample is too
short.  This supports using both observation-space and token-space checks before
building the GPT cache.

### Filtered GPT cache smoke

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/quality_summary.json \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --window-size 20 \
  --window-stride 10 \
  --fps 20 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/gpt_cache_filtered.pt \
  --observation-h5 stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/observations_filtered.h5 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/gpt_cache_filtered_summary.json
```

Result:

```text
windows = 39
latents_shape = [39, 20, 768]
indices_shape = [39, 20, 4]
text_features_shape = [39, 256, 1024]
valid_tokens = 3120
index_min = 0
index_max = 511
unique_sequences = 10
```

Filtered cache token distribution:

| depth | tokens | unique | entropy bits | top frac |
| --- | ---: | ---: | ---: | ---: |
| 0 | 780 | 152 | 6.788135 | 0.041026 |
| 1 | 780 | 254 | 7.688304 | 0.025641 |
| 2 | 780 | 246 | 7.609442 | 0.034615 |
| 3 | 780 | 229 | 7.326107 | 0.050000 |

Head-only train smoke:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/gpt_cache_filtered.pt \
  --val-cache stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/gpt_cache_filtered.pt \
  --init-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --output-dir stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/train_smoke_head \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --train-scope head \
  --gpu 0 \
  --smoke
```

Result:

```text
GPU not detected. Defaulting to CPU.
train_scope=head trainable_parameters=7880448
epoch=0 train=15.7566/acc=0.0125 val=24.6501/acc=0.1750 elapsed=5.0s
```

This confirms that the batch50 accepted-only cache can enter the same
`train_real_text_gpt.py` path as earlier real caches.  It is not a quality
claim because `--smoke` runs only one batch for train and one batch for val.

### Contact-sheet visual audit

Added `Script/stage1/make_bvh_contact_sheet.py` to create static BVH visual
audit sheets from either explicit BVH paths or a `quality_summary.json`.
It reuses the repository BVH parser/forward-kinematics code and samples a fixed
number of frames per motion.  This is faster than opening many individual MP4s
when checking accepted/rejected samples after a quality-filter run.

Commands:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/make_bvh_contact_sheet.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/quality_summary.json \
  --selection accepted \
  --limit-per-class 10 \
  --frames-per-motion 6 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/contact_sheet_accepted.png

/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/make_bvh_contact_sheet.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/quality_summary.json \
  --selection rejected \
  --limit-per-class 10 \
  --frames-per-motion 6 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/contact_sheet_rejected_top10.png

/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/make_bvh_contact_sheet.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/quality_summary.json \
  --selection both \
  --limit-per-class 5 \
  --frames-per-motion 6 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch50_20260613/contact_sheet_accept5_reject5.png
```

Observed from the accepted contact sheet:

- Accepted samples no longer show the large upside-down / prone failures seen
  in the earlier `vec6d` exporter.
- Walking, shuffling, arm motions, jump-rope-like motion, and tentative walking
  are generally readable as coarse stick figures.
- Some accepted rows still show rough motion: exaggerated bending, uncertain
  foot contact, or arms that may need video inspection.  Therefore accepted
  means "candidate training data", not "final verified good motion".

Observed from the rejected top10 contact sheet:

- Several rejections visibly match hard retarget cases, including handstand,
  yoga/leg-high pose, aggressive gestures, and extreme bends.
- Some high-z-score rejected walking/running rows look superficially plausible
  in sparse static frames.  These are not automatically false rejections:
  contact sheets can miss temporal discontinuities, velocity spikes, and token
  collapse.  They should be checked with MP4/video before loosening thresholds.
- The current filter is usefully conservative.  It catches obvious hard poses,
  short clips, and token-collapsed samples, but threshold tuning still needs a
  larger visual audit set.

### Interpretation

- HumanML3D is still the main Stage1 data route.  The current work replaces the
  unreliable hand-written HumanML3D-to-observation cache path with a more
  defensible bridge:

```text
processed HumanML3D
-> joints-IK MoConVQ-template BVH export
-> MoConVQ native BVH-to-character retarget
-> per-file quality filter
-> accepted-only GPT cache
```

- Batch50 improves token distribution over batch10, especially at RVQ depths
  1-3, but only 20% of sampled clips pass the current thresholds.
- The thresholds are still preliminary engineering thresholds.  Before using
  the accepted set for a report-level model claim, accepted and rejected samples
  should be rendered and inspected for false positives/false negatives.  The
  batch50 contact sheets are the first static visual audit for this purpose,
  but MP4 checks are still needed for temporal artifacts.
- A real LLM in-context learning experiment has still not been run.  Codex
  interaction is not counted as that experiment.  If the backup route is used,
  the actual external/local LLM model and saved JSON response must be recorded.

### Next action

Scale the same route to a larger train/val export, render contact sheets for
accepted and rejected samples, and only then run a conservative MoConGPT
fine-tune against the baseline GPT.  If the accepted rate remains too low or
videos show systematic retarget artifacts, restore original HumanML3D/AMASS
source motion or BVH exports instead of overfitting to the processed-joints
bridge.

## 2026-06-13: Batch100 joints-IK export scale check

### Purpose

After batch50, the next question was whether the same processed-HumanML3D
bridge remains stable at a slightly larger scale, and whether accepted-only
cache diversity improves enough to justify moving toward a real train/val
cache.  This run uses the same route:

```text
HumanML3D new_joints/new_joint_vecs
-> export_humanml3d_to_bvh.py --rotation-source joints_ik
-> MoConVQ native MotionDataSet.add_bvh_with_character()
-> per-file quality summary
-> accepted-only GPT cache
-> token/contact-sheet/train-smoke diagnostics
```

### Export

Command:

```bash
/usr/bin/time -f 'elapsed_sec=%e' \
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/export_humanml3d_to_bvh.py \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --split train \
  --limit 100 \
  --seed 47 \
  --output-dir stage1_artifacts/humanml_bvh_export_ik_batch100_20260613 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch100_20260613/export_summary.json
```

Result:

```text
exports = 100
frames min/max/avg = 48 / 199 / 146.97
samples shorter than 120 frames = 38
elapsed_sec = 50.05
```

### Native retarget diagnostic

The first sandboxed `diagnose_bvh_character_retarget.py --per-file` attempt hit
the same MPI/socket sandbox limitation seen in batch50:

```text
unable to create a socket, Operation not permitted
```

The same command succeeded after rerunning with escalated permissions.

Aggregate result:

```text
state_shape = [14697, 20, 13]
observation_shape = [14697, 323]
RVQ token shape = [3674, 4]
observation |z| mean = 0.711424
p50 = 0.365297
p90 = 1.609671
p95 = 2.436284
p99 = 5.521938
max = 67.793686
frac_gt_3 = 0.034255
frac_gt_5 = 0.011879
frac_gt_10 = 0.003274
elapsed_sec = 42.70
```

Token distribution compared to native `simple_motion_data.h5`:

| depth | exported unique | native unique | JS divergence bits | exported entropy | native entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 390 | 139 | 0.690223 | 7.605529 | 5.981861 |
| 1 | 486 | 366 | 0.344627 | 8.431412 | 8.072053 |
| 2 | 495 | 412 | 0.282668 | 8.399105 | 8.378403 |
| 3 | 481 | 418 | 0.284419 | 8.124991 | 8.374331 |

Compared with batch50, batch100 improves JS divergence at every RVQ depth and
keeps aggregate observation p99 lower, but max z-score is still high.  The
per-file filter remains necessary.

### Quality filtering

Default preliminary thresholds were unchanged:

```text
min_frames = 120
min_tokens = 20
max_p99_abs_z = 8.0
max_max_abs_z = 50.0
max_frac_gt_5 = 0.05
max_depth0_top_frac = 0.25
min_depth0_unique = 16
```

Result:

```text
total = 100
accepted = 16
rejected = 84
accepted rate = 16%
```

Accepted labels:

```text
004123, M010515, M005678, M013375, 002805, 000473, 008008,
M012311, 006742, 002381, 001001, 001289, 003806, M001709,
M001507, 002198
```

Reject-reason counts:

| reason | count |
| --- | ---: |
| depth0_unique<16 | 56 |
| depth0_top_frac>0.25 | 48 |
| frames<120 | 38 |
| p99_abs_z>8 | 18 |
| max_abs_z>50 | 13 |
| tokens<20 | 10 |
| frac_gt_5>0.05 | 1 |

The accepted rate drops slightly from batch50's 20% to 16%.  This is not
necessarily worse data: batch100 included many short or near-static clips.
However, it confirms that the current processed-joints bridge needs substantial
oversampling or a smarter source-sample selection strategy to build a large
accepted-only train/val cache.

### Filtered GPT cache

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch100_20260613/quality_summary.json \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --window-size 20 \
  --window-stride 10 \
  --fps 20 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch100_20260613/gpt_cache_filtered.pt \
  --observation-h5 stage1_artifacts/humanml_bvh_export_ik_batch100_20260613/observations_filtered.h5 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch100_20260613/gpt_cache_filtered_summary.json
```

Result:

```text
windows = 61
latents_shape = [61, 20, 768]
indices_shape = [61, 20, 4]
text_features_shape = [61, 256, 1024]
valid_tokens = 4880
index range = 0..511
unique_sequences = 16
```

Filtered cache token distribution:

| depth | tokens | unique | entropy bits | top frac |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1220 | 221 | 7.289220 | 0.022951 |
| 1 | 1220 | 322 | 7.984570 | 0.013934 |
| 2 | 1220 | 314 | 7.898292 | 0.031148 |
| 3 | 1220 | 317 | 7.777926 | 0.051639 |

This is the healthiest accepted-only cache distribution so far.  It is still
too small for a report-level fine-tune, but it is a useful pilot cache for
training-pipeline checks.

### Contact-sheet visual audit

Generated:

```text
stage1_artifacts/humanml_bvh_export_ik_batch100_20260613/contact_sheet_accepted.png
stage1_artifacts/humanml_bvh_export_ik_batch100_20260613/contact_sheet_rejected_top12.png
```

Accepted-sheet observations:

- Most accepted rows remain upright and readable as stick-figure motion.
- No large global inversion/prone failure appears in the accepted sheet.
- Several accepted samples still need video checks before formal training:
  `000473` includes crawling/get-up behavior; `002381`, `006742`, and
  `M001709` include bending or large leg motion that sparse frames cannot fully
  validate.

Rejected top12 observations:

- Rejected hard cases include rolling, floor/get-up, sitting, zombie-pose, and
  high-bend motions.  These align with high z-score or token-collapse reasons.
- Some walking/turning rows look superficially plausible in sparse frames but
  are rejected by high p99/max z-score.  Do not relax thresholds based only on
  contact sheets; MP4 inspection is needed for temporal discontinuities.

### Training smoke

An initial smoke with output under `stage1_artifacts/.../train_smoke_head`
failed while saving `best_val.pth` because the repository filesystem was full:

```text
/dev/sda1  2.9T  2.8T  0  100%  /home/chenjie/cc/robotics
PytorchStreamWriter failed writing file ... file write failed
```

The half-written checkpoint was removed, and the same smoke was rerun with
`--output-dir /tmp/stage1_batch100_train_smoke_head`, where `/tmp` still had
space:

```text
/dev/nvme0n1p3  885G  466G  375G  56%  /tmp
```

After a manual cleanup, the repository filesystem had about 42G available:

```text
/dev/sda1  2.9T  2.7T  42G  99%  /home/chenjie/cc/robotics
```

Result:

```text
GPU not detected. Defaulting to CPU.
train_scope=head trainable_parameters=7880448
epoch=0 train=9.7501/acc=0.0000 val=27.3806/acc=0.0750 elapsed=5.6s
```

This confirms the batch100 accepted-only cache can enter the training path.
The train/val numbers are still smoke-only and should not be reported as model
improvement.

### Interpretation

- Scaling from 50 to 100 source samples improves aggregate token distribution
  and produces a cleaner accepted-only cache distribution.
- The accepted yield remains low: 16 accepted out of 100 sampled clips.
  Building a real fine-tuning set will likely require either:
  - oversampling many more processed HumanML3D clips and filtering; or
  - recovering original AMASS/HumanML3D source motion/BVH to reduce bridge
    artifacts; or
  - using a smarter prefilter to avoid short/static/acrobatics/floor-motion
    clips that the current bridge handles poorly.
- The `/home` filesystem is no longer at 0 bytes free after cleanup, but it is
  still 99% used.  Formal training checkpoints and large generated videos
  should be written to `/tmp` or another spacious filesystem, or old
  `stage1_artifacts` should be cleaned intentionally before a long run.
- A real LLM in-context experiment has still not been run.

### Next action

Do not start a full fine-tune from only 16 accepted sequences.  The next
data-building step should either:

```text
oversample train split to collect roughly 100 accepted clips
-> generate contact sheets/MP4 for accepted and top rejected rows
-> build train/val accepted-only caches
-> run conservative base_head/temporal_base_head fine-tune
```

or recover a more native HumanML3D/AMASS BVH/source-motion route if available.

## 2026-06-13: Batch500 HumanML3D joints-IK BVH scaling check

### Purpose

After batch100, the accepted yield was still too small for any honest
fine-tuning claim.  This run scaled the same processed-HumanML3D bridge to 500
train split samples to check whether simple oversampling can produce a useful
accepted-only MoConGPT cache before changing the retarget algorithm again.

This is still the HumanML3D main route, not an LLM in-context run.  No external
or local LLM response was used in this experiment.

### Commands

Export processed HumanML3D `new_joints/new_joint_vecs` to MoConVQ-template BVH:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/export_humanml3d_to_bvh.py \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --split train \
  --limit 500 \
  --seed 13 \
  --output-dir stage1_artifacts/humanml_bvh_export_ik_batch500_20260613 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/export_summary.json \
  --quiet
```

Result:

```text
exports = 500
rotation_source = joints_ik
elapsed_sec = 202.93
frames min/max/avg = 19 / 199 / 141.422
samples shorter than 120 frames = 205
```

BVH engineering metrics:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/evaluate_bvh_metrics.py \
  stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/*.bvh \
  --expected-min-frames 120 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/export_bvh_metrics.json \
  --quiet
```

Result:

```text
rows = 500
early_stop = 205
elapsed_sec = 1.95
```

MoConVQ-native character retarget diagnostic:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/diagnose_bvh_character_retarget.py \
  stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/*.bvh \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --native-h5 /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --native-observation-key walk1_subject5/observation \
  --fps 20 \
  --per-file \
  --output-json stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/native_retarget_diagnostic.json \
  --quiet
```

The first sandboxed attempt failed with the known MPI/socket initialization
error:

```text
unable to create a socket, Operation not permitted
```

The same command succeeded after rerunning with escalated permissions.

Aggregate retarget result:

```text
state_shape = [70711, 20, 13]
observation_shape = [70711, 323]
RVQ token shape = [17677, 4]
observation |z| mean = 0.696808
p50 = 0.362517
p90 = 1.622412
p95 = 2.411780
p99 = 5.045906
max = 68.327934
frac_gt_3 = 0.032125
frac_gt_5 = 0.010195
frac_gt_10 = 0.002717
elapsed_sec = 185.80
```

Token distribution compared to native `simple_motion_data.h5`:

| depth | exported unique | native unique | JS divergence bits | exported entropy | native entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 494 | 139 | 0.725128 | 7.948797 | 5.981861 |
| 1 | 510 | 366 | 0.296112 | 8.542718 | 8.072053 |
| 2 | 511 | 412 | 0.239919 | 8.532781 | 8.378403 |
| 3 | 508 | 418 | 0.252044 | 8.272264 | 8.374331 |

Compared with batch100, batch500 has more source diversity and better JS
divergence at depths 1-3, but depth0 JS remains high and max observation
z-score is still large.  Per-file filtering is still required.

### Quality filtering

Default preliminary thresholds were unchanged:

```text
min_frames = 120
min_tokens = 20
max_p99_abs_z = 8.0
max_max_abs_z = 50.0
max_frac_gt_5 = 0.05
max_depth0_top_frac = 0.25
min_depth0_unique = 16
```

Command:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/summarize_bvh_retarget_quality.py \
  --retarget-json stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/native_retarget_diagnostic.json \
  --bvh-metrics-json stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/export_bvh_metrics.json \
  --export-summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/export_summary.json \
  --output-json stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_summary.json \
  --output-csv stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_summary.csv \
  --quiet
```

Result:

```text
total = 500
accepted = 90
rejected = 410
accepted rate = 18%
elapsed_sec = 0.36
accepted frames min/max/avg = 120 / 199 / 177.733
accepted tokens min/max/avg = 30 / 49 / 43.900
```

Reject-reason counts:

| reason | count |
| --- | ---: |
| depth0_unique<16 | 297 |
| depth0_top_frac>0.25 | 252 |
| frames<120 | 205 |
| tokens<20 | 84 |
| p99_abs_z>8 | 72 |
| max_abs_z>50 | 47 |
| frac_gt_5>0.05 | 9 |

The accepted rate is similar to batch50/batch100.  The larger run confirms
that simple oversampling can produce roughly 100 accepted processed-HumanML3D
clips, but the filter is still conservative and loses many samples to short
clips and depth0 token collapse.

### Filtered GPT cache

Build command:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_summary.json \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --window-size 50 \
  --window-stride 25 \
  --rvq-depth 4 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_filtered.pt \
  --observation-h5 stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/observations_filtered.h5 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_filtered_summary.json \
  --quiet
```

The first sandboxed attempt failed with the same MPI/socket error and the
rerun with escalated permissions succeeded:

```text
windows = 90
latents_shape = [90, 50, 768]
indices_shape = [90, 50, 4]
text_features_shape = [90, 256, 1024]
valid_tokens = 15804
index range = 0..511
unique_sequences = 90
elapsed_sec = 31.58
```

Filtered cache token distribution:

| depth | tokens | unique | entropy bits | top frac |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3951 | 403 | 7.900698 | 0.038471 |
| 1 | 3951 | 496 | 8.560525 | 0.012655 |
| 2 | 3951 | 498 | 8.508563 | 0.042268 |
| 3 | 3951 | 486 | 8.104762 | 0.076943 |

This is the largest and healthiest accepted-only cache so far.  It is still
small and not split into train/val, but it is now a plausible minimum-size
source for a conservative fine-tune experiment after visual/video checks.

### Contact-sheet visual audit

Generated:

```text
stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/contact_sheet_accepted_top16.png
stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/contact_sheet_rejected_top16.png
```

Accepted top16 observations:

- Most accepted rows remain upright and readable as sparse stick-figure
  sequences.
- Some accepted rows are still risky.  For example `013481` includes a
  forward/drop-to-ground motion, and `003355` includes large high-leg poses.
  These need MP4 inspection before using the accepted set for a report-level
  model claim.

Rejected top16 observations:

- Several rejected rows are plausible hard cases such as floor/crouch motions,
  football gestures, jumps, and high bends, matching the quality-filter intent.
- Some static frames look like ordinary walking despite being rejected by
  p99/max z-score or token-collapse rules.  This suggests the current thresholds
  are conservative and should not be relaxed without temporal MP4 inspection.

### Training path check

To avoid filling the nearly-full `/home` filesystem, the checkpoint directory
was placed under `/tmp`:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_filtered.pt \
  --val-cache stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_filtered.pt \
  --init-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --output-dir /tmp/stage1_batch500_train_head_20260613 \
  --epochs 1 \
  --batch-size 8 \
  --lr 1e-5 \
  --train-scope head \
  --num-workers 0 \
  --gpu 0 \
  --seed 13
```

Result:

```text
GPU not detected. Defaulting to CPU.
train_scope=head trainable_parameters=7880448
epoch=0 train=16.6888/acc=0.0571 val=19.8593/acc=0.0573 elapsed=107.6s
elapsed_sec=116.42
```

This is a training-path check only.  Train and validation used the same cache,
so these numbers do not measure generalization or model improvement.

### Code and verification

The batch tools now support `--quiet` compact JSON output to make larger
experiments easier to log without dumping full per-file payloads:

- `export_humanml3d_to_bvh.py`
- `evaluate_bvh_metrics.py`
- `diagnose_bvh_character_retarget.py`
- `summarize_bvh_retarget_quality.py`
- `build_bvh_character_gpt_cache.py`
- `diagnose_token_distribution.py`

Added `tests/test_stage1_quiet_cli.py` for lightweight CLI coverage.

Verification:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/export_humanml3d_to_bvh.py \
  Script/stage1/evaluate_bvh_metrics.py \
  Script/stage1/diagnose_bvh_character_retarget.py \
  Script/stage1/summarize_bvh_retarget_quality.py \
  Script/stage1/build_bvh_character_gpt_cache.py \
  Script/stage1/diagnose_token_distribution.py \
  tests/test_stage1_quiet_cli.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_quiet_cli \
  -v

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_bvh_metrics \
  tests.test_stage1_bvh_retarget_quality \
  tests.test_stage1_bvh_character_cache \
  tests.test_stage1_bvh_character_retarget \
  tests.test_stage1_humanml3d_bvh_export \
  -v
```

Results:

```text
py_compile passed
tests.test_stage1_quiet_cli: 6 tests passed
related BVH/cache/retarget/export subset: 20 tests passed
```

### Interpretation and next action

- HumanML3D is still the main route.  The current bridge is not perfect, but it
  has now produced 90 accepted samples with a non-collapsed accepted-only token
  distribution.
- The accepted cache is close to a minimum useful size for a conservative
  fine-tune, but it is not yet final training data because it lacks a held-out
  accepted validation split and MP4 temporal audit.
- A real LLM in-context token planning experiment has still not been run.  It
  remains the backup route if a conservative fine-tune on this repaired data
  does not improve multi-stage prompt generation.

Recommended next step:

```text
create accepted-only train/val split from batch500 or a larger batch
-> render MP4 for accepted risk cases and top rejected false-positive candidates
-> train conservative base_head/temporal_base_head model
-> run baseline-vs-finetuned Stage1 model suite on multi-stage prompts
-> record BVH engineering metrics, videos, semantic checklist, and failure modes
```

## 2026-06-13: Batch500 accepted train/val split cache

### Purpose

The previous batch500 cache used all 90 accepted rows as both train and val for
a path check.  This follow-up creates a deterministic accepted-only train/val
split so later MoConGPT fine-tuning has a held-out cache, even if the validation
set is still small.

### Code change

Added:

```text
Script/stage1/split_bvh_quality_summary.py
tests/test_stage1_bvh_quality_split.py
```

The splitter reads a `summarize_bvh_retarget_quality.py` JSON, keeps accepted
rows by default, shuffles with a fixed seed, and writes two quality-summary JSON
files.  The existing `build_bvh_character_gpt_cache.py --quality-summary` path
can then build train and val caches without inventing a new cache format.

### Split command

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/split_bvh_quality_summary.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_summary.json \
  --train-output stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_train_seed13.json \
  --val-output stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_val_seed13.json \
  --seed 13 \
  --val-fraction 0.2 \
  --quiet
```

Result:

```text
train accepted rows = 72
val accepted rows = 18
```

### Train/val cache build

Train cache:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_train_seed13.json \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --window-size 50 \
  --window-stride 25 \
  --rvq-depth 4 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_train_seed13.pt \
  --observation-h5 stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/observations_train_seed13.h5 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_train_seed13_summary.json \
  --quiet
```

The first sandboxed attempt failed with the known MPI/socket error; the same
command succeeded after rerunning outside the sandbox:

```text
windows = 72
valid_tokens = 12552
unique_sequences = 72
elapsed_sec = 27.52
```

Val cache:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_val_seed13.json \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --window-size 50 \
  --window-stride 25 \
  --rvq-depth 4 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_val_seed13.pt \
  --observation-h5 stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/observations_val_seed13.h5 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_val_seed13_summary.json \
  --quiet
```

Result:

```text
windows = 18
valid_tokens = 3252
unique_sequences = 18
elapsed_sec = 13.91
```

### Token distribution

Train split:

| depth | tokens | unique | top frac |
| --- | ---: | ---: | ---: |
| 0 | 3138 | 383 | 0.042384 |
| 1 | 3138 | 484 | 0.010835 |
| 2 | 3138 | 491 | 0.046208 |
| 3 | 3138 | 476 | 0.074570 |

Val split:

| depth | tokens | unique | top frac |
| --- | ---: | ---: | ---: |
| 0 | 813 | 240 | 0.029520 |
| 1 | 813 | 342 | 0.020910 |
| 2 | 813 | 355 | 0.027060 |
| 3 | 813 | 310 | 0.086101 |

The split does not show obvious token collapse.  The val set is small, so its
distribution should be treated as a smoke/early-warning signal rather than a
stable model-selection benchmark.

### Train/val training path check

Command:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_train_seed13.pt \
  --val-cache stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_val_seed13.pt \
  --init-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --output-dir /tmp/stage1_batch500_trainval_head_seed13_20260613 \
  --epochs 1 \
  --batch-size 8 \
  --lr 1e-5 \
  --train-scope head \
  --num-workers 0 \
  --gpu 0 \
  --seed 13
```

Result:

```text
GPU not detected. Defaulting to CPU.
train_scope=head trainable_parameters=7880448
epoch=0 train=16.8986/acc=0.0612 val=17.5875/acc=0.0409 elapsed=65.8s
elapsed_sec = 74.20
```

Detailed log:

```text
train valid_tokens = 12552
train depth_accuracy = [0.1906, 0.0271, 0.0112, 0.0159]
val valid_tokens = 3252
val depth_accuracy = [0.1193, 0.0332, 0.0074, 0.0037]
```

This confirms the split train/val caches are consumable by the real MoConGPT
training path.  It is not yet evidence of model improvement, because only the
head was trained for one epoch on a small accepted set.

### Verification

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/split_bvh_quality_summary.py \
  tests/test_stage1_bvh_quality_split.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_bvh_quality_split \
  -v
```

Results:

```text
py_compile passed
tests.test_stage1_bvh_quality_split: 3 tests passed
```

### Next action

Before a report-level fine-tune, inspect MP4s for accepted risky rows and likely
false-rejected walking rows.  Then run a longer conservative
`base_head`/`temporal_base_head` fine-tune from the train cache, use the val
cache only for early stopping/checking, and compare generated BVH/MP4 against
the baseline GPT on multi-stage prompts.

## 2026-06-13: Batch500 MP4 audit for accepted risk and rejected false-positive candidates

### Purpose

The batch500 contact sheets were useful but only sparse static frames.  Before
using the accepted train/val cache for a report-facing fine-tune, this check
renders a small MP4 audit set:

- accepted rows that looked risky in the contact sheet;
- rejected walking/turning rows that looked plausible statically and may be
  false rejects caused by conservative z-score thresholds.

### Selected samples

Accepted risk candidates:

| label | reason for audit |
| --- | --- |
| `000808` | high p99 among accepted rows, exercise motion |
| `003355` | repeated high-kick motion near depth0 collapse threshold |
| `013481` | walk/drop-to-ground/breast-stroke caption; contact sheet showed floor motion |

Rejected possible false-positive candidates:

| label | reject reasons | reason for audit |
| --- | --- | --- |
| `010684` | `p99_abs_z>8`, `max_abs_z>50` | walking/turning caption looked plausible statically |
| `013114` | `p99_abs_z>8` | simple walking-forward caption |
| `M012928` | `p99_abs_z>8` | multi-turn walking caption looked plausible statically |

### Render command

The audit input directory contains symlinks to the selected BVH files:

```text
stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/mp4_audit_input/
```

Render command:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/render_bvh_to_mp4.py \
  --input stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/mp4_audit_input \
  --output-dir stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/mp4_audit_videos \
  --ffmpeg /home/chenjie/miniconda3/envs/moconvq/bin/ffmpeg \
  --fps 30 \
  --width 960 \
  --height 720 \
  --max-video-frames 240
```

Result:

```text
videos = 6
elapsed_sec = 44.66
```

Generated MP4s:

```text
accepted_000808_exercise.mp4        152 frames, 5.07 s
accepted_003355_high_kick.mp4       199 frames, 6.63 s
accepted_013481_ground.mp4          199 frames, 6.63 s
rejected_010684_walk_turn.mp4       194 frames, 6.47 s
rejected_013114_walk_forward.mp4    199 frames, 6.63 s
rejected_M012928_walk_turn.mp4      199 frames, 6.63 s
```

Screenshot sanity checks were extracted at 2 s and 5 s:

```text
stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/mp4_audit_screenshots/
stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/mp4_audit_screenshots/contact_sheet_t2_t5.png
```

All screenshot files were non-empty and had non-degenerate RGB statistics, so
the MP4s are not blank.

### Engineering metrics

All six audit BVHs have at least 120 frames and are not early-stop cases under
the current engineering metric.

| audit label | accepted | frames | p99 z | max z | depth0 top frac | depth0 unique | root path | root disp | pose vel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `accepted_000808_exercise` | yes | 152 | 7.628 | 18.511 | 0.158 | 22 | 2.296 | 0.424 | 184.551 |
| `accepted_003355_high_kick` | yes | 199 | 6.229 | 24.335 | 0.224 | 20 | 1.176 | 0.052 | 125.139 |
| `accepted_013481_ground` | yes | 199 | 6.555 | 27.278 | 0.245 | 21 | 2.826 | 2.471 | 131.572 |
| `rejected_010684_walk_turn` | no | 194 | 15.993 | 57.024 | 0.146 | 19 | 6.940 | 2.055 | 189.182 |
| `rejected_013114_walk_forward` | no | 199 | 15.851 | 29.700 | 0.122 | 28 | 4.312 | 0.151 | 156.951 |
| `rejected_M012928_walk_turn` | no | 199 | 16.701 | 34.523 | 0.204 | 22 | 6.705 | 0.787 | 152.721 |

### Visual observations from MP4 screenshots

- `accepted_000808_exercise`: upright and readable in the inspected frames, but
  with a leaning/exercise posture.  It looks usable as a challenging accepted
  sample.
- `accepted_003355_high_kick`: upright and readable, but it has extreme leg
  poses.  It may be useful if high-kick motions are desired, but it should not
  dominate a small training set.
- `accepted_013481_ground`: clear floor/prone motion appears by the 5 s
  screenshot.  The filter accepted it because z-score and depth0 thresholds
  passed, but it is a risky sample for the current MoConVQ character bridge and
  should be excluded or separately bucketed for the first conservative
  fine-tune.
- `rejected_010684_walk_turn`: visually plausible walking/turning in the
  inspected frames despite high z-score rejection.  This is a likely
  conservative false reject.
- `rejected_013114_walk_forward`: the inspected frames are upright but show
  bending/leaning; rejection may be reasonable until full video is inspected.
- `rejected_M012928_walk_turn`: visually plausible walking/turning in the
  inspected frames despite high z-score rejection.  This is another likely
  conservative false reject.

### Interpretation

The current quality filter is useful but too blunt for final data selection:

- It catches many difficult floor/crouch/high-bend cases, but accepted rows can
  still contain floor motions such as `013481`.
- Some z-score-only rejected rows look like plausible walking motions in MP4
  screenshots.  A stricter "first fine-tune" subset should exclude accepted
  floor/prone captions, while a later recall-improving pass should review
  z-score-only rejected walking rows before throwing them away.

For the first conservative train run, use the existing batch500 train/val cache
as a path baseline, but consider a filtered-v2 cache that removes floor/prone
accepted rows and possibly restores visually plausible z-score-only walking
rows after full MP4 inspection.

## 2026-06-13: Batch500 filtered-v2 quality summary from MP4 audit

### Purpose

The MP4 audit identified one accepted row that should not be used in the first
conservative fine-tune, and two rejected walking/turning rows that appeared
usable enough to recover.  This experiment turns that audit into a reproducible
quality-summary override and rebuilds train/val caches from the revised summary.

### Code change

Added:

```text
Script/stage1/apply_bvh_quality_overrides.py
tests/test_stage1_bvh_quality_overrides.py
```

The override tool keeps the original quality summary intact and writes a new
summary with explicit `manual_override_summary` metadata.  Rows forced into or
out of the accepted set also receive a `manual_overrides` entry.

### Overrides

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/apply_bvh_quality_overrides.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_summary.json \
  --exclude '013481:mp4_audit_floor_prone_motion' \
  --include '010684:mp4_audit_plausible_walk_turn' \
  --include 'M012928:mp4_audit_plausible_walk_turn' \
  --output-json stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_summary_v2_mp4_audit.json \
  --quiet
```

Result:

```text
original accepted = 90
v2 accepted = 91
v2 rejected = 409

manual exclude:
  013481  mp4_audit_floor_prone_motion

manual include:
  010684   mp4_audit_plausible_walk_turn
  M012928  mp4_audit_plausible_walk_turn
```

### Train/val split

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/split_bvh_quality_summary.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_summary_v2_mp4_audit.json \
  --train-output stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_train_v2_seed13.json \
  --val-output stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_val_v2_seed13.json \
  --seed 13 \
  --val-fraction 0.2 \
  --quiet
```

Result:

```text
train accepted rows = 73
val accepted rows = 18
```

The split summaries point back to `quality_summary_v2_mp4_audit.json` via
`source_summary`; the top-level manual override metadata remains in that source
summary.

### Cache build

Train cache:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_train_v2_seed13.json \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --window-size 50 \
  --window-stride 25 \
  --rvq-depth 4 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_train_v2_seed13.pt \
  --observation-h5 stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/observations_train_v2_seed13.h5 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_train_v2_seed13_summary.json \
  --quiet
```

Result:

```text
windows = 73
valid_tokens = 12836
unique_sequences = 73
elapsed_sec = 27.35
```

Val cache:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/build_bvh_character_gpt_cache.py \
  --quality-summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/quality_val_v2_seed13.json \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --window-size 50 \
  --window-stride 25 \
  --rvq-depth 4 \
  --output stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_val_v2_seed13.pt \
  --observation-h5 stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/observations_val_v2_seed13.h5 \
  --summary stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_val_v2_seed13_summary.json \
  --quiet
```

Result:

```text
windows = 18
valid_tokens = 3160
unique_sequences = 18
elapsed_sec = 13.31
```

### Token distribution

Train v2:

| depth | tokens | unique | top frac |
| --- | ---: | ---: | ---: |
| 0 | 3209 | 375 | 0.037706 |
| 1 | 3209 | 492 | 0.010907 |
| 2 | 3209 | 497 | 0.036772 |
| 3 | 3209 | 480 | 0.077594 |

Val v2:

| depth | tokens | unique | top frac |
| --- | ---: | ---: | ---: |
| 0 | 790 | 214 | 0.039241 |
| 1 | 790 | 325 | 0.021519 |
| 2 | 790 | 328 | 0.060759 |
| 3 | 790 | 308 | 0.065823 |

The v2 split remains non-collapsed.  The val set is still very small, so these
numbers should be treated as a smoke check rather than a stable validation
distribution.

### Train/val training path check

Command:

```bash
/usr/bin/time -f elapsed_sec=%e /home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_train_v2_seed13.pt \
  --val-cache stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_val_v2_seed13.pt \
  --init-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --output-dir /tmp/stage1_batch500_v2_trainval_head_seed13_20260613 \
  --epochs 1 \
  --batch-size 8 \
  --lr 1e-5 \
  --train-scope head \
  --num-workers 0 \
  --gpu 0 \
  --seed 13
```

Result:

```text
GPU not detected. Defaulting to CPU.
train_scope=head trainable_parameters=7880448
epoch=0 train=16.7571/acc=0.0587 val=19.9805/acc=0.0472 elapsed=72.0s
elapsed_sec = 79.92
```

Detailed log:

```text
train valid_tokens = 12836
train depth_accuracy = [0.1736, 0.0337, 0.0128, 0.0150]
val valid_tokens = 3160
val depth_accuracy = [0.1544, 0.0203, 0.0114, 0.0025]
```

This confirms the filtered-v2 train/val caches are consumable by the real
MoConGPT training path.  It is still a path check only, not an improvement claim.

### Verification

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/apply_bvh_quality_overrides.py \
  tests/test_stage1_bvh_quality_overrides.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_bvh_quality_overrides \
  -v
```

Results:

```text
py_compile passed
tests.test_stage1_bvh_quality_overrides: 3 tests passed
```

### Interpretation

The filtered-v2 cache is a better candidate for the first conservative
fine-tune than the raw batch500 accepted split because it incorporates explicit
MP4 audit evidence.  It still remains small and should not be used for a final
paper-level claim by itself.  The next step is to run a longer conservative
`base_head` or `temporal_base_head` fine-tune from this v2 train cache, then
compare baseline vs fine-tuned generation on multi-stage prompts with BVH
metrics, videos, and a semantic checklist.

## 2026-06-13: Batch500 filtered-v2 conservative base_head fine-tune and generation check

This experiment is the first conservative training/generation check using the
MP4-audited batch500 filtered-v2 cache:

```text
HumanML3D joints_ik BVH
  -> MoConVQ-native character retarget
  -> per-file quality filter + MP4-audit overrides
  -> deterministic accepted train/val split
  -> accepted-only GPT cache
  -> protected base_head fine-tune
  -> baseline-vs-finetuned top-p BVH/MP4 comparison
```

It is still small-scale: `73` train windows and `18` val windows.  The purpose
is to test whether the repaired data route can produce a usable MoConGPT
checkpoint without immediately destroying baseline behavior.  It is not a final
Stage1 success claim.

### Training command

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_train_v2_seed13.pt \
  --val-cache stage1_artifacts/humanml_bvh_export_ik_batch500_20260613/gpt_cache_val_v2_seed13.pt \
  --init-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --output-dir /tmp/stage1_batch500_v2_basehead_seed13_5ep_20260613 \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-5 \
  --train-scope base_head \
  --depth-weights 1.0,0.7,0.4,0.2 \
  --baseline-kl-weight 0.05 \
  --kl-temperature 2.0 \
  --end-token-weight 0.01 \
  --num-workers 0 \
  --gpu 0 \
  --seed 13 \
  --save-every 5
```

Runtime context:

```text
PyTorch CUDA was unavailable in the moconvq environment.
Training ran on CPU.
train_scope = base_head
trainable_parameters = 30,577,152
```

Clean 5-epoch result:

| epoch | train loss | val loss | train acc | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 15.8651 | 17.6901 | 0.0566 | 0.0472 |
| 1 | 14.9218 | 17.1599 | 0.0575 | 0.0468 |
| 2 | 14.5209 | 16.5973 | 0.0591 | 0.0462 |
| 3 | 13.8787 | 16.0007 | 0.0580 | 0.0462 |
| 4 | 13.2425 | 15.3757 | 0.0579 | 0.0465 |

The loss curve is monotonic over these five epochs, but token accuracy barely
changes.  This suggests the model is fitting the small filtered-v2 distribution
without yet showing clear token-level generalization.

Important artifact hygiene note:

```text
formal checkpoint for this experiment:
  /tmp/stage1_batch500_v2_basehead_seed13_5ep_20260613/checkpoint_epoch_5.pth

do not use as the formal 5-epoch checkpoint:
  /tmp/stage1_batch500_v2_basehead_seed13_5ep_20260613/best_val.pth
```

After the clean 5-epoch run, an attempted manual resume appended duplicate
epoch `3`/`4` rows in the same `/tmp` directory and overwrote `best_val.pth`.
Therefore the formal 5-epoch comparison below uses `checkpoint_epoch_5.pth`,
which was written at the clean epoch-5 boundary.  The training script has been
updated after this incident so future `--append-log` runs must start from the
next logged epoch and initialize `best_val_loss` from the existing log.

### Top-p generation comparison

Command shape:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/run_stage1_model_suite.py \
  --run-id batch500_v2_basehead_epoch5_top_p_len75_20260613 \
  --suite-dir /tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613 \
  --bvh-dir /tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613/bvh \
  --baseline-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --finetuned-checkpoint /tmp/stage1_batch500_v2_basehead_seed13_5ep_20260613/checkpoint_epoch_5.pth \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-encoder t5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --max-length 75 \
  --generation-mode auto \
  --context-size 30 \
  --chunk-size 20 \
  --top-k 0 \
  --top-p 0.95 \
  --temperature 1.0 \
  --progress-conditioning auto \
  --baseline-progress-conditioning none \
  --progress-scale 0.5 \
  --progress-context-size 51 \
  --progress-prefix-cap 25 \
  --seed 123 \
  --expected-min-frames 1200 \
  --skip-backup
```

Rendered MP4s were then produced from the existing BVHs with the explicit conda
ffmpeg path:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/run_text_gpt_comparison.py \
  --prompts /tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613/prompts.tsv \
  --bvh-dir /tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613/bvh \
  --video-dir /tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613/video \
  --ffmpeg /home/chenjie/miniconda3/envs/moconvq/bin/ffmpeg \
  --skip-generation \
  ...
```

Generated frames:

| prompt | baseline frames | finetuned frames |
| --- | ---: | ---: |
| `walk_turn_wave` | 816 | 816 |
| `circle_crouch_stand` | 1176 | 1368 |
| `walk_jump_dance` | 1392 | 1320 |
| `sidestep_kick_turn` | 864 | 864 |

Model averages from BVH engineering metrics:

| model | avg frames | early stop rate | avg root path | avg pose velocity | avg pose variance | lag20 repeat >0.995 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline top-p | 1062 | 0.75 | 3.472 | 14.052 | 141.194 | 0.000 |
| finetuned top-p | 1092 | 0.50 | 3.073 | 24.086 | 297.008 | 0.000 |

Artifacts:

```text
BVH:
  /tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613/bvh/

metrics:
  /tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613/summary_metrics.json
  /tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613/bvh/summary_metrics_script.json

side-by-side MP4:
  /tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613/video/

contact sheet:
  /tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613/contact_sheet.png
```

Visual/contact-sheet check:

- no obvious full-body inversion or floor-prone collapse was visible in the
  sampled contact sheet;
- `walk_turn_wave` and `sidestep_kick_turn` are nearly unchanged from baseline,
  including the same early-stop length;
- `circle_crouch_stand` is longer than baseline and crosses the 1200-frame
  threshold, but it also has higher pose velocity and pose variance;
- `walk_jump_dance` is slightly shorter than baseline and also has higher pose
  velocity/variance.

### Interpretation

This run is useful evidence that the repaired HumanML3D BVH-to-character route
can produce a trainable MoConGPT cache and a non-collapsed generation
checkpoint.  It does not yet solve Stage1:

- the fine-tuned model only slightly improves average length and early-stop
  rate on four long prompts;
- two prompts are effectively unchanged from baseline;
- higher pose velocity/variance indicates possible instability or more abrupt
  generated motion;
- semantic quality still needs MP4 review beyond the contact sheet;
- paper-level FID/R-precision remains unavailable without the HumanML3D
  evaluator assets.

Current conclusion: batch500 filtered-v2 is a better foundation than the old
hand-written retarget cache, but it is too small and too weak to claim
improvement over baseline.  The next main-route step should scale the accepted
BVH cache beyond batch500 and/or fix CUDA training throughput, then repeat the
same comparison.  The real external/local LLM in-context planning route remains
unrun and should still be treated as a backup path, not as current evidence.

### Code hygiene fix from this run

The attempted resume exposed a training-script bug:

```text
--append-log previously allowed duplicate epoch ids and reset best_val_loss.
```

`Script/stage1/train_real_text_gpt.py` now:

- reads existing `train_log.jsonl` state;
- rejects duplicate existing epoch ids;
- requires `--start-epoch` to be exactly the next epoch after the existing log
  when `--append-log` is used;
- initializes `best_val_loss` from the existing log during append mode;
- names periodic checkpoints by global epoch id, e.g. resumed epoch 4 writes
  `checkpoint_epoch_5.pth`.

## 2026-06-13 True workdir sync and long-H5 BVH-native retarget smoke

### Repository/workdir correction

The true Stage1 experiment workdir is:

```text
/home/chenjie/cc/robotics/MoConVQ
```

The GitHub push target is not `origin/stage1`.  The remote `origin/main`
contains Stage1 under:

```text
stage1/
```

Since Git cannot directly track a remote branch subdirectory as the root of a
normal worktree, the current safe workflow is:

```text
MoConVQ/ experiments and edits
  -> sync code/docs to MoConVQ-main/stage1/
  -> commit from MoConVQ-main
  -> git push origin HEAD:main
```

Added:

```text
Script/stage1/sync_stage1_to_main_worktree.py
```

This helper excludes local experiment outputs, model/data files, and private
agent docs by default.

After syncing, a content comparison excluding local data/artifacts reported no
differences:

```bash
diff -qr \
  --exclude=.git \
  --exclude=stage1_artifacts \
  --exclude=__pycache__ \
  --exclude='*.pyc' \
  --exclude='*.h5' \
  --exclude='*.pth' \
  --exclude='*.data' \
  --exclude=midterm-report \
  --exclude=midterm_figures \
  --exclude=request.txt \
  /home/chenjie/cc/robotics/MoConVQ \
  /home/chenjie/cc/robotics/MoConVQ-main/stage1
```

### Long HumanML3D H5 to BVH bridge

Added:

```text
Script/stage1/export_long_humanml3d_to_bvh.py
```

Purpose:

```text
synthesize_long_humanml3d.py output
  long_sequences.h5 + manifest.jsonl
-> long BVH files using the MoConVQ BVH template
-> MoConVQ native MotionDataSet.add_bvh_with_character()
-> GPT cache
```

Smoke command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/export_long_humanml3d_to_bvh.py \
  --long-h5 stage1_artifacts/long_humanml3d_fixed/train/long_sequences.h5 \
  --manifest stage1_artifacts/long_humanml3d_fixed/train/manifest.jsonl \
  --limit 20 \
  --output-dir /tmp/stage1_long_fixed_bvh_native_smoke_true_workdir_20260613/bvh \
  --summary /tmp/stage1_long_fixed_bvh_native_smoke_true_workdir_20260613/export_summary.json \
  --quiet
```

Result:

```text
exports = 20
rotation_source = joints_ik
```

BVH metrics:

```text
rows = 20
early_stop = 2
```

Native character retarget diagnostic:

```text
paths = 20
state_shape = [8231, 20, 13]
observation_shape = [8231, 323]
token_shape = [2057, 4]
p99_abs_z = 5.6049
max_abs_z = 67.2166
per_file = 20
comparisons = 1
```

Quality summary:

```text
total = 20
accepted = 11
rejected = 9
```

Accepted GPT cache:

```text
windows = 40
valid_tokens = 8000
unique_sequences = 11
```

Token distribution for the accepted native-retarget long cache:

| depth | top fraction | unique tokens |
| --- | ---: | ---: |
| 0 | 0.0530 | 225 |
| 1 | 0.0315 | 342 |
| 2 | 0.0835 | 379 |
| 3 | 0.0755 | 350 |

For comparison, the old hand-written HumanML3D-to-observation cache had much
stronger token collapse:

```text
old fixed cache depth0 top fraction = 0.2171
old fixed cache depth1 top fraction = 0.3342
```

### Interpretation

This smoke result changes the diagnosis of Stage1:

- the existing long-sequence synthesis is not the primary failure source;
- the old hand-written HumanML3D-to-MoConVQ body-state/cache path is the likely
  high-impact failure source;
- the new long-H5-to-BVH bridge lets synthesized long sequences enter MoConVQ
  through its native character retarget path and produces a much healthier RVQ
  token distribution;
- the next main experiment should scale this long BVH-native route, then train
  and compare baseline vs fine-tuned generation on long prompts.

## 2026-06-13 Long HumanML3D BVH-native 200-sequence training run

### Goal

Check the three Stage1 failure candidates under one reproducible run:

```text
long-sequence synthesis
  -> HumanML3D/BVH mapping into MoConVQ body state and observation
  -> text-conditioned MoConGPT fine-tuning and long-prompt generation
```

This run uses the existing synthesized long HumanML3D dataset, exports 200 long
sequences to BVH, filters them through MoConVQ's original
`MotionDataSet.add_bvh_with_character()` path, builds a GPT cache from accepted
samples, trains a conservative `base_head` checkpoint, and compares it against
the original `text_generation_GPT.pth` baseline.

### Data export and retarget quality

Export command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/export_long_humanml3d_to_bvh.py \
  --long-h5 stage1_artifacts/long_humanml3d_fixed/train/long_sequences.h5 \
  --manifest stage1_artifacts/long_humanml3d_fixed/train/manifest.jsonl \
  --limit 200 \
  --output-dir /tmp/stage1_long_fixed_bvh_native_200_20260613/train_bvh \
  --summary /tmp/stage1_long_fixed_bvh_native_200_20260613/export_summary.json \
  --quiet
```

Result:

```text
exports = 200
rotation_source = joints_ik
```

BVH engineering metrics:

```text
rows = 200
early_stop = 27     # expected_min_frames = 240
```

Native retarget diagnostic:

```text
paths = 200
state_shape = [84321, 20, 13]
observation_shape = [84321, 323]
token_shape = [21080, 4]
p99_abs_z = 5.8860
max_abs_z = 90.7866
per_file = 200
comparisons = 1
```

Quality filter:

```text
total = 200
accepted = 91
rejected = 109
```

Main reject reasons:

| reason | count |
| --- | ---: |
| depth0_top_frac > 0.25 | 64 |
| max_abs_z > 50 | 52 |
| p99_abs_z > 8 | 46 |
| depth0_unique < 16 | 19 |
| frames < 120 | 5 |

Interpretation: long-sequence duration is not the major problem.  Most rejected
samples fail because the MoConVQ-native retargeted observation becomes
out-of-distribution or the depth0 token stream collapses for that sample.  This
points to HumanML3D-to-character mapping/retarget quality as the current main
data bottleneck.

### GPT cache

Accepted rows were split with seed 13:

```text
train sequences = 73
val sequences = 18
```

Cache summaries:

| split | windows | valid RVQ tokens | unique sequences |
| --- | ---: | ---: | ---: |
| train | 278 | 55,516 | 73 |
| val | 66 | 13,200 | 18 |

Token distribution:

| split | depth0 top frac | depth1 top frac | depth2 top frac | depth3 top frac |
| --- | ---: | ---: | ---: | ---: |
| train | 0.0626 | 0.0219 | 0.0481 | 0.0709 |
| val | 0.0473 | 0.0382 | 0.0424 | 0.0803 |

For comparison, the old hand-written fixed cache had:

```text
depth0 top fraction = 0.2171
depth1 top fraction = 0.3342
```

Conclusion: the BVH-native cache is much less collapsed than the old
hand-written HumanML3D-to-observation cache.

### Training

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/train_real_text_gpt.py \
  --train-cache /tmp/stage1_long_fixed_bvh_native_200_20260613/train_cache.pt \
  --val-cache /tmp/stage1_long_fixed_bvh_native_200_20260613/val_cache.pt \
  --init-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --output-dir /tmp/stage1_long_fixed_bvh_native_200_basehead_seed13_5ep_20260613 \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --gpu 0 \
  --seed 13 \
  --save-every 1 \
  --num-workers 2 \
  --train-scope base_head \
  --baseline-kl-weight 0.05 \
  --kl-temperature 2.0 \
  --progress-conditioning auto \
  --progress-scale 0.5 \
  --teacher-progress-conditioning none \
  --context-size 51
```

Training curve:

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8.5605 | 0.0422 | 7.9130 | 0.0531 |
| 1 | 7.4678 | 0.0453 | 7.1523 | 0.0568 |
| 2 | 6.9107 | 0.0503 | 6.6901 | 0.0643 |
| 3 | 6.5063 | 0.0580 | 6.4046 | 0.0688 |
| 4 | 6.2825 | 0.0643 | 6.2298 | 0.0739 |

Checkpoint used for generation:

```text
/tmp/stage1_long_fixed_bvh_native_200_basehead_seed13_5ep_20260613/checkpoint_epoch_5.pth
```

Interpretation: the repaired cache and training code are working.  Loss and
accuracy move in the expected direction, unlike the earlier invalid/weak runs.

### Baseline vs fine-tuned generation

Generation suite:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/run_stage1_model_suite.py \
  --run-id long_fixed_native200_basehead_epoch5_20260613 \
  --suite-dir /tmp/stage1_long_fixed_bvh_native_200_basehead_epoch5_compare_20260613 \
  --baseline-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --finetuned-checkpoint /tmp/stage1_long_fixed_bvh_native_200_basehead_seed13_5ep_20260613/checkpoint_epoch_5.pth \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-encoder t5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --max-length 75 \
  --generation-mode auto \
  --context-size 30 \
  --chunk-size 20 \
  --top-k 0 \
  --top-p 0.95 \
  --temperature 1.0 \
  --progress-conditioning auto \
  --baseline-progress-conditioning none \
  --progress-scale 0.5 \
  --progress-context-size 51 \
  --progress-prefix-cap 25 \
  --seed 123 \
  --expected-min-frames 1200 \
  --skip-backup
```

Frames by prompt:

| prompt | baseline frames | fine-tuned frames | baseline early stop | fine-tuned early stop |
| --- | ---: | ---: | --- | --- |
| walk_turn_wave | 816 | 864 | true | true |
| circle_crouch_stand | 1176 | 1296 | true | false |
| walk_jump_dance | 1392 | 1656 | false | false |
| sidestep_kick_turn | 864 | 864 | true | true |

Model averages:

| model | avg frames | early stop rate | avg root path | avg root displacement | avg pose velocity | avg pose variance | lag20 repeat >0.995 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline top-p | 1062 | 0.75 | 3.472 | 1.268 | 14.052 | 141.194 | 0.000 |
| fine-tuned top-p | 1170 | 0.50 | 3.538 | 1.231 | 29.972 | 277.800 | 0.000 |

Artifacts:

```text
BVH:
  /tmp/stage1_long_fixed_bvh_native_200_basehead_epoch5_compare_20260613/bvh/

metrics:
  /tmp/stage1_long_fixed_bvh_native_200_basehead_epoch5_compare_20260613/summary_metrics.json
  /tmp/stage1_long_fixed_bvh_native_200_basehead_epoch5_compare_20260613/bvh/summary_metrics_script.json

side-by-side MP4:
  /tmp/stage1_long_fixed_bvh_native_200_basehead_epoch5_compare_20260613/video/

contact sheet:
  /tmp/stage1_long_fixed_bvh_native_200_basehead_epoch5_compare_20260613/contact_sheet.png
```

Visual/contact-sheet audit:

- no obvious fall-over, full-body inversion, or broken skeleton was visible in
  the sampled contact sheet;
- fine-tuned output is longer for 3 of 4 prompts;
- `circle_crouch_stand` crosses the 1200-frame threshold after fine-tuning;
- `walk_jump_dance` becomes substantially longer but also has much higher pose
  velocity/variance, so its extra motion is more energetic and less smooth;
- `sidestep_kick_turn` remains effectively unchanged in length and still early
  stops.

### Paper metrics readiness

Readiness check:

```text
paper_metrics_ready = false
missing:
  - HumanML3D text-motion evaluator source files
  - pretrained HumanML3D evaluator / motion-feature extractor checkpoints
```

Therefore this run can claim improvement only on Stage1 engineering metrics
and visual audit, not on MoConVQ paper metrics FID/R-precision.

### Conclusion

This is the first main-route HumanML3D experiment that gives a measurable
generation-side improvement over the original MoConVQ text GPT baseline:

```text
average frames:     1062 -> 1170
early-stop rate:    0.75 -> 0.50
root path length:   3.472 -> 3.538
```

It also identifies the remaining bottleneck:

```text
200 exported long sequences
  -> 91 accepted by native retarget quality filter
  -> 109 rejected mostly for token collapse or observation z-score outliers
```

So the current Stage1 answer is:

- long HumanML3D synthesis is workable;
- the old hand-written HumanML3D-to-MoConVQ cache path should not be used for
  final training claims;
- BVH-native retarget is the working replacement path;
- conservative GPT fine-tuning on the native cache works and improves length
  and early-stop engineering metrics;
- visual quality is somewhat better than previous failed fine-tunes, but still
  not fully stable because pose velocity/variance increase;
- FID/R-precision still require missing HumanML3D evaluator assets.

## 2026-06-13 Long-native head-only conservative ablation

### Motivation

The `base_head` long-native run improved length and early-stop rate, but pose
velocity and pose variance increased strongly:

```text
pose velocity: 14.052 -> 29.972
pose variance: 141.194 -> 277.800
```

To check whether the instability comes from updating too many GPT layers, I ran
the same cache and generation setup with:

```text
--train-scope head
```

This trains only the transformer head parameters:

```text
trainable_parameters = 7,880,448
```

### Training

Output directory:

```text
/tmp/stage1_long_fixed_bvh_native_200_head_seed13_5ep_20260613
```

Training curve:

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 9.0885 | 0.0414 | 8.9803 | 0.0523 |
| 1 | 8.8491 | 0.0420 | 8.8365 | 0.0523 |
| 2 | 8.7996 | 0.0428 | 8.6999 | 0.0534 |
| 3 | 8.6119 | 0.0436 | 8.5675 | 0.0549 |
| 4 | 8.4628 | 0.0453 | 8.4427 | 0.0560 |

This trains more weakly than `base_head` but still moves in the expected
direction.

### Generation comparison

Checkpoint:

```text
/tmp/stage1_long_fixed_bvh_native_200_head_seed13_5ep_20260613/checkpoint_epoch_5.pth
```

Suite:

```text
/tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613
```

Frames by prompt:

| prompt | baseline frames | fine-tuned frames | baseline early stop | fine-tuned early stop |
| --- | ---: | ---: | --- | --- |
| walk_turn_wave | 816 | 864 | true | true |
| circle_crouch_stand | 1176 | 1656 | true | false |
| walk_jump_dance | 1392 | 1392 | false | false |
| sidestep_kick_turn | 864 | 864 | true | true |

Model averages:

| model | avg frames | early stop rate | avg root path | avg root displacement | avg pose velocity | avg pose variance | lag20 repeat >0.995 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline top-p | 1062 | 0.75 | 3.472 | 1.268 | 14.052 | 141.194 | 0.000 |
| head fine-tuned top-p | 1194 | 0.50 | 4.187 | 1.361 | 19.133 | 190.011 | 0.002 |
| base_head fine-tuned top-p | 1170 | 0.50 | 3.538 | 1.231 | 29.972 | 277.800 | 0.000 |

Visual/contact-sheet audit:

- no obvious fall-over or full-body inversion was visible in the sampled
  contact sheet;
- `circle_crouch_stand` is much longer than baseline and crosses the
  1200-frame threshold;
- `walk_jump_dance` and `sidestep_kick_turn` are mostly unchanged in length;
- compared with `base_head`, the head-only model keeps a better stability
  tradeoff: similar early-stop improvement, higher average frames, and much
  smaller pose velocity/variance increase.

Current recommended checkpoint for Stage1 engineering comparison:

```text
/tmp/stage1_long_fixed_bvh_native_200_head_seed13_5ep_20260613/checkpoint_epoch_5.pth
```

Current recommended comparison artifacts:

```text
BVH:
  /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/bvh/

side-by-side MP4:
  /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/video/

contact sheet:
  /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/contact_sheet.png
```

## 2026-06-13 Stage1 automatic run report helper

Added:

```text
Script/stage1/summarize_stage1_run.py
tests/test_stage1_run_summary.py
```

Purpose:

```text
quality summary
  + cache summaries
  + token distribution diagnostics
  + train_log.jsonl
  + baseline-vs-finetuned BVH metrics
  + video summary
  + evaluator readiness
-> report.json + report.md
```

This reduces manual transcription errors when moving numbers into the final
report, and keeps the paper-metric readiness gate visible next to engineering
metrics.

Generated the current best-run report with:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/summarize_stage1_run.py \
  --run-name long_fixed_native200_head_epoch5_20260613 \
  --quality-summary /tmp/stage1_long_fixed_bvh_native_200_20260613/quality_summary.json \
  --train-cache-summary /tmp/stage1_long_fixed_bvh_native_200_20260613/train_cache_summary.json \
  --val-cache-summary /tmp/stage1_long_fixed_bvh_native_200_20260613/val_cache_summary.json \
  --train-token-distribution /tmp/stage1_long_fixed_bvh_native_200_20260613/train_token_distribution.json \
  --val-token-distribution /tmp/stage1_long_fixed_bvh_native_200_20260613/val_token_distribution.json \
  --train-log /tmp/stage1_long_fixed_bvh_native_200_head_seed13_5ep_20260613/train_log.jsonl \
  --metrics-json /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/summary_metrics.json \
  --comparison-video-summary /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/video/summary.json \
  --evaluation-readiness /tmp/stage1_eval_readiness_long_native_200_20260613.json \
  --checkpoint /tmp/stage1_long_fixed_bvh_native_200_head_seed13_5ep_20260613/checkpoint_epoch_5.pth \
  --notes "Current best Stage1 engineering run: long HumanML3D -> BVH -> MoConVQ native retarget, head-only conservative fine-tune." \
  --output-json /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/stage1_run_report.json \
  --output-md /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/stage1_run_report.md
```

Report output:

```text
/tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/stage1_run_report.json
/tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/stage1_run_report.md
```

Key report contents:

```text
accepted rate = 0.455
train cache = 278 windows, 55,516 valid tokens
val cache = 66 windows, 13,200 valid tokens
train token top fractions = depth0 0.063, depth1 0.022, depth2 0.048, depth3 0.071
val token top fractions = depth0 0.047, depth1 0.038, depth2 0.042, depth3 0.080
val loss = 8.980 -> 8.443
avg frames = 1062 -> 1194
early stop rate = 0.75 -> 0.50
root path length = 3.472 -> 4.187
pose velocity = 14.052 -> 19.133
pose variance = 141.194 -> 190.011
paper_metrics_ready = false
```

## 2026-06-13 Prompt segmentation and evaluator readiness follow-up

### Then-aware long prompt generation

I rechecked the current inference path after the question about whether long
prompts are split by `then`.

The answer is yes for the main generation path used in the current best run:

```text
--generation-mode auto
--segment-joiner " then "
```

In `Script/stage1/generate_long_motion.py`, `resolve_generation_mode("auto",
text, " then ")` calls `split_text_segments()`.  If the prompt contains more
than one non-empty segment after splitting on the joiner, it switches to
segmented generation; otherwise it uses rolling generation.

The segmented path:

```text
long prompt
  -> split by " then "
  -> encode each segment text separately
  -> sample each segment with previous generated latents as context/prefix
  -> concatenate segment latents
  -> decode BVH
```

This means prompts like:

```text
a person walks forward then turns around then waves both arms
```

are not treated as one monolithic fixed text condition in the current default
`auto` setup.  They are split into local text conditions while preserving motion
history across segments.

Existing regression tests cover this behavior:

```text
tests/test_stage1_real_generate.py
  test_auto_generation_mode_selects_segmented_for_joined_text
  test_segmented_generation_uses_local_text_per_segment
  test_segmented_generation_carries_previous_segment_latents_as_context
```

Current limitation: the splitter is still literal joiner-based.  It handles the
default English `" then "` separator used by Stage1 synthetic captions and
prompt suites, but it does not yet parse Chinese "然后" or more general
multi-clause syntax.

### Paper-metric readiness check strengthened

I strengthened `Script/stage1/check_evaluation_readiness.py` so the readiness
report explicitly checks a T2M-GPT/text-to-motion-compatible evaluator layout.
The expected source files and assets are:

```text
models/evaluator_wrapper.py
utils/eval_trans.py
options/get_eval_option.py
checkpoints/t2m/text_mot_match/model/finest.tar
checkpoints/t2m/text_mot_match/opt.txt
glove/our_vab_data.npy
glove/our_vab_words.pkl
```

Validation:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/check_evaluation_readiness.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_evaluation_readiness -v
```

Result:

```text
3 tests passed
```

Current local readiness:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/check_evaluation_readiness.py \
  --repo-root . \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --output /tmp/stage1_eval_readiness_current.json
```

Result:

```text
paper_metrics_ready = false
missing:
  - HumanML3D text-motion evaluator source files
  - pretrained HumanML3D evaluator / motion-feature extractor checkpoints
```

T2M-GPT source-only inspection:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/check_evaluation_readiness.py \
  --repo-root . \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --evaluator-root /tmp/T2M-GPT-stage1-inspect \
  --output /tmp/stage1_eval_readiness_t2m_source_only.json
```

Result:

```text
detected source files:
  - models/evaluator_wrapper.py
  - utils/eval_trans.py
  - options/get_eval_option.py

missing assets:
  - checkpoints/t2m/text_mot_match/model/finest.tar
  - checkpoints/t2m/text_mot_match/opt.txt
  - glove/our_vab_data.npy
  - glove/our_vab_words.pkl

paper_metrics_ready = false
```

Important remaining gap:

```text
Even after the evaluator assets are available, generated MoConVQ BVH/character
motion still needs a conversion path back to HumanML3D 263-d motion features
before FID/R-precision are directly comparable to the MoConVQ paper.
```

## 2026-06-13 Generation-mode ablation: then-segmented vs rolling

### Motivation

After confirming that `--generation-mode auto` splits multi-stage prompts by
the default `" then "` joiner, I ran a direct ablation to check whether this
choice helps the current best checkpoint.

Only the generation mode was changed:

```text
checkpoint: /tmp/stage1_long_fixed_bvh_native_200_head_seed13_5ep_20260613/checkpoint_epoch_5.pth
prompts: same 4 Stage1 multi-stage prompts
seed: 123
top-p: 0.95
top-k: 0
temperature: 1.0
max-length: 75
context-size: 30
chunk-size: 20
progress-scale: 0.5
```

Existing best run:

```text
--generation-mode auto
```

Since all four prompts contain `" then "`, this resolves to segmented
generation.

New ablation run:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/run_stage1_model_suite.py \
  --run-id long_fixed_native200_head_epoch5_rolling_ablation_20260613 \
  --suite-dir /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_rolling_ablation_20260613 \
  --baseline-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --finetuned-checkpoint /tmp/stage1_long_fixed_bvh_native_200_head_seed13_5ep_20260613/checkpoint_epoch_5.pth \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --motion-dataset /home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --text-encoder t5 \
  --max-text-length 256 \
  --max-length 75 \
  --generation-mode rolling \
  --context-size 30 \
  --chunk-size 20 \
  --top-k 0 \
  --top-p 0.95 \
  --temperature 1.0 \
  --progress-conditioning auto \
  --baseline-progress-conditioning none \
  --progress-scale 0.5 \
  --progress-context-size 51 \
  --progress-prefix-cap 25 \
  --seed 123 \
  --gpu 0 \
  --expected-min-frames 1200 \
  --skip-backup
```

Outputs:

```text
/tmp/stage1_long_fixed_bvh_native_200_head_epoch5_rolling_ablation_20260613/suite_summary.json
/tmp/stage1_long_fixed_bvh_native_200_head_epoch5_rolling_ablation_20260613/summary_metrics.json
/tmp/stage1_long_fixed_bvh_native_200_head_epoch5_rolling_ablation_20260613/bvh/
```

### Results

Model averages:

| generation mode | model | avg frames | early stop rate | avg root path | avg root displacement | avg pose velocity | avg pose variance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auto / then-segmented | baseline top-p | 1062 | 0.75 | 3.472 | 1.268 | 14.052 | 141.194 |
| auto / then-segmented | head fine-tuned top-p | 1194 | 0.50 | 4.187 | 1.361 | 19.133 | 190.011 |
| forced rolling | baseline top-p | 768 | 0.75 | 3.166 | 1.020 | 37.226 | 380.162 |
| forced rolling | head fine-tuned top-p | 930 | 0.50 | 4.279 | 1.627 | 39.383 | 374.359 |

Frames by prompt:

| prompt | segmented baseline | segmented fine-tuned | rolling baseline | rolling fine-tuned |
| --- | ---: | ---: | ---: | ---: |
| walk_turn_wave | 816 | 864 | 432 | 432 |
| circle_crouch_stand | 1176 | 1656 | 816 | 1416 |
| walk_jump_dance | 1392 | 1392 | 1416 | 1416 |
| sidestep_kick_turn | 864 | 864 | 408 | 456 |

### Interpretation

The `then`-aware segmented mode is better for the current Stage1 long prompts:

```text
segmented fine-tuned avg frames: 1194
rolling fine-tuned avg frames:    930

segmented fine-tuned pose velocity / variance: 19.133 / 190.011
rolling fine-tuned pose velocity / variance:   39.383 / 374.359
```

Rolling still preserves the same early-stop rate difference between baseline
and fine-tuned models, but it produces substantially shorter and more energetic
motions.  The recommended Stage1 inference setting therefore remains:

```text
--generation-mode auto
--segment-joiner " then "
```

or explicitly:

```text
--generation-mode segmented
```

for the current synthetic long-caption convention.

## 2026-06-13 BVH-to-HumanML3D feature adapter smoke

### Motivation

The paper metrics used by MoConVQ for Text2Motion are FID and R-precision on
HumanML3D/SMPL-style motion features.  T2M-GPT's evaluator confirms that the
T2M setting uses:

```text
dim_pose = 263
motion input = HumanML3D new_joint_vecs
motion encoder input = motions[..., :-4]
```

Therefore generated MoConVQ BVH files cannot be passed directly to the
HumanML3D evaluator.  They first need a BVH/character-motion to HumanML3D
263-d feature adapter.

### Added adapter

Added:

```text
Script/stage1/bvh_to_humanml3d_features.py
tests/test_stage1_bvh_to_humanml3d_features.py
```

The adapter path is:

```text
generated MoConVQ/base.bvh
  -> parse BVH + forward kinematics
  -> resample to 20 FPS
  -> map BVH nodes to approximate HumanML3D 22 joints
  -> call HumanML3D scripts/generate_motion_representation.py::process_file()
  -> write 263-d new_joint_vecs .npy
```

Mapping note:

```text
Directly corresponding BVH nodes are copied into HumanML3D joints by name.
HumanML3D spine/neck/head intermediate joints 9, 12, and 15 are approximated
from the MoConVQ torso_head joint and its end site.
```

Because this is a skeleton approximation, it is not yet a paper-level metric
adapter by itself.  It removes the previous hard blocker, but it still needs
calibration against known HumanML3D examples or another trusted BVH/SMPL export.

### Validation

Commands:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/bvh_to_humanml3d_features.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_bvh_to_humanml3d_features -v
```

Result:

```text
3 tests passed
```

Real generated BVH smoke:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/bvh_to_humanml3d_features.py \
  /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/bvh/circle_crouch_stand__finetuned_top_p.bvh \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --output-dir /tmp/stage1_bvh_to_humanml3d_smoke \
  --save-joints \
  --summary /tmp/stage1_bvh_to_humanml3d_smoke/summary.json
```

Smoke output:

```text
input BVH:
  /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/bvh/circle_crouch_stand__finetuned_top_p.bvh

source frames:    1656
source fps:       120.0048
target fps:       20
resampled joints: 276 frames
feature output:
  /tmp/stage1_bvh_to_humanml3d_smoke/new_joint_vecs/circle_crouch_stand__finetuned_top_p.npy
feature shape:
  275 x 263
```

### Readiness update

I updated `check_evaluation_readiness.py` so the paper-metric gate now checks:

```text
1. HumanML3D/T2M evaluator source/assets
2. Stage1 BVH-to-HumanML3D feature adapter
```

Validation:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_evaluation_readiness \
  tests.test_stage1_bvh_to_humanml3d_features \
  -v
```

Result:

```text
7 tests passed
```

Current local readiness:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/check_evaluation_readiness.py \
  --repo-root . \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --output /tmp/stage1_eval_readiness_after_adapter.json
```

Result:

```text
paper_metrics_ready = false

missing:
  - HumanML3D text-motion evaluator source files
  - pretrained HumanML3D evaluator / motion-feature extractor checkpoints

bvh_to_humanml3d_adapter:
  exists = true
  status = available_approximate_adapter_needs_metric_calibration
```

So the immediate remaining paper-metric blockers after the adapter smoke are:

```text
1. obtain T2M evaluator assets;
2. download or copy:
   - checkpoints/t2m/text_mot_match/model/finest.tar
   - checkpoints/t2m/text_mot_match/opt.txt
   - glove/our_vab_data.npy
   - glove/our_vab_words.pkl
3. calibrate the approximate BVH-to-HumanML3D feature adapter before using
   FID/R-precision as final paper-level claims.
```

## 2026-06-13 T2M evaluator asset preparation helper and download attempt

### Motivation

After adding the BVH-to-HumanML3D feature adapter, the remaining paper-metric
blocker is external evaluator assets.  T2M-GPT's official instructions download
the evaluator extractor and glove resources from Google Drive using `gdown`.

### gdown installation

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m pip install gdown
```

Result:

```text
Successfully installed beautifulsoup4-4.15.0 gdown-5.2.2 soupsieve-2.7
```

### Download attempt

Without proxy:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m gdown \
  --fuzzy https://drive.google.com/file/d/1FIiqtkt4F-GVWmnBgtZnv9W3cPWS-oM-/view \
  -O /tmp/stage1_t2m_evaluator_assets/downloads/t2m.zip
```

Result:

```text
Network is unreachable
```

With the configured proxy:

```bash
export http_proxy="http://127.0.0.1:7898"
export https_proxy="http://127.0.0.1:7898"

/home/chenjie/miniconda3/envs/moconvq/bin/python -m gdown \
  --fuzzy https://drive.google.com/file/d/1FIiqtkt4F-GVWmnBgtZnv9W3cPWS-oM-/view \
  -O /tmp/stage1_t2m_evaluator_assets/downloads/t2m.zip
```

Result:

```text
connection succeeded
t2m.zip size: 1.22GB
after 2m42s: about 31.5MB downloaded
speed dropped from about 300KB/s to about 70KB/s
download interrupted manually because expected completion was too long
partial zip removed
```

So evaluator assets remain unavailable locally.  This is now an external asset
availability / transfer-speed problem, not a missing-code problem.

### Added helper

Added:

```text
Script/stage1/prepare_t2m_evaluator_assets.py
tests/test_stage1_prepare_t2m_evaluator_assets.py
```

Capabilities:

```text
check an evaluator asset root;
copy required evaluator source files from a T2M-GPT checkout;
unpack already-downloaded t2m.zip and glove.zip;
print the official gdown/proxy/unpack/readiness commands.
```

Validation:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/prepare_t2m_evaluator_assets.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_prepare_t2m_evaluator_assets -v
```

Result:

```text
5 tests passed
```

Copied evaluator sources from the T2M-GPT inspection checkout:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/prepare_t2m_evaluator_assets.py \
  --root /tmp/stage1_t2m_evaluator_assets \
  --source-root /tmp/T2M-GPT-stage1-inspect \
  --copy-sources
```

Result:

```text
copied:
  - models/evaluator_wrapper.py
  - utils/eval_trans.py
  - options/get_eval_option.py

missing assets:
  - checkpoints/t2m/text_mot_match/model/finest.tar
  - checkpoints/t2m/text_mot_match/opt.txt
  - glove/our_vab_data.npy
  - glove/our_vab_words.pkl
```

Readiness after copying sources:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/check_evaluation_readiness.py \
  --repo-root . \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --evaluator-root /tmp/stage1_t2m_evaluator_assets \
  --output /tmp/stage1_eval_readiness_t2m_sources_copied.json
```

Result:

```text
paper_metrics_ready = false
missing:
  - pretrained HumanML3D evaluator / motion-feature extractor checkpoints

t2m_evaluator.detected_source_files:
  - models/evaluator_wrapper.py
  - utils/eval_trans.py
  - options/get_eval_option.py

bvh_to_humanml3d_adapter.exists = true
```

Next evaluator-assets route:

```text
Either run the helper's printed gdown commands as a long background download,
use a faster mirror/manual copy for t2m.zip and glove.zip, or place the four
required files directly under /tmp/stage1_t2m_evaluator_assets before rerunning
check_evaluation_readiness.py.
```

## 2026-06-13 BVH-to-HumanML3D adapter calibration

### Motivation

The BVH-to-HumanML3D feature adapter removes the hard blocker for feeding
generated MoConVQ BVHs into a HumanML3D/T2M evaluator, but it is approximate.
Before using it for paper-style FID/R-precision, I calibrated the adapter with a
roundtrip test on known HumanML3D motions:

```text
HumanML3D new_joints/new_joint_vecs
  -> export_humanml3d_to_bvh.py using MoConVQ/base.bvh and joints_ik
  -> bvh_to_humanml3d_features.py
  -> reconstructed HumanML3D 22-joint positions + 263-d feature
  -> compare against the original HumanML3D arrays
```

### Added script and tests

Added:

```text
Script/stage1/calibrate_bvh_to_humanml3d_adapter.py
tests/test_stage1_bvh_to_humanml3d_calibration.py
```

The calibration script reports:

```text
feature MAE/RMSE/p95/max
standardized feature z MAE/RMSE/p95/max using HumanML3D Mean.npy/Std.npy
joint MPJPE mean/p95
root-position error mean/p95
local joint MPJPE after subtracting root
```

Validation:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/calibrate_bvh_to_humanml3d_adapter.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_bvh_to_humanml3d_calibration \
  tests.test_stage1_bvh_to_humanml3d_features \
  -v
```

Result:

```text
8 tests passed
```

### Three-sample smoke

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/calibrate_bvh_to_humanml3d_adapter.py \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --sample-id 000021 \
  --sample-id 000001 \
  --sample-id 000002 \
  --output-dir /tmp/stage1_bvh_to_humanml3d_calibration_smoke \
  --summary /tmp/stage1_bvh_to_humanml3d_calibration_smoke/summary.json
```

Result:

```text
samples:                 3
avg feature MAE:         0.0693
avg feature RMSE:        0.1450
avg feature z MAE:       0.2938
avg feature z RMSE:      0.5262
avg feature z p95 abs:   1.2715
avg joint MPJPE:         0.0787
avg local joint MPJPE:   0.0787
avg root position error: 4.71e-7
```

### Test-split 20-sample calibration

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/calibrate_bvh_to_humanml3d_adapter.py \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --split test \
  --limit 20 \
  --seed 13 \
  --output-dir /tmp/stage1_bvh_to_humanml3d_calibration_test20_20260613 \
  --summary /tmp/stage1_bvh_to_humanml3d_calibration_test20_20260613/summary.json
```

Result:

```text
samples:                      20
avg feature MAE:              0.0722
max feature MAE:              0.1009
avg feature RMSE:             0.1540
max feature RMSE:             0.2115
avg feature p95 abs:          0.3951
avg feature z MAE:            0.3049
max feature z MAE:            0.4637
avg feature z RMSE:           0.5694
max feature z RMSE:           0.9235
avg feature z p95 abs:        1.3696
max feature z p95 abs:        1.8286
avg joint MPJPE:              0.0796
max joint MPJPE:              0.0841
avg local joint MPJPE:        0.0796
avg root position error:      4.85e-7
max root position error mean: 5.15e-7
```

### Interpretation

The root trajectory roundtrip is effectively exact, so BVH root translation and
20 FPS timing are stable.  The remaining error is dominated by skeleton
adaptation: MoConVQ/base.bvh has a 20-body character hierarchy, while HumanML3D
uses 22 joints and has extra spine/neck/head joints.  Those joints are currently
approximated from the MoConVQ torso/head chain.

This means the adapter is good enough for engineering readiness and for a
clearly labeled approximate evaluator-adapter route, but it is not a zero-error
bridge.  If the T2M evaluator assets become available, any FID/R-precision
computed through this adapter must be reported with this calibration caveat.  It
should not be presented as native HumanML3D/SMPL evaluation.

The current paper-metric blockers are therefore:

```text
1. missing pretrained T2M evaluator checkpoint/glove assets;
2. nonzero BVH-to-HumanML3D adapter calibration error, which must be disclosed
   or reduced before strong paper-level claims.
```

## 2026-06-13 T2M paper-metric runner skeleton and readiness tightening

### Motivation

After adding the BVH-to-HumanML3D adapter and calibration, the next paper-metric
gap was not only missing external assets.  The previous readiness gate was also
too weak: it checked `evaluator_wrapper.py`, `eval_trans.py`, and
`get_eval_option.py`, but T2M-GPT's evaluator imports additional files:

```text
models/modules.py
utils/word_vectorizer.py
```

and `WordVectorizer('./glove', 'our_vab')` requires:

```text
glove/our_vab_idx.pkl
```

Without these, readiness could report a false ready state and fail only at
runtime.

### Code changes

Updated evaluator readiness / asset helper requirements:

```text
Script/stage1/check_evaluation_readiness.py
Script/stage1/prepare_t2m_evaluator_assets.py
```

Required source files are now:

```text
models/evaluator_wrapper.py
models/modules.py
utils/eval_trans.py
utils/word_vectorizer.py
options/get_eval_option.py
```

Required assets are now:

```text
checkpoints/t2m/text_mot_match/model/finest.tar
checkpoints/t2m/text_mot_match/opt.txt
glove/our_vab_data.npy
glove/our_vab_words.pkl
glove/our_vab_idx.pkl
```

Added a Stage1 paper-metric runner skeleton:

```text
Script/stage1/evaluate_t2m_paper_metrics.py
tests/test_stage1_t2m_paper_metrics.py
```

The runner supports:

```text
generated MoConVQ/base.bvh files
  -> prompt TSV mapping from prompt id to caption/tokens
  -> approximate BVH-to-HumanML3D 263-d feature conversion
  -> HumanML3D Mean/Std normalization
  -> T2M evaluator text/motion embeddings
  -> FID against a HumanML3D reference split
  -> R-precision and matching score per generated model group
```

It also supports `--check-only`, which reports planned inputs and missing
evaluator files without importing the evaluator or requiring checkpoint assets.

### Validation

Commands:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/evaluate_t2m_paper_metrics.py \
  Script/stage1/prepare_t2m_evaluator_assets.py \
  Script/stage1/check_evaluation_readiness.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_t2m_paper_metrics \
  tests.test_stage1_prepare_t2m_evaluator_assets \
  tests.test_stage1_evaluation_readiness \
  -v
```

Result:

```text
14 tests passed
```

### Local evaluator source update

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/prepare_t2m_evaluator_assets.py \
  --root /tmp/stage1_t2m_evaluator_assets \
  --source-root /tmp/T2M-GPT-stage1-inspect \
  --copy-sources
```

Result:

```text
copied:
  - models/evaluator_wrapper.py
  - models/modules.py
  - utils/eval_trans.py
  - utils/word_vectorizer.py
  - options/get_eval_option.py

sources_ready = true
assets_ready  = false
```

### Check-only paper-metric route on current best BVHs

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/evaluate_t2m_paper_metrics.py \
  /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/bvh/*.bvh \
  --prompts /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/prompts.tsv \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --evaluator-root /tmp/stage1_t2m_evaluator_assets \
  --output-dir /tmp/stage1_t2m_paper_metrics_check_20260613 \
  --summary /tmp/stage1_t2m_paper_metrics_check_20260613/summary_after_sources.json \
  --check-only
```

Result:

```text
ready = false
sources_ready = true
assets_ready = false

planned prompts:
  - circle_crouch_stand
  - sidestep_kick_turn
  - walk_jump_dance
  - walk_turn_wave

planned models:
  - baseline_top_p
  - finetuned_top_p

missing assets:
  - checkpoints/t2m/text_mot_match/model/finest.tar
  - checkpoints/t2m/text_mot_match/opt.txt
  - glove/our_vab_data.npy
  - glove/our_vab_words.pkl
  - glove/our_vab_idx.pkl
```

### Interpretation

This closes the code-side paper-metric preparation gap: once the five evaluator
assets are present under `/tmp/stage1_t2m_evaluator_assets`, the same runner can
be executed without `--check-only` to produce approximate T2M evaluator FID,
R-precision, and matching score for the current baseline/finetuned BVH suite.

It still does not complete paper-level evaluation because the assets are absent
locally and because the route remains approximate due to BVH-to-HumanML3D
skeleton adaptation and 196-frame evaluator truncation.

## 2026-06-14 artifact run: segment-aligned native-retarget Stage1 route

### Purpose

The previous best native-retarget run improved generation length and early-stop
rate, but it trained every window with the full long caption while inference used
segmented local prompts split by `" then "`.  That mismatch made the result hard
to interpret: the finetuned GPT was not trained under the same text-conditioning
semantics used at rollout time.

This run fixes that consistency gap and tests the current main Stage1 route:

```text
HumanML3D long sequence synthesis
  -> export long HumanML3D motions to BVH
  -> MoConVQ MotionDataSet.add_bvh_with_character()
  -> MoConVQ simulator character observation
  -> agent.encode_seq_all()
  -> segment-aligned GPT cache
  -> conservative head-only Text2Motion_Transformer fine-tune
  -> segmented long-text inference with the same " then " convention
```

This is still a HumanML3D-based route.  We are not abandoning HumanML3D; we are
avoiding the old hand-written HumanML3D-to-MoConVQ body-state mapping because it
caused token collapse and poor observation distribution.  No external LLM has
been called for the results below; the LLM token-planning code remains a backup
path only.

### Code changes

- `Script/stage1/build_bvh_character_gpt_cache.py`
  - added segment-aligned cache metadata for local-caption training:
    `segment_idxs`, `num_segments`, `segment_progress`, `prefix_lengths`,
    `target_ranges`, `segment_ranges`, `target_masks`, and `end_masks`.
  - supported segment-prefix sampling so training windows match segmented
    inference more closely.
- `Script/stage1/summarize_bvh_retarget_quality.py`
  - updated for the same segment-aware cache summaries.
- `Script/stage1/run_stage1_model_suite.py`
  - added `--segment-joiner` and recorded it in `suite_summary.json`.
- `Script/stage1/run_text_gpt_comparison.py`
  - added the same `--segment-joiner` forwarding and summary field so direct
    comparison runs cannot silently diverge from suite runs.
- Tests were extended for segment metadata and segment-joiner propagation.

### Segment-aligned cache

Artifacts:

```text
/tmp/stage1_segment_aligned_bvh_native_200_20260614/train_cache.pt
/tmp/stage1_segment_aligned_bvh_native_200_20260614/val_cache.pt
```

Cache summary:

| Split | Windows | Valid RVQ tokens | Unique long sequences |
|---|---:|---:|---:|
| train | 476 | 85,328 | 73 |
| val | 117 | 20,756 | 18 |

Token distribution is not collapsed:

| Split | Depth0 top frac | Depth1 top frac | Depth2 top frac | Depth3 top frac |
|---|---:|---:|---:|---:|
| train | 0.0566 | 0.0247 | 0.0479 | 0.0700 |
| val | 0.0450 | 0.0334 | 0.0495 | 0.0934 |

Interpretation:

- The cache is much smaller than the old hand-written HumanML3D observation
  cache, but its token distribution is far healthier.
- Training and inference now share the same local segment semantics: captions
  are split with the literal joiner `" then "`.
- Current splitter is intentionally simple and reproducible.  It handles the
  synthetic caption convention but not arbitrary `Then`, Chinese `然后`, or
  punctuation-only clauses.

### Head-only segment-aligned training

Command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/train_real_text_gpt.py \
  --train-cache /tmp/stage1_segment_aligned_bvh_native_200_20260614/train_cache.pt \
  --val-cache /tmp/stage1_segment_aligned_bvh_native_200_20260614/val_cache.pt \
  --init-checkpoint /home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --output-dir /tmp/stage1_segment_aligned_bvh_native_200_head_seed13_5ep_20260614 \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --gpu 0 \
  --seed 13 \
  --save-every 1 \
  --num-workers 2 \
  --train-scope head \
  --progress-conditioning auto \
  --progress-scale 0.5 \
  --teacher-progress-conditioning none \
  --context-size 51
```

Checkpoint:

```text
/tmp/stage1_segment_aligned_bvh_native_200_head_seed13_5ep_20260614/checkpoint_epoch_5.pth
```

Training curve:

| Epoch | Train loss | Val loss | Train acc | Val acc |
|---:|---:|---:|---:|---:|
| 0 | 14.6498 | 16.8632 | 0.0576 | 0.0701 |
| 1 | 14.3336 | 16.5705 | 0.0567 | 0.0702 |
| 2 | 13.9194 | 16.2719 | 0.0587 | 0.0708 |
| 3 | 13.5453 | 15.9782 | 0.0588 | 0.0711 |
| 4 | 13.1992 | 15.6747 | 0.0598 | 0.0713 |

The loss is high because the segment-aligned cache is harder and smaller, but it
decreases monotonically.  This checkpoint is the current recommended
training/inference-consistent Stage1 model.

### Four hand-written long-prompt suite

Suite:

```text
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch5_compare_20260614
```

Generation settings:

```text
generation_mode = auto
segment_joiner = " then "
max_length = 75
context_size = 30
chunk_size = 20
top_p = 0.95
temperature = 1.0
progress_conditioning = auto
baseline_progress_conditioning = none
progress_scale = 0.5
```

Engineering metrics:

| Metric | Baseline | Finetuned |
|---|---:|---:|
| avg frames | 1062 | 1194 |
| early-stop rate | 0.75 | 0.50 |
| avg root path | 3.4716 | 3.4891 |
| avg pose velocity | 14.0521 | 14.2359 |
| avg pose variance | 141.1937 | 250.3179 |
| lag20 repeat fraction | 0.0000 | 0.0039 |

Approximate T2M evaluator metrics:

| Metric | Baseline | Finetuned |
|---|---:|---:|
| FID lower is better | 28.2732 | 25.9026 |
| R-precision@1 higher is better | 0.50 | 0.25 |
| R-precision@2 higher is better | 0.50 | 0.75 |
| R-precision@3 higher is better | 1.00 | 1.00 |
| matching score lower is better | 5.0807 | 5.8823 |

Video/contact-sheet artifacts:

```text
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch5_compare_20260614/video/
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch5_compare_20260614/contact_sheet.png
```

Interpretation:

- Finetuned improves average length, early-stop rate, and approximate FID.
- R-precision is mixed: top2 improves, top3 ties, top1 worsens.
- Visual contact sheet does not show obvious blank frames, full-body inversion,
  or catastrophic fall-over, but it is not enough to claim precise semantic
  success.

### Failed branch: base_head + KL on the segment-aligned cache

Purpose: test whether updating more GPT layers improves semantics.

Result:

| Metric | Baseline | Finetuned base_head+KL |
|---|---:|---:|
| avg frames | 1062 | 1356 |
| early-stop rate | 0.75 | 0.25 |
| avg pose velocity | 14.0521 | 26.1391 |
| FID | 28.2732 | 28.3293 |
| R-precision@1 | 0.50 | 0.25 |
| R-precision@2 | 0.50 | 0.50 |
| R-precision@3 | 1.00 | 0.50 |
| matching score | 5.0807 | 5.5170 |

Interpretation:

- Lower validation loss did not translate into better long-rollout or paper
  metrics.
- Updating base layers is currently less reliable than head-only conservative
  fine-tuning.

### Failed branch: head-only top_p=0.90, temperature=0.8

Result:

| Metric | Baseline | Finetuned |
|---|---:|---:|
| avg frames | 1158 | 1020 |
| early-stop rate | 0.50 | 1.00 |
| FID | 27.8603 | 26.8815 |
| R-precision@1 | 0.50 | 0.25 |
| R-precision@2 | 0.50 | 0.50 |
| R-precision@3 | 1.00 | 1.00 |

Interpretation: the sampling change helps FID slightly but hurts duration and
does not fix R-precision@1, so it is not the recommended setting.

### Held-out Val8 long-caption suite

Because four hand-written prompts are very small and noisy, the same checkpoint
was also evaluated on eight held-out long captions from the segment-aligned val
split.

Suite:

```text
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch5_val8_compare_20260614
```

Paper-metric summary:

```text
/tmp/stage1_t2m_paper_metrics_segment_aligned_head_val8_20260614/summary.json
```

Engineering metrics:

| Metric | Baseline | Finetuned |
|---|---:|---:|
| avg frames | 1296 | 1329 |
| early-stop rate | 0.375 | 0.125 |
| avg root path | 2.2293 | 2.5027 |
| avg pose velocity | 16.0732 | 18.7995 |
| avg pose variance | 158.5108 | 178.3341 |
| lag20 repeat fraction | 0.0064 | 0.0048 |

Approximate T2M evaluator metrics:

| Metric | Baseline | Finetuned |
|---|---:|---:|
| FID lower is better | 18.1357 | 16.8122 |
| R-precision@1 higher is better | 0.25 | 0.375 |
| R-precision@2 higher is better | 0.625 | 0.50 |
| R-precision@3 higher is better | 1.00 | 0.50 |
| matching score lower is better | 4.3217 | 4.2714 |

Video/contact-sheet artifacts:

```text
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch5_val8_compare_20260614/video/
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch5_val8_compare_20260614/contact_sheet.png
```

Visual audit:

- The contact sheet shows no blank frames, obvious full-body inversion, or
  global pose explosion.
- `train_000077` is the clearest visual improvement: the baseline ends near the
  ground, while finetuned reaches a crouch/kneel state and returns to standing.
- Some samples mainly improve length/motion coverage rather than detailed
  semantic correctness.

Interpretation:

- This is currently the strongest Stage1 result: finetuned improves FID,
  R-precision@1, matching score, average length, early-stop rate, and repetition
  proxy on held-out long captions.
- It is not a clean win on every paper metric cutoff because R-precision@2 and
  R-precision@3 drop.
- The evaluator route is approximate: generated MoConVQ BVHs are converted
  through a `base.bvh` to HumanML3D 22-joint adapter, and the T2M evaluator
  truncates long sequences to at most 196 frames at 20 FPS.

### Paper evaluator readiness

Assets and source files are now present under:

```text
/tmp/stage1_t2m_evaluator_assets
```

Readiness check:

```text
paper_metrics_ready = true
missing = []
```

Required files found:

```text
checkpoints/t2m/text_mot_match/model/finest.tar
checkpoints/t2m/text_mot_match/opt.txt
glove/our_vab_data.npy
glove/our_vab_words.pkl
glove/our_vab_idx.pkl
```

### Validation

Commands:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/run_text_gpt_comparison.py \
  Script/stage1/run_stage1_model_suite.py \
  Script/stage1/generate_long_motion.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_text_gpt_comparison \
  tests.test_stage1_model_suite \
  tests.test_stage1_real_generate \
  -v
```

Result:

```text
20 tests passed
```

### Current honest conclusion

The current best Stage1 claim is:

```text
Segment-aligned HumanML3D -> BVH -> MoConVQ-native retarget + head-only GPT
fine-tuning gives a reproducible partial improvement over the MoConVQ baseline
on long multi-stage prompts.  On the held-out Val8 suite it improves approximate
FID, R-precision@1, matching score, early-stop rate, average length, and the
lag20 repetition proxy.  It does not yet improve every R-precision cutoff, so
the final report should present it as a meaningful but incomplete Stage1 result,
not as a full paper-metric victory.
```
