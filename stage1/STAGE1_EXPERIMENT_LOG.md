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
