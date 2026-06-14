# Stage1 Final Result Summary

Date: 2026-06-14

This file is the report-facing summary for Stage1.  It includes the final
result plus the key intermediate experiments that determined the final route.
Full diagnostic history is kept in `STAGE1_EXPERIMENT_LOG.md`; implementation
and reproduction notes are kept in `STAGE1_README.md`.  A fuller
presentation/report draft with methods, intermediate experiments, final metrics
and video paths is available in `STAGE1_METHOD_RESULTS_FOR_PRESENTATION.md`.

## Goal Status

Stage1 now has a reproducible end-to-end pipeline:

```text
HumanML3D long sequence synthesis
  -> HumanML3D long motion export to BVH
  -> MoConVQ native MotionDataSet.add_bvh_with_character() retarget
  -> simulator character observation
  -> MoConVQ encoder encode_seq_all()
  -> RVQ token / latent cache with segment-prefix metadata
  -> conservative fine-tune of text-conditioned MoConGPT
  -> explicit segment + segment-length long-text inference
  -> BVH generation
  -> engineering metrics, contact sheets / videos
  -> approximate T2M evaluator FID, R-precision and matching score
```

The route uses HumanML3D as the main data source.  HumanML3D was not abandoned.
The old hand-written HumanML3D-to-MoConVQ body-state/cache path was replaced for
the final claim because it caused token collapse and observation-distribution
mismatch.  No external LLM or in-context LLM response is used in the selected
results; `llm_token_planning.py` remains a backup route only.

## What Works

The working data and model route is:

```text
HumanML3D synthesized long sequences
  -> BVH
  -> MoConVQ-native character retarget
  -> accepted-only GPT cache
  -> base_head fine-tune
```

The selected training cache is the segment-aligned BVH-native cache:

| Item | Value |
| --- | --- |
| Train long sequences | 73 |
| Val long sequences | 18 |
| Train windows | 476 |
| Val windows | 117 |
| Train valid RVQ tokens | 85,328 |
| Val valid RVQ tokens | 20,756 |
| Token top fraction | depth0 0.0566, depth1 0.0247, depth2 0.0479, depth3 0.0700 |

Selected fine-tune run:

```text
/tmp/stage1_segment_aligned_bvh_native_200_basehead_seed13_3ep_20260614
```

Training configuration:

```text
train_scope = base_head
trainable_parameters = 30,577,152
learning_rate = 5e-6
epochs = 3
progress_conditioning = auto
progress_scale = 0.5
context_size = 51
```

Training curve:

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13.8335 | 14.9117 | 0.0579 | 0.0707 |
| 1 | 11.7993 | 12.5843 | 0.0573 | 0.0715 |
| 2 | 9.8848 | 10.7311 | 0.0598 | 0.0719 |

## Training And Inference Consistency

Formal evaluation does not rely on naive `" then "` splitting.  The strict
prompt protocol exports the validation prompts directly from cache metadata:

```text
name<TAB>long_text<TAB>segments_json<TAB>scaled_lengths_json
```

Inference forwards:

```text
--segments-json
--segment-lengths
```

This preserves the original HumanML3D clip-caption boundaries.  It avoids the
failure mode where a raw caption containing sentence-internal `then` is split
into a segment that never existed during training.

Prompt export command:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/export_cache_prompt_tsv.py \
  --cache /tmp/stage1_segment_aligned_bvh_native_200_20260614/val_cache.pt \
  --output /tmp/stage1_segment_aligned_val18_explicit_segments_scaled75_prompts.tsv \
  --summary /tmp/stage1_segment_aligned_val18_explicit_segments_scaled75_prompts_summary.json \
  --total-length 75
```

Result:

```text
num_prompts = 18
```

## Intermediate Experiments And Decisions

The central Stage1 finding was that long-sequence text synthesis alone was not
enough.  The dominant failure mode was how HumanML3D motion was mapped into the
MoConVQ simulator character state, observation space and RVQ token space.  The
experiments below explain why the final route uses HumanML3D-derived BVH plus
MoConVQ-native retargeting instead of the original hand-written cache.

| Experiment / Route | Purpose | Key Result | Decision |
| --- | --- | --- | --- |
| Fixed HumanML3D long dataset with hand-written retarget | First end-to-end long-text cache and GPT fine-tune | 1000 train / 200 val sequences; cache had 2958 train windows and 598 val windows. Token training looked strong: train loss 1.6198, val loss 1.7807, train acc 0.5569, val acc 0.5236. | Rejected for final claim because video quality and token diagnostics did not match the good loss. |
| Training-target repair on old cache | Align GPT training target with inference latent usage | Changed target to previous reconstructed 4-layer RVQ latent -> current RVQ indices; added train scopes, KL/teacher options, end-token auxiliary loss and progress conditioning. | Kept the code fixes, but old data still had unhealthy token distribution. |
| Old-cache token and observation diagnostics | Check whether the old cache represented MoConVQ body state correctly | Depth0 top token fraction reached 0.2171 and depth1 reached 0.3342, far above native MoConVQ reference behavior. | Identified HumanML3D-to-MoConVQ body-state mapping as the main bottleneck. |
| Rest-pose / rotation calibration | Reduce obvious skeleton-frame mismatch in the hand-written mapping | Improved some local-rotation outliers, but angular velocity and retarget distribution mismatch remained. | Not sufficient as a final data route. |
| Caption granularity / atomic-caption diagnosis | Test whether HumanML3D captions were too composite for segment training | Prefer-atomic captions reduced non-atomic captions to about 8 percent, but token collapse remained: atomic cache depth0 top fractions still included 22.69 percent and 11.79 percent tokens. | Caption cleaning alone cannot fix the mapping problem. |
| Segment-prefix / progress conditioning | Make training resemble segmented long-text inference | Added prefix motion context plus local segment caption, segment progress features and segmented generation. | Kept as part of the final method, but it needed a healthier motion cache. |
| MoConVQ native BVH-to-character smoke | Verify that original MoConVQ retarget path can produce GPT cache | `base.bvh`: 1 window, 16 valid tokens, index range 27..489. `track.bvh`: 4 windows, 800 valid tokens, index range 3..511. Track smoke training: train loss 7.6435, val loss 7.8137, train acc 0.0050, val acc 0.0175. | Native BVH retarget path works; local BVH files were too few for final fine-tuning. |
| Batch500 processed HumanML3D -> BVH | Test HumanML3D -> BVH -> native retarget at a larger scale | Accepted 72 train / 18 val sequences from filtered Batch500. Token distribution no longer showed the extreme old-cache collapse. Base-head 5 epoch training reached about train loss 4.8432, val loss 4.8578, train acc 0.1516, val acc 0.1492. | Validated the replacement data route, but data scale and export quality were still limited. |
| Long-H5 BVH-native 200 sequence route | Build a more coherent long-sequence BVH-native cache | 20-smoke accepted top fractions were depth0 0.0583, depth1 0.0354, depth2 0.0500, depth3 0.0800. Full run accepted 481 train windows / 120 val windows, 87,472 / 21,592 valid tokens, 73 / 18 unique train/val sequences. | Became the basis for the final segment-aligned cache. |
| Paper-metric readiness audit | Check whether MoConVQ paper Text2Motion metrics can be computed | Original HumanML3D evaluator assets were not fully native to this repo. Implemented an approximate BVH-to-HumanML3D 22-joint / 263-d adapter and evaluator route. | Use FID, R-precision and matching score as approximate evaluator-adapter metrics, with caveats. |
| Segment-aligned final cache | Align training examples with long-prompt segment boundaries | Final selected cache: 476 train windows, 117 val windows, 85,328 / 20,756 valid tokens, 73 / 18 train/val long sequences. Token top fractions: depth0 0.0566, depth1 0.0247, depth2 0.0479, depth3 0.0700. | Selected as final training data. |
| Explicit segment protocol | Remove inference-time ambiguity from raw `" then "` splitting | Exported `segments_json` and `segment_lengths` directly from cache metadata. This avoids splitting sentence-internal words such as "then" inside HumanML3D captions. | Required for fair training/inference consistency. |
| Head-only segment-aligned fine-tune | Check whether only adapting the GPT head is enough | Under strict explicit segment and scaled-length protocol, head-only improved some R-precision/matching signals but did not reliably improve FID and had worse early-stop behavior. | Head-only was under-capacity for the final strict protocol. |
| Base-head micro fine-tune | Allow limited adaptation of base plus head on the final cache | Selected run used `base_head`, lr 5e-6 and 3 epochs. It reduced val loss from 14.9117 to 10.7311 and gave the strongest strict Val8 result. | Selected as final fine-tuned model family. |
| Decoding sweep | Test whether lower entropy sampling improves the final model | `top_p=0.90`, `temperature=0.8` was negative: baseline FID/R@1/R@2/R@3/matching = 13.2935 / 0.3333 / 0.4444 / 0.5556 / 4.8050; fine-tuned = 14.3544 / 0.2778 / 0.3889 / 0.5000 / 4.8366. | Keep `top_p=0.95`, `temperature=1.0` for final comparison. |
| HumanML3D-test Long100 evaluation | Move beyond small Val8/Val18 checks | On 100 held-out three-segment prompts, fine-tuning improved R@2, matching score, average length, early-stop rate and root path; it tied R@1/R@3 and slightly worsened FID. | Final claim must be partial improvement, not full paper-metric dominance. |

Report-facing interpretation:

```text
Early token-level improvements were rejected after video, token-distribution
and observation-space diagnostics.  The main technical result is the working
HumanML3D -> BVH -> MoConVQ-native character retarget -> segment-aligned GPT
cache pipeline.  The final model improves some long-prompt metrics and videos,
but does not uniformly beat baseline on all approximate paper-style metrics.
```

## Selected Results

### Larger HumanML3D-Test Long100 Check

After the initial Val8/Val18 checks, a larger independent prompt suite was
constructed from the HumanML3D `test.txt` split.  It samples 300 held-out
HumanML3D captions and combines them into 100 three-segment long prompts:

```text
test caption A then test caption B then test caption C
segments_json = [A, B, C]
segment_lengths = [25, 25, 25]
```

This check is more convincing than the Val8/Val18 sanity sets because it uses
100 generated prompts rather than 8 or 18.  The result is mixed, not a full
win:

| Metric | Baseline | Fine-tuned epoch3 |
| --- | ---: | ---: |
| prompts | 100 | 100 |
| generated BVHs | 100 | 100 |
| avg frames | 1216.56 | 1230.96 |
| early-stop rate | 0.44 | 0.41 |
| root path | 3.3679 | 3.4323 |
| pose velocity mean | 38.0968 | 37.5597 |
| approximate FID lower is better | 7.0400 | 7.2092 |
| approximate R@1 higher is better | 0.050 | 0.050 |
| approximate R@2 higher is better | 0.110 | 0.140 |
| approximate R@3 higher is better | 0.160 | 0.160 |
| approximate matching score lower is better | 5.0077 | 4.9591 |

Interpretation:

```text
Fine-tuning slightly improves average duration, early-stop rate, root-path
coverage, R@2 and matching score on N=100.  It ties R@1/R@3 and slightly
worsens FID.  Therefore the robust conclusion is a partial semantic/rollout
improvement, not paper-level dominance over the original baseline.
```

Artifacts:

```text
/tmp/stage1_humanml_test_long100_basehead_epoch3_batch_20260614
/tmp/stage1_t2m_paper_metrics_humanml_test_long100_basehead_epoch3_20260614/summary.json
stage1_artifacts/review_videos_20260614/long100_metrics/
```

### Primary Val8 Result

This is the clearest strict-protocol positive result and the best video
showcase, but it should now be treated as a small-sample positive example:

```text
checkpoint = /tmp/stage1_segment_aligned_bvh_native_200_basehead_seed13_3ep_20260614/checkpoint_epoch_3.pth
decoding = top_p=0.95, temperature=1.0
prompt protocol = explicit segment JSON + scaled segment lengths, total length 75
```

| Metric | Baseline | Fine-tuned |
| --- | ---: | ---: |
| avg frames | 1182 | 1197 |
| early-stop rate | 0.50 | 0.50 |
| root path | 1.6818 | 2.0738 |
| pose velocity / variance | 16.104 / 181.560 | 17.732 / 193.894 |
| lag20 repeat fraction | 0.0020 | 0.0028 |
| approximate FID lower is better | 20.2790 | 14.9851 |
| approximate R@1 higher is better | 0.375 | 0.375 |
| approximate R@2 higher is better | 0.500 | 0.750 |
| approximate R@3 higher is better | 0.625 | 0.875 |
| approximate matching score lower is better | 4.8132 | 4.3839 |

Interpretation:

```text
Fine-tuned improves approximate FID, R@2, R@3, matching score and root path,
while tying R@1 and early-stop rate.  Pose energy and lag20 repetition are
slightly higher, so visual inspection is still required.
```

Artifacts:

```text
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614
/tmp/stage1_t2m_paper_metrics_segment_aligned_basehead_epoch3_val8_explicit_scaled75_20260614/summary.json
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614/contact_sheet.png
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614/video/train_000057__baseline_vs_basehead.mp4
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614/video/train_000077__baseline_vs_basehead.mp4
```

### Conservative Val18 Result

For the full 18-sequence validation prompt set, `checkpoint_epoch_2` is the
safer metric checkpoint:

```text
checkpoint = /tmp/stage1_segment_aligned_bvh_native_200_basehead_seed13_3ep_20260614/checkpoint_epoch_2.pth
decoding = top_p=0.95, temperature=1.0
prompt protocol = explicit segment JSON + scaled segment lengths, total length 75
```

| Metric | Baseline | Fine-tuned epoch2 |
| --- | ---: | ---: |
| avg frames | 1292 | 1304 |
| early-stop rate | 0.2778 | 0.3333 |
| root path | 2.6053 | 2.7284 |
| root displacement | 0.8678 | 0.9158 |
| pose velocity mean | 27.3341 | 27.6977 |
| pose variance mean | 339.6971 | 328.4767 |
| lag20 repeat fraction | 0.0075 | 0.0063 |
| approximate FID lower is better | 13.7255 | 13.0602 |
| approximate R@1 higher is better | 0.2222 | 0.2222 |
| approximate R@2 higher is better | 0.4444 | 0.4444 |
| approximate R@3 higher is better | 0.4444 | 0.5000 |
| approximate matching score lower is better | 4.8802 | 4.7885 |

Interpretation:

```text
Fine-tuned epoch2 improves approximate FID, R@3, matching score, average length,
root path, root displacement, pose variance and lag20 repetition.  R@1 and R@2
tie baseline.  Early-stop rate is worse than baseline, so this is not a complete
win, but it is the most conservative full-Val18 metric selection.
```

Artifacts:

```text
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch2_val18_explicit_scaled75_compare_20260614
/tmp/stage1_t2m_paper_metrics_segment_aligned_basehead_epoch2_val18_explicit_scaled75_20260614/summary.json
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch2_val18_explicit_scaled75_compare_20260614/contact_sheet.png
```

### Val18 Epoch3 Trade-off

`checkpoint_epoch_3` is better on FID, R@1 and matching score, but regresses R@2:

| Metric | Baseline | Fine-tuned epoch3 |
| --- | ---: | ---: |
| approximate FID lower is better | 13.7255 | 12.6332 |
| approximate R@1 higher is better | 0.2222 | 0.2778 |
| approximate R@2 higher is better | 0.4444 | 0.2778 |
| approximate R@3 higher is better | 0.4444 | 0.4444 |
| approximate matching score lower is better | 4.8802 | 4.6093 |

This should be described as a stronger FID/R@1/matching checkpoint, not as a
uniform R-precision improvement on full Val18.

## Negative Results

The conservative decoding probe should not be used for the final claim:

```text
checkpoint = checkpoint_epoch_2.pth
decoding = top_p=0.90, temperature=0.8
```

| Metric | Baseline | Fine-tuned |
| --- | ---: | ---: |
| approximate FID lower is better | 13.2935 | 14.3544 |
| approximate R@1 higher is better | 0.3333 | 0.2778 |
| approximate R@2 higher is better | 0.4444 | 0.3889 |
| approximate R@3 higher is better | 0.5556 | 0.5000 |
| approximate matching score lower is better | 4.8050 | 4.8366 |

This makes the baseline stronger and the fine-tuned model worse on all
approximate T2M metrics.

## Visual Conclusion

Manual contact-sheet/video inspection supports a cautious positive result:

```text
No blank-frame, whole-body inversion or explosive-pose failure is visible in the
selected Val8/Val18 sheets.  Fine-tuned outputs usually maintain longer motion
and slightly larger root/path coverage.  Some low-posture prompts preserve
crouch/crawl families without immediate collapse.
```

Remaining visual issues:

```text
Some poses are still awkward, semantic details are inconsistent, and the model
does not yet look like a polished long-horizon text-to-motion generator.
```

Therefore the report wording should be:

```text
The Stage1 route gives a partial but meaningful improvement over baseline on
long multi-stage prompts, with better approximate FID / selected R-precision
cutoffs and slightly better video stability.  It is not a solved long-horizon
motion generation system.
```

## Metric Caveat

MoConVQ paper metrics for Text2Motion are FID and R-precision on HumanML3D /
SMPL-style features.  This project now has a working evaluator route, but it is
still approximate:

```text
MoConVQ BVH
  -> approximate MoConVQ/base.bvh to HumanML3D 22-joint adapter
  -> HumanML3D 263-d feature extraction
  -> T2M evaluator FID, R-precision and matching score
```

The adapter has nonzero skeleton error, and the T2M evaluator truncates long
sequences to `max_motion_length=196` frames at 20 FPS.  In the final report,
these numbers should be labeled:

```text
approximate T2M evaluator-adapter metrics
```

They should not be presented as native MoConVQ paper / SMPL evaluation.

## Report-Ready Claim

Recommended final claim:

```text
Stage1 successfully constructs a reproducible HumanML3D-to-MoConVQ long-sequence
fine-tuning pipeline.  Replacing the old hand-written HumanML3D-to-state cache
with MoConVQ-native BVH-to-character retargeting fixes the main data-mapping
failure.  With explicit segment-boundary inference and a conservative base_head
fine-tune, the fine-tuned MoConGPT improves over the original baseline on
approximate paper-style FID and selected R-precision cutoffs, and shows slightly
better long-motion stability in contact sheets/videos.  The improvement is
partial: full Val18 R-precision is mixed, early stopping is not fully solved,
and visual semantics remain imperfect.
```

## Verification

Commands run from both the true workdir and the pushed `main/stage1` worktree:

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python -m py_compile \
  Script/stage1/export_cache_prompt_tsv.py \
  Script/stage1/run_stage1_model_suite.py \
  Script/stage1/generate_long_motion.py \
  Script/stage1/run_text_gpt_comparison.py \
  Script/stage1/evaluate_t2m_paper_metrics.py

/home/chenjie/miniconda3/envs/moconvq/bin/python -m unittest \
  tests.test_stage1_cache_prompt_export \
  tests.test_stage1_model_suite \
  tests.test_stage1_real_generate \
  tests.test_stage1_text_gpt_comparison \
  tests.test_stage1_evaluation_readiness \
  tests.test_stage1_t2m_paper_metrics \
  -v
```

Result:

```text
36 tests passed
```

The pushed `main` worktree was checked for forbidden private or large files
before commit:

```text
AGENT.md, AGENTS.md, CODEX.md, CLAUDE.md, .codex/, .claude/
stage1_artifacts/, *.h5, *.pth, *.data, *.npy, *.zip, *.tar, *.pkl, *.pt,
*.mp4, *.png, midterm-report/, midterm_figures/, request.txt
```
