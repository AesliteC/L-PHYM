# Stage1 Final Result Summary

Date: 2026-06-14

This file is the report-facing summary for Stage1.  Full diagnostic history is
kept in `STAGE1_EXPERIMENT_LOG.md`; implementation and reproduction notes are
kept in `STAGE1_README.md`.

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

## Selected Results

### Primary Val8 Result

This is the clearest strict-protocol positive result and the best video
showcase:

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

