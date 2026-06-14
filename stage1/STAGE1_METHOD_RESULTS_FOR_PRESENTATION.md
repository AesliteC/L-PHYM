# Stage1 方法论、过程实验与最终结果汇总

更新时间：2026-06-14

用途：给检查、oral presentation 和最终实验报告使用。本文档把 Stage1 从旧路线失败、
诊断、修复、备选路线、最终主结果到视频/指标证据统一整理。完整逐条日志仍见
`STAGE1_EXPERIMENT_LOG.md`；可复现命令和代码说明见 `STAGE1_README.md`；
更短的最终结论摘要见 `STAGE1_FINAL_RESULT_SUMMARY.md`。

## 1. 视频和可视化结果在哪里

视频和 contact sheet 没有 push 到 GitHub，因为它们属于生成 artifact，按仓库卫生
规则不提交 `.mp4` / `.png`。当前主要保存在 `/tmp`。

### 1.1 最终最推荐展示的视频

这两个是当前最适合 oral presentation 展示的 baseline vs fine-tuned side-by-side
视频。它们来自最终主结果之一：

```text
route      = HumanML3D long sequence -> BVH -> MoConVQ native character retarget
model      = base_head checkpoint_epoch_3
protocol   = explicit HumanML3D clip segments + scaled segment lengths
decoding   = top_p=0.95, temperature=1.0
suite      = Held-out Val8 strict protocol
```

视频路径：

```text
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614/video/train_000057__baseline_vs_basehead.mp4
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614/video/train_000077__baseline_vs_basehead.mp4
```

对应 contact sheet：

```text
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614/contact_sheet.png
```

视频结论：没有空帧、整体倒置、骨架爆炸。fine-tuned 通常更能持续运动，root/path
coverage 更大。`train_000077` 是较好的展示样例，低姿态/蹲跪动作没有马上崩掉。
但仍有动作语义细节不稳定和姿态不自然的问题。

### 1.2 最终 Val18 静态可视化

完整 18 条 validation sequence 的最终评估主要保存了 contact sheet：

```text
epoch2, conservative Val18 metric checkpoint:
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch2_val18_explicit_scaled75_compare_20260614/contact_sheet.png

epoch3, stronger FID/R@1/matching checkpoint:
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val18_explicit_scaled75_compare_20260614/contact_sheet.png
```

Val18 contact sheet 结论：没有空帧、整体倒置或爆炸。部分 crouch/crawl/low-pose
样例能维持低姿态。仍存在姿态弯折和语义细节失败，所以最终表述应是
“partial but meaningful improvement”，不是“完全解决 long-horizon generation”。

### 1.3 历史对比视频目录

这些视频适合展示方法演进，不建议作为最终主结果单独引用。

早期 Batch500 / processed HumanML3D joints-IK 路线：

```text
/tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613/video/
/tmp/stage1_batch500_v2_basehead_epoch5_top_p_len75_20260613/contact_sheet.png
```

Long-H5 BVH-native 200 sequence 工程改进路线：

```text
/tmp/stage1_long_fixed_bvh_native_200_basehead_epoch5_compare_20260613/video/
/tmp/stage1_long_fixed_bvh_native_200_basehead_epoch5_compare_20260613/contact_sheet.png

/tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/video/
/tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/contact_sheet.png
```

Segment-aligned head-only Val8 路线：

```text
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch3_val8_compare_20260614/video/
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch3_val8_compare_20260614/contact_sheet.png

/tmp/stage1_segment_aligned_bvh_native_200_head_epoch5_val8_compare_20260614/video/
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch5_val8_compare_20260614/contact_sheet.png
```

Segment-aligned hand-written prompt suite：

```text
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch5_compare_20260614/video/
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch5_compare_20260614/contact_sheet.png
```

## 2. 最终 Stage1 问题定义

Stage1 的目标不是训练一个普通文本 GPT，而是在 MoConVQ 预训练系统上完成：

```text
long multi-stage text
  -> text-conditioned MoConGPT
  -> RVQ motion token sequence
  -> MoConVQ decoder/controller
  -> BVH / character motion
```

实际要解决三个核心问题：

1. 数据构造：如何把 HumanML3D 短动作片段合成长动作-文本序列。
2. 表示映射：如何把 HumanML3D motion 映射到 MoConVQ simulator character
   observation / RVQ token，而不是产生 token collapse。
3. 模型训练与推理一致性：训练时看到的文本片段、motion prefix、latent space、
   推理时使用的分段方式必须一致，否则 token loss 降低也不代表视频变好。

## 3. 最终采用的主路线

最终主路线是：

```text
HumanML3D long sequence synthesis
  -> export long HumanML3D motion to BVH
  -> MoConVQ native MotionDataSet.add_bvh_with_character()
  -> simulator character state / observation
  -> MoConVQ agent.encode_seq_all()
  -> RVQ indices + latent_vq + T5 text feature cache
  -> segment-prefix / segment-aligned GPT fine-tuning
  -> explicit segment + segment-length inference
  -> generated BVH
  -> engineering metrics + video/contact sheet
  -> approximate T2M evaluator FID/R-precision
```

关键判断：HumanML3D 没有被放弃；被放弃的是旧的 hand-written
HumanML3D-to-MoConVQ body-state/cache 映射。最终没有使用外部 LLM 或我自己作为
LLM 去生成 motion token；`llm_token_planning.py` 只是备选工程路线。

## 4. 为什么旧路线不能作为最终结果

### 4.1 旧 fixed dataset / hand-written retarget 结果

早期构造过 fixed HumanML3D 长序列数据：

| split | sequences | avg clips | avg frames | transitions | forced transitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1000 | 2.945 | 416.593 | 1945 | 0 |
| val | 200 | 2.990 | 410.200 | 398 | 0 |

固定 cache：

| split | windows | unique sequences | valid RVQ target tokens |
| --- | ---: | ---: | ---: |
| train | 2958 | 1000 | 417,512 |
| val | 598 | 200 | 81,716 |

旧 20 epoch 训练能让 token-level val loss / acc 看起来不错，例如：

```text
epoch 20 / best val:
  train loss = 1.6198
  val loss   = 1.7807
  train acc  = 0.5569
  val acc    = 0.5236
```

但后来判定旧 checkpoint 不能作为有效模型，原因有两个：

1. 训练/推理 latent 不一致：训练时 cache 里的 `latent_vq` 来自 MoConVQ encoder 的
   8-layer RVQ 总和，而推理时 Text2Motion GPT 实际采样前 4 层 RVQ token 并用前
   4 层 codebook embedding 求和作为下一步 latent context。
2. 旧 hand-written HumanML3D-to-state 映射导致 observation distribution mismatch
   和 RVQ token collapse，尤其 depth0/depth1 token 分布异常集中。

因此，早期“token loss 好看”和“生成更长”不能证明 Stage1 成功。

### 4.2 修复训练目标但仍暴露 retarget 问题

训练代码后来修复为：

```text
previous reconstructed 4-layer RVQ latent -> current RVQ indices
```

并加入：

```text
--train-scope {all, base_head, head}
baseline KL / teacher checkpoint
end-token auxiliary loss
progress conditioning
```

但当数据仍来自 hand-written HumanML3D-to-observation cache 时，token distribution
仍然不健康。典型症状：

```text
old fixed cache depth0 top fraction = 0.2171
old fixed cache depth1 top fraction = 0.3342
```

这说明主要瓶颈不是 epoch 不够，而是数据映射进 MoConVQ body state / observation
的路线不可靠。

## 5. 中途关键诊断和转折

### 5.1 Top-p 采样修复

早期 greedy / fixed-top-k 视频不能再作为当前结论。代码增加了：

```text
top_p
top_k
temperature
```

统一使用更公平的 sampling protocol：

```text
top_p=0.95
top_k=0
temperature=1.0
seed=123
```

这个改动让 baseline 和 finetuned 在同一采样策略下比较。

### 5.2 Segment-progress / segment-prefix 思路

问题：长 prompt 如：

```text
a person walks forward then turns around then waves both arms
```

如果训练窗口只看单 clip 或同一个完整 long caption，推理却分段生成，会产生训练/推理
语义不一致。

加入的思想：

```text
previous motion prefix + current local segment caption -> next segment tokens
```

并把 progress feature 注入 MoConVQ GPT 原有 `clip_feature` 条件通路。

对应代码：

```text
Script/stage1/segment_conditioning.py
Script/stage1/real_moconvq_cache.py --sample-mode segment_prefix
Script/stage1/train_real_text_gpt.py --progress-conditioning
Script/stage1/generate_long_motion.py segmented generation
```

这个方向能让模型生成更长，但在旧 retarget 数据上仍有 pose velocity / variance
过高和语义不稳的问题。

### 5.3 Caption granularity 诊断

HumanML3D 原始 caption 经常已经包含多个动作。统计结果：

| split | first caption non-atomic | prefer_atomic non-atomic | atomic keep rate |
| --- | ---: | ---: | ---: |
| train | 35.20% | 8.09% | 91.91% |
| val | 34.38% | 7.53% | 92.47% |
| test | 35.77% | 8.39% | 91.61% |

做了 atomic-caption 小数据诊断后发现：caption 变干净并不能修复 depth0 token collapse。

Atomic cache depth0 top fractions：

```text
token 492: 22.69%
token 338: 11.79%
```

对比 native MoConVQ：

```text
top token around 6.8%
```

结论：caption granularity 是问题之一，但不是主因；主因仍是 HumanML3D-to-MoConVQ
表示映射。

### 5.4 MoConVQ 原生 BVH-to-character 路线验证

用 MoConVQ 原仓库路径：

```text
BVH
  -> MotionDataSet.add_bvh_with_character()
  -> simulator character observation
  -> encode_seq_all()
  -> RVQ tokens
```

在 `base.bvh` 和 `track.bvh` 上 smoke：

| source | windows | valid tokens | index range |
| --- | ---: | ---: | --- |
| base.bvh | 1 | 16 | 27..489 |
| track.bvh | 4 | 800 | 3..511 |

Observation z-score：

```text
mean abs z = 0.5626
p95 abs z = 1.8741
p99 abs z = 2.8260
frac abs z > 5 = 0.155%
```

对比旧 hand-written route，BVH-to-character 原生路径健康得多。

结论：需要把 HumanML3D 导出为 BVH，再走 MoConVQ native retarget，而不是继续直接
手写 HumanML3D joints 到 MoConVQ body state。

### 5.5 LLM in-context backup route

为了不被主线卡死，实现了 MoConVQ 论文启发的 backup 工程路径：

```text
GPT cache -> example bank -> retrieval / LLM prompt
-> validated RVQ token JSON -> decode BVH
```

新增：

```text
Script/stage1/llm_token_planning.py
tests/test_stage1_llm_token_planning.py
```

retrieval-only smoke 结果：

| item | value |
| --- | ---: |
| exported examples | 200 |
| retrieval-only RVQ tuples | 36 |
| decoded BVH frames | 864 |
| duration | 7.20 s |
| root path | 1.739 |
| pose velocity mean | 11.362 |
| lag20 repeat > 0.995 | 0.00% |

注意：这不是 LLM semantic-quality 结果。没有调用真实外部 LLM。最终主结果也不是这条路线。

## 6. Batch500 processed HumanML3D -> BVH 中间路线

先尝试把 processed HumanML3D `new_joints/new_joint_vecs` 导出成 MoConVQ-template
BVH，使用 `joints_ik` 生成 BVH rotation。

Batch500 结果：

```text
exports = 500
frames min/max/avg = 19 / 199 / 141.422
samples shorter than 120 frames = 205
```

MoConVQ-native retarget 后总体 observation：

```text
state_shape = [70711, 20, 13]
observation_shape = [70711, 323]
token_shape = [17677, 4]
p99_abs_z = 5.0459
max_abs_z = 68.3279
frac_gt_5 = 0.0102
```

质量过滤：

```text
total = 500
accepted = 90
accepted rate = 18%
```

主要 reject 原因：

| reason | count |
| --- | ---: |
| depth0_unique < 16 | 297 |
| depth0_top_frac > 0.25 | 252 |
| frames < 120 | 205 |
| p99_abs_z > 8 | 72 |
| max_abs_z > 50 | 47 |

Accepted cache token distribution 不再 collapse：

| depth | tokens | unique | entropy | top frac |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3951 | 403 | 7.9007 | 0.0385 |
| 1 | 3951 | 496 | 8.5605 | 0.0127 |
| 2 | 3951 | 498 | 8.5086 | 0.0423 |
| 3 | 3951 | 486 | 8.1048 | 0.0769 |

做过 MP4 audit 后手动 override：

```text
exclude 013481: floor/prone motion
include 010684: plausible walk/turn
include M012928: plausible walk/turn
v2 accepted = 91
```

用 batch500 filtered-v2 做 5 epoch `base_head` fine-tune：

| epoch | train loss | val loss | train acc | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 15.8651 | 17.6901 | 0.0566 | 0.0472 |
| 1 | 14.9218 | 17.1599 | 0.0575 | 0.0468 |
| 2 | 14.5209 | 16.5973 | 0.0591 | 0.0462 |
| 3 | 13.8787 | 16.0007 | 0.0580 | 0.0462 |
| 4 | 13.2425 | 15.3757 | 0.0579 | 0.0465 |

四个手写 prompt 对比：

| model | avg frames | early stop | root path | pose velocity | pose variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1062 | 0.75 | 3.472 | 14.052 | 141.194 |
| finetuned | 1092 | 0.50 | 3.073 | 24.086 | 297.008 |

结论：这条路线证明 “processed HumanML3D -> BVH -> native retarget” 可训练，但数据太小、
效果太弱，不能作为最终主结果。它是关键中间结果。

## 7. Long-H5 BVH-native 200 sequence 路线

下一步不是单 clip processed HumanML3D，而是直接把已有 HumanML3D long sequence H5
导出成 long BVH：

```text
long_sequences.h5 + manifest.jsonl
  -> export_long_humanml3d_to_bvh.py
  -> long BVH
  -> MoConVQ native character retarget
```

### 7.1 20 条 smoke

```text
exports = 20
accepted = 11
cache windows = 40
valid tokens = 8000
```

Accepted token top fraction：

| depth | top fraction | unique |
| --- | ---: | ---: |
| 0 | 0.0530 | 225 |
| 1 | 0.0315 | 342 |
| 2 | 0.0835 | 379 |
| 3 | 0.0755 | 350 |

对比旧 hand-written cache：

```text
old depth0 top fraction = 0.2171
old depth1 top fraction = 0.3342
```

这就是 Stage1 的关键转折：long-sequence synthesis 本身不是主问题，retarget/cache
路径才是主问题。

### 7.2 200 sequence run

```text
exports = 200
accepted = 91
rejected = 109
train sequences = 73
val sequences = 18
```

Cache：

| split | windows | valid RVQ tokens | unique sequences |
| --- | ---: | ---: | ---: |
| train | 278 | 55,516 | 73 |
| val | 66 | 13,200 | 18 |

Token distribution：

| split | depth0 top | depth1 top | depth2 top | depth3 top |
| --- | ---: | ---: | ---: | ---: |
| train | 0.0626 | 0.0219 | 0.0481 | 0.0709 |
| val | 0.0473 | 0.0382 | 0.0424 | 0.0803 |

### 7.3 Base-head 5 epoch

Training：

| epoch | train loss | val loss | train acc | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8.5605 | 7.9130 | 0.0422 | 0.0531 |
| 1 | 7.4678 | 7.1523 | 0.0453 | 0.0568 |
| 2 | 6.9107 | 6.6901 | 0.0503 | 0.0643 |
| 3 | 6.5063 | 6.4046 | 0.0580 | 0.0688 |
| 4 | 6.2825 | 6.2298 | 0.0643 | 0.0739 |

Generation 工程指标：

| model | avg frames | early stop | root path | pose velocity | pose variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1062 | 0.75 | 3.472 | 14.052 | 141.194 |
| base_head | 1170 | 0.50 | 3.538 | 29.972 | 277.800 |

结论：生成更长、早停更少，但 pose velocity/variance 增加明显。

### 7.4 Head-only ablation

Head-only 更保守：

| model | avg frames | early stop | root path | pose velocity | pose variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1062 | 0.75 | 3.472 | 14.052 | 141.194 |
| head-only | 1194 | 0.50 | 4.187 | 19.133 | 190.011 |
| base_head | 1170 | 0.50 | 3.538 | 29.972 | 277.800 |

Head-only 是当时最好的 engineering trade-off，但还没有 FID/R-precision。

## 8. Paper-style metric 准备

MoConVQ 论文中 Text2Motion 使用 HumanML3D test set 的：

```text
FID
R-precision
```

需要 HumanML3D/T2M motion feature extractor。MoConVQ 生成的是 character BVH，
不能直接喂给 HumanML3D evaluator，于是实现：

```text
MoConVQ BVH
  -> parse BVH + forward kinematics
  -> resample to 20 FPS
  -> approximate HumanML3D 22 joints
  -> HumanML3D process_file()
  -> 263-d new_joint_vecs
  -> T2M evaluator
```

新增：

```text
Script/stage1/bvh_to_humanml3d_features.py
Script/stage1/calibrate_bvh_to_humanml3d_adapter.py
Script/stage1/evaluate_t2m_paper_metrics.py
Script/stage1/prepare_t2m_evaluator_assets.py
```

Adapter calibration on 20 HumanML3D test samples：

```text
avg feature MAE = 0.0722
avg feature z MAE = 0.3049
avg feature z RMSE = 0.5694
avg feature z p95 abs = 1.3696
avg joint MPJPE = 0.0796
avg root position error ~= 4.85e-7
```

解释：

```text
root trajectory roundtrip 基本精确；
主要误差来自 MoConVQ 20-body skeleton 到 HumanML3D 22-joint skeleton 的近似，
尤其 spine/neck/head 插值。
```

最终指标必须标注为：

```text
approximate T2M evaluator-adapter metrics
```

不能说是完全等价于 MoConVQ paper 的原生 SMPL/HumanML3D 评估。

## 9. Segment-aligned final route

Long-native 200 sequence 路线解决了 token collapse，但还有一个不一致：

```text
training: 每个 window 可能用 full long caption
inference: prompt 按 " then " 分段，用 local caption 逐段生成
```

因此做了 segment-aligned cache：

```text
segment_idx
num_segments
segment_progress
segment_ranges
target_ranges
prefix_lengths
target_masks
end_masks
```

训练语义变成：

```text
previous segment prefix + local segment caption -> current segment tokens
```

Segment-aligned cache：

| split | windows | valid RVQ tokens | unique long sequences |
| --- | ---: | ---: | ---: |
| train | 476 | 85,328 | 73 |
| val | 117 | 20,756 | 18 |

Token distribution：

| split | depth0 top | depth1 top | depth2 top | depth3 top |
| --- | ---: | ---: | ---: | ---: |
| train | 0.0566 | 0.0247 | 0.0479 | 0.0700 |
| val | 0.0450 | 0.0334 | 0.0495 | 0.0934 |

这就是最终主结果使用的数据路线。

## 10. Explicit segment protocol

后来又发现：HumanML3D 原始 caption 内部可能包含 `then`，例如：

```text
they then scrub ...
right foot, then moves ...
left arm and then brings ...
```

如果推理时裸用：

```text
text.split(" then ")
```

会把一个训练时的 clip caption 错拆成多个 inference segment。

因此新增正式协议：

```text
prompt TSV:
name<TAB>long_text<TAB>segments_json<TAB>scaled_lengths_json

generate_long_motion.py:
--segments-json
--segment-lengths
```

Formal evaluation 用显式 HumanML3D clip boundary，而不是只靠 `" then "`。

可复现导出命令：

```bash
/home/chenjie/miniconda3/envs/moconvq/bin/python \
  Script/stage1/export_cache_prompt_tsv.py \
  --cache /tmp/stage1_segment_aligned_bvh_native_200_20260614/val_cache.pt \
  --output /tmp/stage1_segment_aligned_val18_explicit_segments_scaled75_prompts.tsv \
  --summary /tmp/stage1_segment_aligned_val18_explicit_segments_scaled75_prompts_summary.json \
  --total-length 75
```

结果：

```text
num_prompts = 18
```

## 11. Head-only segment-aligned results

### 11.1 Head-only training

```text
checkpoint dir = /tmp/stage1_segment_aligned_bvh_native_200_head_seed13_5ep_20260614
train_scope = head
epochs = 5
```

Training curve：

| epoch | train loss | val loss | train acc | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 14.6498 | 16.8632 | 0.0576 | 0.0701 |
| 1 | 14.3336 | 16.5705 | 0.0567 | 0.0702 |
| 2 | 13.9194 | 16.2719 | 0.0587 | 0.0708 |
| 3 | 13.5453 | 15.9782 | 0.0588 | 0.0711 |
| 4 | 13.1992 | 15.6747 | 0.0598 | 0.0713 |

### 11.2 Held-out Val8 plain `" then"` epoch3

| metric | baseline | finetuned epoch3 |
| --- | ---: | ---: |
| avg frames | 1296 | 1308 |
| early-stop rate | 0.375 | 0.25 |
| root path | 2.2293 | 2.3737 |
| FID | 18.1357 | 16.2093 |
| R@1 | 0.25 | 0.375 |
| R@2 | 0.625 | 0.625 |
| R@3 | 1.00 | 0.625 |
| matching | 4.3217 | 4.1888 |

结论：有 partial improvement，但 R@3 下降。

### 11.3 Explicit segment + scaled75 head-only

| metric | baseline | finetuned epoch3 |
| --- | ---: | ---: |
| avg frames | 1182 | 1194 |
| early-stop rate | 0.50 | 0.625 |
| root path | 1.6818 | 1.7485 |
| FID | 20.2790 | 20.2900 |
| R@1 | 0.375 | 0.500 |
| R@2 | 0.500 | 0.625 |
| R@3 | 0.625 | 0.750 |
| matching | 4.8132 | 4.6263 |

结论：更严格协议下 R-precision 和 matching 改善，但 FID 没有赢，early-stop 变差。
这说明 data/protocol 修复还不够，head-only capacity 不足。

## 12. 最终 base_head micro fine-tune

在同一个 segment-aligned native cache 上，用更小学习率微调 `base_head`：

```text
checkpoint dir = /tmp/stage1_segment_aligned_bvh_native_200_basehead_seed13_3ep_20260614
train_scope = base_head
trainable parameters = 30,577,152
lr = 5e-6
epochs = 3
progress_conditioning = auto
progress_scale = 0.5
context_size = 51
```

Training curve：

| epoch | train loss | val loss | train acc | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13.8335 | 14.9117 | 0.0579 | 0.0707 |
| 1 | 11.7993 | 12.5843 | 0.0573 | 0.0715 |
| 2 | 9.8848 | 10.7311 | 0.0598 | 0.0719 |

这个结果说明 segment-aligned native cache 是可学习的，head-only 之前主要是容量不足。

## 13. 最终主要指标

### 13.1 最终 Val8 主正结果

推荐用于 oral presentation 的主表：

```text
checkpoint = checkpoint_epoch_3.pth
protocol = explicit segment JSON + scaled segment lengths
total length = 75
top_p = 0.95
temperature = 1.0
```

| metric | baseline | fine-tuned |
| --- | ---: | ---: |
| avg frames | 1182 | 1197 |
| early-stop rate | 0.50 | 0.50 |
| root path | 1.6818 | 2.0738 |
| root displacement | 0.5340 | 0.8632 |
| pose velocity / variance | 16.104 / 181.560 | 17.732 / 193.894 |
| lag20 repeat fraction | 0.0020 | 0.0028 |
| approximate FID lower is better | 20.2790 | 14.9851 |
| approximate R@1 higher is better | 0.375 | 0.375 |
| approximate R@2 higher is better | 0.500 | 0.750 |
| approximate R@3 higher is better | 0.625 | 0.875 |
| approximate matching lower is better | 4.8132 | 4.3839 |

结论：

```text
FID / R@2 / R@3 / matching / root path 改善；
R@1 和 early-stop 持平；
pose energy 和 lag20 repetition 略高。
```

### 13.2 完整 Val18 保守结果

Val18 上最保守的 checkpoint 是 epoch2：

| metric | baseline | fine-tuned epoch2 |
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
| approximate matching lower is better | 4.8802 | 4.7885 |

结论：

```text
FID / R@3 / matching / root path / root displacement / repetition 改善；
R@1/R@2 持平；
early-stop 变差。
```

### 13.3 Val18 epoch3 trade-off

epoch3 在 Val18 上 FID/R@1/matching 更好，但 R@2 下降：

| metric | baseline | fine-tuned epoch3 |
| --- | ---: | ---: |
| FID | 13.7255 | 12.6332 |
| R@1 | 0.2222 | 0.2778 |
| R@2 | 0.4444 | 0.2778 |
| R@3 | 0.4444 | 0.4444 |
| matching | 4.8802 | 4.6093 |

讲 presentation 时建议这么说：

```text
Epoch3 gives stronger FID/R@1/matching, while epoch2 is safer for full Val18
because it avoids R@1/R@2 regression and improves R@3.
```

### 13.4 Negative decoding result

降低 sampling entropy 不是答案：

```text
checkpoint = epoch2
top_p = 0.90
temperature = 0.8
```

| metric | baseline | fine-tuned |
| --- | ---: | ---: |
| FID | 13.2935 | 14.3544 |
| R@1 | 0.3333 | 0.2778 |
| R@2 | 0.4444 | 0.3889 |
| R@3 | 0.5556 | 0.5000 |
| matching | 4.8050 | 4.8366 |

这个设置让 baseline 更强、fine-tuned 更差，所以最终不采用。

## 14. 最终结论怎么讲

### 14.1 一句话结论

```text
Stage1 成功构建了一个 HumanML3D -> MoConVQ native character retarget ->
text-conditioned MoConGPT fine-tuning -> long text generation -> approximate
FID/R-precision evaluation 的可复现 pipeline。修复数据映射和训练/推理
segment 对齐后，fine-tuned model 在长多阶段 prompt 上相对 baseline 取得
partial but meaningful improvement，但还不是完全解决。
```

### 14.2 主要贡献

1. 找出旧路线失败的核心：不是长序列合成本身，而是 hand-written
   HumanML3D-to-MoConVQ body-state/cache 映射导致 token collapse。
2. 用 BVH-to-character native retarget 替换旧 cache 路线，显著改善 RVQ token
   distribution。
3. 修复训练/推理 latent space 不一致：训练使用与 `model.sample()` 一致的 4-layer
   RVQ latent context。
4. 引入 segment-prefix / segment-aligned 训练，让模型学习
   `previous motion context + current local caption`。
5. 引入 explicit segment JSON + segment lengths，解决 HumanML3D caption 内部
   `then` 被误拆的问题。
6. 接入 approximate T2M evaluator route，能输出 FID/R-precision/matching。
7. 保留并实现 LLM in-context token planning backup 工程路径，但最终结果没用它。

### 14.3 局限性

1. FID/R-precision 是 approximate evaluator-adapter route，不是原生 SMPL 评估。
2. Val18 上不同 checkpoint 存在 R@1/R@2/R@3 trade-off，没有所有 retrieval cutoff
   全赢。
3. 视频上仍有姿态不自然和语义细节失败。
4. 训练数据规模仍小：最终 segment-aligned cache 是 73 train / 18 val long sequences。
5. evaluator 会截断 long sequence 到 196 frames at 20 FPS，因此长程后半段语义不能被
   FID/R-precision 完整衡量。

## 15. Oral presentation 建议结构

### Slide 1: Problem

```text
Goal: fine-tune MoConVQ text-conditioned GPT for long multi-stage text prompts.
Challenge: HumanML3D representation is not MoConVQ simulator character state.
```

### Slide 2: Why naive fine-tuning failed

```text
Old cache showed good token loss but bad videos.
Root causes:
  training/inference latent mismatch
  hand-written retarget token collapse
  long text vs local segment mismatch
```

### Slide 3: Diagnostics

Show:

```text
old fixed cache depth0 top fraction = 0.2171
native-retarget long cache depth0 top fraction ~= 0.05-0.06
```

Message:

```text
Long sequence synthesis was not the main bottleneck; representation mapping was.
```

### Slide 4: Final pipeline

Use this diagram:

```text
HumanML3D long sequences
  -> BVH export
  -> MoConVQ native character retarget
  -> encode_seq_all()
  -> segment-aligned GPT cache
  -> base_head fine-tune
  -> explicit segment inference
  -> BVH + videos + approximate FID/R-precision
```

### Slide 5: Training/inference consistency

```text
Do not split only by "then".
Use cache-exported segments_json and scaled_lengths_json.
```

### Slide 6: Main Val8 result

Show table:

```text
FID: 20.279 -> 14.985
R@2: 0.500 -> 0.750
R@3: 0.625 -> 0.875
matching: 4.813 -> 4.384
root path: 1.682 -> 2.074
```

### Slide 7: Full Val18 result

Show conservative epoch2 table:

```text
FID: 13.726 -> 13.060
R@3: 0.444 -> 0.500
matching: 4.880 -> 4.789
R@1/R@2 tie baseline
```

Also mention epoch3 trade-off:

```text
epoch3 improves FID/R@1/matching more strongly but loses R@2.
```

### Slide 8: Video

Play:

```text
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614/video/train_000057__baseline_vs_basehead.mp4
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614/video/train_000077__baseline_vs_basehead.mp4
```

Message:

```text
Fine-tuned is more persistent and covers more trajectory, no catastrophic
collapse, but semantic precision is still limited.
```

### Slide 9: Limitations

```text
approximate evaluator adapter
small final training set
mixed Val18 R-precision
visual artifacts remain
```

### Slide 10: Final takeaway

```text
Stage1 is complete as a reproducible pipeline and partial improvement result.
The next stage should scale accepted BVH-native data and improve retarget/evaluator fidelity.
```

## 16. 关键代码与文档位置

核心脚本：

```text
Script/stage1/export_long_humanml3d_to_bvh.py
Script/stage1/build_bvh_character_gpt_cache.py
Script/stage1/train_real_text_gpt.py
Script/stage1/generate_long_motion.py
Script/stage1/run_stage1_model_suite.py
Script/stage1/export_cache_prompt_tsv.py
Script/stage1/evaluate_t2m_paper_metrics.py
Script/stage1/bvh_to_humanml3d_features.py
Script/stage1/llm_token_planning.py
```

文档：

```text
STAGE1_README.md
STAGE1_EXPERIMENT_LOG.md
STAGE1_FINAL_RESULT_SUMMARY.md
STAGE1_METHOD_RESULTS_FOR_PRESENTATION.md
```

最终结果 artifact：

```text
Val8:
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614
/tmp/stage1_t2m_paper_metrics_segment_aligned_basehead_epoch3_val8_explicit_scaled75_20260614/summary.json

Val18 epoch2:
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch2_val18_explicit_scaled75_compare_20260614
/tmp/stage1_t2m_paper_metrics_segment_aligned_basehead_epoch2_val18_explicit_scaled75_20260614/summary.json

Val18 epoch3:
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val18_explicit_scaled75_compare_20260614
/tmp/stage1_t2m_paper_metrics_segment_aligned_basehead_epoch3_val18_explicit_scaled75_20260614/summary.json
```

## 17. 报告中推荐措辞

可以直接使用：

```text
We construct a reproducible Stage1 pipeline for long-horizon text-to-motion
fine-tuning in MoConVQ.  The key finding is that the naive HumanML3D-to-MoConVQ
state mapping, rather than long-sequence synthesis itself, caused token collapse
and unstable generation.  We therefore route synthesized HumanML3D motions
through BVH export and MoConVQ's native character retargeting path, then build a
segment-aligned GPT cache and fine-tune the text-conditioned MoConGPT with
training/inference-consistent segment prompts.

Under the strict explicit-segment protocol, the fine-tuned model improves over
the original baseline on approximate FID, selected R-precision cutoffs, matching
score and root/path coverage.  The improvement is partial: full Val18
R-precision remains mixed, early stopping is not fully solved, and the visual
quality still has semantic and pose artifacts.  The reported FID/R-precision
numbers are approximate evaluator-adapter metrics because generated MoConVQ BVHs
are converted to HumanML3D 22-joint / 263-d features before evaluation.
```

