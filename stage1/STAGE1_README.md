# Stage1 真实实验交接说明

本文档用于说明本仓库当前 Stage1 的目标、已经实现的内容、运行方式、上传 GitHub 时应该包含哪些文件，以及后续还需要完成的工作。

## 1. 仓库目的

本项目用于课程作业 Stage1：利用本地 `HumanML3D` 数据集合成长动作-文本序列，并将这些长序列转换成 MoConVQ 中 text-conditioned GPT 模型可以训练的格式，最后微调 `MoConVQ` 仓库里的 GPT 模型。

整体目标链路是：

```text
HumanML3D short clips
  -> transition-constrained long motion-language sequences
  -> MoConVQ 20-body state
  -> MoConVQ 323-d observation
  -> MoConVQ encoder encode_seq_all()
  -> latent_vq + RVQ token indices
  -> T5 text features
  -> fine-tune Text2Motion_Transformer
```

注意：这里的 GPT 不是 HuggingFace 的 `CausalLM` 架构。它是 MoConVQ 自己实现的 `Text2Motion_Transformer`，输入不是 `input_ids/labels`，而是：

```text
motion latent:  (B, T, 768)
RVQ indices:    (B, T, 4)
text features:  (B, L, 1024)
text mask:      (B, L)
clip feature:   (B, 512), 当前默认全 0
```

训练目标是预测每帧的 4 层 RVQ codebook token，而不是预测文本 token。

## 0. 当前结论

截至当前版本，Stage1 的端到端测试链路已经跑通：

```text
HumanML3D 长序列合成
  -> MoConVQ observation/cache 构建
  -> Text2Motion_Transformer 微调
  -> 文本生成 BVH
```

已经完成过一次真实 cache 构建、GPT 微调和 baseline/finetuned BVH 对比。代码层面可以从合成数据一路跑到生成 `.bvh` 文件，说明工程链路是连通的。

但此前训练出的 checkpoint 不能作为有效 finetune 结果继续使用。排查发现旧训练代码和推理路径存在两个关键不一致：训练时使用 cache 中的 `latent_vq` 作为上下文 latent，但该 `latent_vq` 来自 MoConVQ encoder 的 8 层 RVQ 总和；Text2Motion GPT 推理时实际只采样前 4 层 RVQ token，并把前 4 层 codebook embedding 求和作为下一步上下文。也就是说，旧训练看到的是 8-layer latent，上线推理看到的是 4-layer latent。同时旧实验默认全量更新 GPT，容易覆盖 baseline 已经具备的文本语义和运动先验；这解释了“baseline 动作还行但没做完，finetuned 更长但视觉效果很差”的现象。

当前代码已修复为：训练时从 `indices` 动态重建与 `model.sample()` 一致的前 4 层 RVQ latent，再用 `previous latent -> current RVQ indices` 训练目标；并新增 `--train-scope {all,base_head,head}` 支持保守微调。需要用修复后的代码重新训练，旧 `fixed_dataset_stage1_20260529_135401` checkpoint 不应再用于效果结论。

```text
测试链路已成功跑通；旧 finetune checkpoint 已判定为无效反例；后续应基于修复后的 GPT 上下文 latent 逻辑重新训练，并优先使用保守微调范围评估。
```

2026-06-13 更新：当前更准确的 Stage1 诊断是：

```text
HumanML3D 长序列合成本身基本可用；
旧 hand-written HumanML3D -> MoConVQ body state/cache 路径会造成明显 token collapse；
long_sequences.h5 -> BVH -> MoConVQ 原生 character retarget 是当前最可信主线；
修复后的 train_real_text_gpt.py 能在该 native cache 上稳定下降 loss；
finetuned 生成在工程指标的平均长度和早停率上超过 baseline；
HumanML3D/T2M evaluator assets 已经补齐，但当前 FID/R-precision 仍通过
MoConVQ BVH -> approximate HumanML3D 22-joint adapter 计算，不等价于论文原生 SMPL 评估。
```

2026-06-14 artifact 更新：当前最推荐的 Stage1 结果来自
`segment-aligned` BVH-native 实验。这个版本修复了一个关键训练/推理不一致：
训练 cache 使用局部 segment caption，推理也用同一个字面 joiner `" then "` 拆分长文本，
每个 segment 用本地文本条件和前一段生成 latent 作为上下文。
后续诊断又发现一个更细的粒度问题：HumanML3D 原始 clip caption 本身可能包含
`then`，所以裸 `text.split(" then ")` 会把部分训练时的单个 clip caption 误拆成
更小推理段。代码现在额外支持 `prompts.tsv` 第三列显式 segment JSON /
`generate_long_motion.py --segments-json`，以及第四列 JSON segment lengths /
`--segment-lengths`，用于严格按训练 clip 边界和段长比例推理；旧两列 TSV
和默认 `" then "` 行为保持兼容。`evaluate_t2m_paper_metrics.py` 也已同步兼容
2/3/4 列 prompt TSV，评估时仍用第二列完整长文本计算 text-motion matching。

当前更推荐的训练/推理一致 checkpoint 有两个角色：

```text
/tmp/stage1_segment_aligned_bvh_native_200_basehead_seed13_3ep_20260614/checkpoint_epoch_3.pth
/tmp/stage1_segment_aligned_bvh_native_200_basehead_seed13_3ep_20260614/checkpoint_epoch_2.pth
```

这两个 checkpoint 使用同一个 segment-aligned native cache，但把微调范围从
`head` 扩大到 `base_head`，学习率降到 `5e-6`，训练 3 epoch。`checkpoint_epoch_3`
在 Held-out Val8 的 stricter explicit segment + scaled segment lengths 协议下
取得最强正结果，并且已经有 side-by-side MP4；`checkpoint_epoch_2` 在完整
Val18 strict prompt 上更稳，approximate FID、R@3 和 matching score 优于
baseline，同时 R@1/R@2 与 baseline 持平。因此最终报告中可把 epoch3 作为
小样本强正结果和视频展示，把 epoch2 作为更大 Val18 上更保守的指标选择。

之前的 plain-`" then "` metric-balanced checkpoint 仍保留为历史对照：

```text
/tmp/stage1_segment_aligned_bvh_native_200_head_seed13_5ep_20260614/checkpoint_epoch_3.pth
```

`checkpoint_epoch_5.pth` 的 validation loss 更低，但 Val8 approximate
paper metrics 更混合；head-only `checkpoint_epoch_3.pth` 是 plain-`" then "`
协议下更适合作为对照的 checkpoint selection。

Segment-aligned cache：

| item | result |
| --- | --- |
| route | HumanML3D long sequence -> BVH -> MoConVQ native character retarget |
| train cache | 476 windows, 85,328 valid RVQ tokens, 73 long sequences |
| val cache | 117 windows, 20,756 valid RVQ tokens, 18 long sequences |
| train token top fraction | depth0 0.0566, depth1 0.0247, depth2 0.0479, depth3 0.0700 |
| training | initial head-only, 5 epochs, val loss 16.863 -> 15.675; selected base_head, 3 epochs, val loss 14.912 -> 10.731 |
| generation convention | `generation_mode=auto`, `segment_joiner=" then "`, `top_p=0.95` |

Held-out Val8 long-caption suite:

| metric | baseline | finetuned |
| --- | ---: | ---: |
| avg frames | 1296 | 1308 |
| early-stop rate | 0.375 | 0.25 |
| root path | 2.2293 | 2.3737 |
| pose velocity / variance | 16.073 / 158.511 | 16.016 / 166.197 |
| lag20 repeat fraction | 0.0064 | 0.0105 |
| approximate FID lower is better | 18.1357 | 16.2093 |
| approximate R-precision@1 higher is better | 0.25 | 0.375 |
| approximate R-precision@2 higher is better | 0.625 | 0.625 |
| approximate R-precision@3 higher is better | 1.00 | 0.625 |
| approximate matching score lower is better | 4.3217 | 4.1888 |

Val8 artifacts：

```text
/tmp/stage1_segment_aligned_bvh_native_200_head_epoch3_val8_compare_20260614
/tmp/stage1_t2m_paper_metrics_segment_aligned_head_epoch3_val8_20260614/summary.json
```

视觉结论：Val8 contact sheet 未出现空帧、明显倒置或整体发散；
`train_000077` 中 baseline 最后接近地面停住，finetuned 能从蹲/跪动作回到站姿。
但语义细节仍不完全稳定，所以当前结论应表述为“partial but meaningful improvement”：
FID、R@1、matching score 和早停率优于 baseline，R@2 持平，R@3 仍低于 baseline。

显式 clip-boundary Val8 诊断：

| metric | baseline | finetuned epoch3 | finetuned epoch5 |
| --- | ---: | ---: | ---: |
| avg frames | 1185 | 1242 | 1242 |
| early-stop rate | 0.50 | 0.375 | 0.375 |
| root path | 1.8738 | 1.9619 | 1.9595 |
| approximate FID lower is better | 16.1725 | 17.0217 | 17.0251 |
| approximate R@1/R@2/R@3 | 0.25 / 0.25 / 0.75 | 0.25 / 0.25 / 0.75 | 0.25 / 0.25 / 0.75 |
| approximate matching score lower is better | 5.0352 | 5.0437 | 5.0782 |

显式边界加训练段长比例的 scaled75 诊断：

| metric | baseline | head epoch3 | base_head epoch3 |
| --- | ---: | ---: | ---: |
| avg frames | 1182 | 1194 | 1197 |
| early-stop rate | 0.50 | 0.625 | 0.50 |
| root path | 1.6818 | 1.7485 | 2.0738 |
| pose velocity / variance | 16.104 / 181.560 | 16.880 / 180.231 | 17.732 / 193.894 |
| lag20 repeat fraction | 0.0020 | 0.0020 | 0.0028 |
| approximate FID lower is better | 20.2790 | 20.2900 | 14.9851 |
| approximate R@1/R@2/R@3 higher is better | 0.375 / 0.500 / 0.625 | 0.500 / 0.625 / 0.750 | 0.375 / 0.750 / 0.875 |
| approximate matching score lower is better | 4.8132 | 4.6263 | 4.3839 |

base_head artifacts:

```text
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614
/tmp/stage1_t2m_paper_metrics_segment_aligned_basehead_epoch3_val8_explicit_scaled75_20260614/summary.json
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614/contact_sheet.png
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614/video/train_000057__baseline_vs_basehead.mp4
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val8_explicit_scaled75_compare_20260614/video/train_000077__baseline_vs_basehead.mp4
```

为了避免 Val8 prompt TSV 依赖手工重建，新增了可复现导出脚本：

```bash
python Script/stage1/export_cache_prompt_tsv.py \
  --cache /tmp/stage1_segment_aligned_bvh_native_200_20260614/val_cache.pt \
  --output /tmp/stage1_segment_aligned_val18_explicit_segments_scaled75_prompts.tsv \
  --summary /tmp/stage1_segment_aligned_val18_explicit_segments_scaled75_prompts_summary.json \
  --total-length 75
```

该脚本直接从 segment-prefix cache 的 `captions`、`segment_ranges`、`segment_idxs`
和 `num_segments` 还原每条长序列的真实 clip caption 边界，并按原始 segment
长度比例缩放到给定 token budget。它输出 4 列 TSV：

```text
name<TAB>long_text<TAB>segments_json<TAB>scaled_lengths_json
```

用这个 TSV 对完整 18 条 held-out validation sequence 做 Val18 严格协议复现：

| metric | baseline | base_head epoch3 |
| --- | ---: | ---: |
| avg frames | 1292 | 1304 |
| early-stop rate | 0.2778 | 0.2778 |
| root path | 2.6053 | 2.7979 |
| root displacement | 0.8678 | 1.0033 |
| pose velocity / variance | 27.334 / 339.697 | 28.353 / 356.831 |
| lag20 repeat fraction | 0.0075 | 0.0063 |
| approximate FID lower is better | 13.7255 | 12.6332 |
| approximate R@1/R@2/R@3 higher is better | 0.222 / 0.444 / 0.444 | 0.278 / 0.278 / 0.444 |
| approximate matching score lower is better | 4.8802 | 4.6093 |

Val18 artifacts:

```text
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val18_explicit_scaled75_compare_20260614
/tmp/stage1_t2m_paper_metrics_segment_aligned_basehead_epoch3_val18_explicit_scaled75_20260614/summary.json
/tmp/stage1_segment_aligned_bvh_native_200_basehead_epoch3_val18_explicit_scaled75_compare_20260614/contact_sheet.png
```

Val18 结论更混合但仍支持主结论：finetuned 在 approximate FID、R@1、
matching score、平均长度、root path、root displacement 和 lag20 重复率上优于
baseline；R@2 下降，R@3 与 early-stop 持平，pose energy 略高。contact sheet
未见空帧、整体倒置或爆炸，部分 crouch/crawl 样例保持了低姿态，但仍有姿态怪异
和语义细节不稳定的失败模式。最终报告中可把 Val8 作为 strict protocol 主正结果，
Val18 作为更大样本复现和局限性说明。

进一步做了 Val18 checkpoint / decoding sweep，用同一个 4 列 prompt TSV、
explicit segment JSON 和 scaled segment lengths，比较 base_head epoch1/2/3
以及一个更保守的 decoding 设置：

| Val18 setting | baseline FID | finetuned FID | baseline R@1/R@2/R@3 | finetuned R@1/R@2/R@3 | baseline match | finetuned match |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| epoch1, top_p=0.95, temp=1.0 | 13.7255 | 14.1481 | 0.222/0.444/0.444 | 0.222/0.389/0.500 | 4.8802 | 4.6518 |
| epoch2, top_p=0.95, temp=1.0 | 13.7255 | 13.0602 | 0.222/0.444/0.444 | 0.222/0.444/0.500 | 4.8802 | 4.7885 |
| epoch3, top_p=0.95, temp=1.0 | 13.7255 | 12.6332 | 0.222/0.444/0.444 | 0.278/0.278/0.444 | 4.8802 | 4.6093 |
| epoch2, top_p=0.90, temp=0.8 | 13.2935 | 14.3544 | 0.333/0.444/0.556 | 0.278/0.389/0.500 | 4.8050 | 4.8366 |

结论是：epoch3 的 FID、R@1 和 matching 最强，但 R@2 回落；epoch2 的收益
更保守，FID、R@3 和 matching 改善，R@1/R@2 不退步，因此更适合作为 Val18
稳定性 checkpoint。降低采样温度和 top-p 的 conservative decoding 是负结果：
它提升了 baseline，却让 finetuned 全部 approximate T2M 指标变差。epoch2 的
Val18 contact sheet 同样未见空帧、整体倒置或爆炸，但仍存在姿态弯折和语义细节
失败；因此视频结论仍应表述为“比旧失败路线稳定，finetuned 在持续性和覆盖上有
小幅改善，但还不够自然”。

旧 head-only scaled75 诊断：

| metric | baseline | finetuned epoch3 |
| --- | ---: | ---: |
| avg frames | 1182 | 1194 |
| early-stop rate | 0.50 | 0.625 |
| root path | 1.6818 | 1.7485 |
| pose velocity / variance | 16.104 / 181.560 | 16.880 / 180.231 |
| lag20 repeat fraction | 0.0020 | 0.0020 |
| approximate FID lower is better | 20.2790 | 20.2900 |
| approximate R@1/R@2/R@3 higher is better | 0.375 / 0.500 / 0.625 | 0.500 / 0.625 / 0.750 |
| approximate matching score lower is better | 4.8132 | 4.6263 |

这说明显式 segment 边界是更严格的训练/推理一致性修复；head-only 容量不足，
虽然改善 R-precision/matching，但 FID 和 early-stop 没有赢。`base_head`
微调后，finetuned 在 approximate FID、R@2/R@3、matching score、平均帧数和
root path 上超过 baseline，R@1 与 early-stop 持平；lag20 repetition 和
pose variance 略高。contact sheet 未见空帧、整体倒置或爆炸，`train_000057`
和 `train_000077` 已输出 side-by-side MP4 供人工检查。最终报告应把它写成
“stricter protocol 下的主要正结果”，并标注 approximate evaluator-adapter
route 的局限。

四个手写 prompt suite 的结果也支持部分改进：finetuned 提升平均长度
`1062 -> 1194`、早停率 `0.75 -> 0.50`、approximate FID `28.273 -> 25.903`，
但 R-precision@1 从 `0.50` 降到 `0.25`，说明小样本手写 prompt 的语义指标仍不稳。

上一阶段的工程最好结果来自 200 条 long HumanML3D 的 BVH-native 实验：

| item | result |
| --- | --- |
| exported long BVH | 200 |
| native-retarget accepted | 91 |
| train/val split | 73 / 18 sequences |
| train cache | 278 windows, 55,516 valid RVQ tokens |
| val cache | 66 windows, 13,200 valid RVQ tokens |
| train token top fraction | depth0 0.063, depth1 0.022 |
| training | head-only, 5 epochs, val loss 8.980 -> 8.443 |
| generation avg frames | baseline 1062, finetuned 1194 |
| generation early-stop rate | baseline 0.75, finetuned 0.50 |
| pose velocity / variance | baseline 14.052 / 141.194, finetuned 19.133 / 190.011 |
| visual audit | no obvious fall-over/full-body inversion in contact sheet; finetuned is longer with moderate, not extreme, motion-energy increase |

This is the first run where the main HumanML3D reconstruction route gives a
measurable generation-side improvement over the MoConVQ baseline on the Stage1
engineering metrics.  At the time of that run, it was not a paper-metric claim
because the pretrained HumanML3D evaluator assets had not yet been installed.
The later segment-aligned run above adds approximate FID/R-precision through
the local T2M evaluator adapter route.

That historical run can be summarized reproducibly with:

```bash
cd /home/chenjie/cc/robotics/MoConVQ

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

## 2. 工作区结构

当前工作区在：

```text
./robotics/
```

主要包含两个重要目录：

```text
HumanML3D/   # 本地数据集与 HumanML3D 原始/中间处理文件
MoConVQ/    # 主体代码仓库，Stage1 代码都放在这里
```

本项目默认使用以下本地路径：

```text
HumanML3D 数据根目录: ../HumanML3D/HumanML3D
MoConVQ 主仓库:       robotics/MoConVQ
预训练 MoConVQ:       moconvq_base.data
预训练 GPT:           text_generation_GPT.pth
输出目录:             stage1_artifacts/
```

已配置环境：moconvq

### 2.1 远程 main/stage1 同步约定

真实实验工作目录是：

```text
/home/chenjie/cc/robotics/MoConVQ
```

GitHub 远程 `origin/main` 的仓库根目录不是 MoConVQ 本身，而是：

```text
README.md
stage1/
```

因此 `MoConVQ/` 不能直接 track `origin/main` 的 `stage1/` 子目录。当前采用的安全工作流是：

```text
在 /home/chenjie/cc/robotics/MoConVQ 中运行实验和改代码
  -> 同步可提交文件到 /home/chenjie/cc/robotics/MoConVQ-main/stage1
  -> 从 /home/chenjie/cc/robotics/MoConVQ-main 提交
  -> git push origin HEAD:main
```

同步脚本：

```bash
cd /home/chenjie/cc/robotics/MoConVQ
python Script/stage1/sync_stage1_to_main_worktree.py
```

脚本默认排除本地实验大文件和私有 agent 文档：

```text
stage1_artifacts/
*.h5
*.pth
*.data
AGENT.md
AGENTS.md
CODEX.md
CLAUDE.md
.codex/
.claude/
```

推送前必须从 `MoConVQ-main` 检查：

```bash
cd /home/chenjie/cc/robotics/MoConVQ-main
git status --short --branch
git ls-files | rg -i '(^|/)(AGENT\.md|AGENTS\.md|CODEX\.md|CLAUDE\.md|\.codex/|\.claude/)'
git diff --cached --name-only | rg -i '(^|/)(AGENT\.md|AGENTS\.md|CODEX\.md|CLAUDE\.md|\.codex/|\.claude/)'
```

如果后两个命令有输出，说明私有 agent 文档被纳入 Git，需要先移除再提交。

## 3. 已完成内容

### 3.1 HumanML3D 数据读取

文件：

```text
Script/stage1/humanml3d.py
```

功能：

- 读取 `HumanML3D/HumanML3D/all.txt` 和 split 文件；
- 为每个 sample 建立 `texts/new_joints/new_joint_vecs/index.csv` 的索引；
- 明确使用 `HumanML3D/HumanML3D` 作为 canonical dataset；
- 避免直接枚举 `joints/` 这类中间目录。

当前验证过的数据数量：

```text
all:       29228
train:     23384
val:        1460
test:       4384
train_val: 24844
```

### 3.2 长动作序列合成

文件：

```text
Script/stage1/synthesize_long_humanml3d.py
```

功能：

- 从指定 HumanML3D split 中采样短 clip；
- 按过渡约束选择后续 clip，而不是完全随机拼接；
- transition score 包含：
  - 根关节末帧/首帧位置差；
  - 根关节速度差；
  - 面向方向 yaw 差；
  - 脚部高度差；
  - 脚部速度差；
- 对后续 clip 做根位置和 yaw 对齐；
- 使用 `blend-frames` 对拼接边界做短过渡平滑；
- caption 使用 `" then "` 拼成长文本。
- 默认拒绝超过 `--transition-max-score` 的 forced transition，避免大量不连续边界污染训练；如果要复现旧数据，必须显式传 `--allow-forced-transitions`。

输出：

```text
manifest.jsonl
long_sequences.h5
summary.json
```

示例命令：

```bash
cd /home/chenjie/cc/robotics/MoConVQ
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq

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

### 3.3 HumanML3D 到 MoConVQ token cache

文件：

```text
Script/stage1/convert_humanml3d_to_moconvq_observation.py
Script/stage1/real_moconvq_cache.py
Script/stage1/build_real_moconvq_gpt_cache.py
```

功能：

- 读取合成后的 `long_sequences.h5`；
- 将 HumanML3D 的 `(T, 22, 3)` joints retarget 到 MoConVQ 的 20-body state；
- 构造 MoConVQ state，shape 为 `(T, 20, 13)`；
- 调用 MoConVQ 的 `state2ob()` 得到 `(T, 323)` observation；
- 加载 `moconvq_base.data`；
- 调用 `agent.encode_seq_all(None, observation)` 得到：
  - `latent_vq`: `(T_latent, 768)`
  - `indices`: `(T_latent, 4)`
- 使用 T5 编码文本，默认 `t5-large`；
- 按 `window-size=50` 和 `window-stride=25` 切成训练窗口；
- 保存为 GPT 训练 cache。

窗口长度默认是 50，因为 `Text2Motion_Transformer` 的 `block_size=52` 会在 motion latent 前额外加入一个 condition token；cache 构建脚本会拒绝 `window-size > 51`。默认 `--max-text-length 256` 会把 T5 文本特征固定为 `(256, 1024)`，过长 caption 会按 T5 tokenizer 截断；`--caption-mode` 当前默认是 `window`，让每个 50-token motion window 使用对应局部 caption。

HumanML3D 的 `new_joints` 只有关节位置，不包含 MoConVQ 物理角色刚体的局部坐标系旋转。当前 cache 默认使用 `--rotation-calibration rest`，把手写骨向量 quaternion 的静止姿态对齐到 `Data/Misc/world.json` 中 MoConVQ 角色的静止 body quaternion。不要把旧的未校准 cache 用作下一轮有效结论。

如果只想先检查 HumanML3D retarget 到 MoConVQ observation 是否合理，可以先运行独立转换脚本。它不会调用 GPT，也不会构建 T5 cache：

```bash
python Script/stage1/convert_humanml3d_to_moconvq_observation.py \
  --long-h5 stage1_artifacts/long_humanml3d/train/long_sequences.h5 \
  --manifest stage1_artifacts/long_humanml3d/train/manifest.jsonl \
  --output-h5 stage1_artifacts/long_humanml3d/train/moconvq_observations.h5 \
  --summary stage1_artifacts/long_humanml3d/train/moconvq_observations_summary.json
```

输出 H5 中每条序列包含：

```text
state_20x13:     (T, 20, 13)
observation_323: (T, 323)
```

cache 字段：

```text
latents:       (N, 50, 768)
indices:       (N, 50, 4)
text_features: (N, L, 1024)
text_masks:    (N, L)
captions:      list[str]
sequence_ids:  list[str]
window_ranges: list[tuple[int, int]]
sample_ids:    list[list[str]]
config:        dict
```

示例命令：

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
  --rotation-calibration rest \
  --gpu 0 \
  --output stage1_artifacts/gpt_cache/train_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/train_failures.jsonl
```

`--caption-mode sequence` 会把整条长序列 caption 复制给每个 window；`--caption-mode window` 会根据 `clip_boundaries` 给每个训练 window 选择重叠 clip 的局部 caption。真实长序列实验默认使用 `window`，因为它减少“当前动作窗口和整段长文本不对应”的噪声。

### 3.4 GPT 微调

文件：

```text
Script/stage1/train_real_text_gpt.py
```

功能：

- 构建 MoConVQ 原仓库的 `Text2Motion_Transformer`；
- 从 `moconvq_base.data` 读取 RVQ codebook embedding；
- 加载 `text_generation_GPT.pth` 作为初始化；
- 训练目标为每帧 4 层 RVQ token；
- 使用自回归对齐：`condition, reconstructed_latent[0], ..., reconstructed_latent[T-2]` 预测 `indices[0], ..., indices[T-1]`；
- `reconstructed_latent` 由 cache 中的前 4 层 RVQ `indices` 和 GPT codebook embedding 动态重建，和 `Text2Motion_Transformer.sample()` 推理时的 latent 空间保持一致；
- 使用 `logits[:, :, :4, :]` 对齐 4 层 RVQ depth；
- 支持 `--train-scope all/base_head/head`；早期实验把 `base_head` 作为保守起点，当前 2026-06-13 long-native 结果更推荐 `head`，因为它在长度/早停改善接近或更好时保留了更低的 pose velocity/variance；
- 支持 `--depth-weights`，可对 4 层 RVQ CE 加权，优先稳定前两层主体动作 token；
- 支持 `--baseline-kl-weight` 和 `--kl-temperature`，使用冻结的 baseline GPT 做 logits distillation，降低小规模 HumanML3D 微调破坏原始运动先验的风险；
- 支持 `--end-token-weight`，在 padding 后第一步加入小权重 end-token 辅助 loss，用于控制早停/不结束倾向；
- 支持 padding token `513` 的 ignore；
- 记录 train/val loss、CE loss、KL loss、end loss、token accuracy、per-depth accuracy；
- 保存 checkpoint 和日志。

示例命令：

```bash
python Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache/train_cache.pt \
  --val-cache stage1_artifacts/gpt_cache/val_cache.pt \
  --init-checkpoint text_generation_GPT.pth \
  --base-data moconvq_base.data \
  --output-dir stage1_artifacts/checkpoints/real_stage1 \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --train-scope base_head \
  --depth-weights 1.0,0.7,0.4,0.2 \
  --baseline-kl-weight 0.05 \
  --kl-temperature 2.0 \
  --end-token-weight 0.01 \
  --gpu 0 \
  --seed 0 \
  --save-every 1 \
  --num-workers 4
```

### 3.5 GPT forward 修复

文件：

```text
MoConVQCore/Model/cross_trans_ori_fixsum.py
```

修复内容：

- `trans_temporal()` 会把 clip condition token 拼到时间维前面；
- 原 forward 直接将这个 feature reshape 成 `(B*T, C)`，会导致 feature 比 indices 多一帧；
- 当前修复逻辑是在 forward 中检测并去掉额外 condition frame；
- 同时将 `.view()` 改为 `.reshape()`，避免非 contiguous tensor 出问题。

如果不上传这个修复，Stage1 GPT 微调可能出现 shape mismatch。

## 4. 已验证内容

运行环境：

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd /home/chenjie/cc/robotics/MoConVQ
```

单测命令：

```bash
python -m unittest \
  tests.test_stage1_humanml3d \
  tests.test_stage1_motion_bridge \
  tests.test_stage1_gpt \
  tests.test_stage1_real_synthesis \
  tests.test_stage1_real_cache \
  tests.test_stage1_real_train \
  tests.test_stage1_real_generate -v
```

最近一次验证结果：

```text
Ran 23 tests in 21.015s
OK
```

端到端实验状态：

- 已生成 `stage1_artifacts/long_humanml3d/` 下的 HumanML3D 合成长序列数据；
- 已构建真实 MoConVQ GPT cache：
  - `stage1_artifacts/gpt_cache/train_cache.pt`
  - `stage1_artifacts/gpt_cache/val_cache.pt`
- 已完成过一次旧版 `train_real_text_gpt.py` 微调，输出 checkpoint：
  - `stage1_artifacts/checkpoints/real_stage1/checkpoint_epoch_5.pth`
  - `stage1_artifacts/checkpoints/real_stage1/checkpoint_epoch_10.pth`
  - `stage1_artifacts/checkpoints/real_stage1/checkpoint_epoch_15.pth`
  - `stage1_artifacts/checkpoints/real_stage1/checkpoint_epoch_20.pth`
  - `stage1_artifacts/checkpoints/real_stage1/best_val.pth`
  - `stage1_artifacts/checkpoints/real_stage1/last.pth`
- 已使用 `checkpoint_epoch_5.pth` 和 baseline `text_generation_GPT.pth` 生成 BVH 对比：
  - `stage1_artifacts/generated_bvh_compare/real_stage1_epoch5_vs_baseline/`

注意：这些旧 checkpoint 是在错误训练目标下得到的，不能作为有效模型使用，只能作为问题样例保留。旧 `checkpoint_epoch_5.pth` 对应的日志指标：

```text
train loss:       0.09024
train token acc:  0.98169
val loss:         0.03926
val token acc:    0.99300
val depth acc:    0.99965 / 0.99044 / 0.99233 / 0.98957
```

这些指标虚高，不能说明模型学会了正确的自回归生成。修复后 1-batch smoke 的 loss 约为 6.74，明显高于旧指标，符合去掉当前帧/当前 depth 泄漏后的真实训练难度。

已生成的 epoch5 对比中，baseline 在长文本 prompt 上经常提前结束，而 epoch5 微调模型基本能生成到目标长度：

```text
walk_turn_return:      baseline 696 frames,  epoch5 2880 frames
walk_run_jump:         baseline 936 frames,  epoch5 2880 frames
circle_wave_crouch:    baseline 1176 frames, epoch5 2880 frames
sidestep_kick_turn:    baseline 408 frames,  epoch5 2880 frames
long_sequence_mixed:   baseline 1176 frames, epoch5 2880 frames
```

这只能说明旧错误模型更愿意生成长序列，不能说明动作语义和运动质量合格。实际 token 诊断显示旧 finetuned 模型后半段会塌缩到重复 RVQ tuple，因此这些结果不能作为最终实验结论。

额外 smoke test：

- 真实 encoder 小烟测：
  - `24x22x3 joints -> 24x20x13 state`
  - `24x20x13 state -> 24x323 observation`
  - `agent.encode_seq_all() -> 6x768 latent / 6x4 indices`
- 真实 HumanML3D 小规模合成：
  - 2 条 train 长序列；
  - 平均 348 帧；
  - 输出 `/tmp/stage1_real_synth_smoke`。
- 真实 MoConVQ encoder cache smoke：
  - 2 条长序列生成 5 个训练窗口；
  - `latents: (5, 50, 768)`
  - `indices: (5, 50, 4)`
- GPT 训练 smoke：
  - 使用真实 encoder cache 加注入式假 text feature；
  - `train_real_text_gpt.py --smoke` 完成 forward/backward/save；
  - 输出 `/tmp/stage1_real_train_smoke/last.pth`。

注意：`t5-large` 已下载到本地缓存目录并用于真实 cache/生成链路。后续如果换机器运行，需要确认本地模型路径或 HuggingFace cache 是否可用。

## 5. 旧 scaffold 与真实实验主线的关系

仓库中还保留了早期 scaffold：

```text
Script/stage1/build_long_horizon_manifest.py
Script/stage1/build_moconvq_token_cache.py
Script/stage1/train_text_gpt.py
Script/stage1/motion_bridge.py
Script/stage1/text_encoding.py
```

这些文件用于较早的快速 pipeline：

```text
HumanML3D 263-d vector -> heuristic 768-d latent -> RVQ quantization -> GPT smoke
```

它们适合 smoke test 和 debug，但不是当前真实实验主线。

当前真实实验主线应优先使用：

```text
Script/stage1/synthesize_long_humanml3d.py
Script/stage1/check_stage1_data_readiness.py
Script/stage1/export_humanml3d_to_bvh.py
Script/stage1/build_real_moconvq_gpt_cache.py
Script/stage1/build_bvh_character_gpt_cache.py
Script/stage1/summarize_bvh_retarget_quality.py
Script/stage1/split_bvh_quality_summary.py
Script/stage1/apply_bvh_quality_overrides.py
Script/stage1/train_real_text_gpt.py
```

当前 `/home/chenjie/cc/robotics/HumanML3D` 的 canonical processed payload
已经可用：`all.txt` 中的 29228 个样本均有对应的 `texts/new_joints/new_joint_vecs`。
但当前本地没有 `pose_data/`、标准 AMASS motion `.npz` 或大规模 BVH exports，
所以首选的 MoConVQ 原生 `MotionDataSet.add_bvh_with_character()` 路线还不能直接
在 HumanML3D 规模上构建 cache。先运行 `check_stage1_data_readiness.py` 确认
source motion/BVH 状态，再决定是恢复 AMASS/HumanML3D source motion，还是实现并验证
`new_joints` 到 MoConVQ-compatible BVH 的导出。

`export_humanml3d_to_bvh.py` 已经提供一个 processed HumanML3D 到 `base.bvh`
hierarchy 的桥接路径：导出的 BVH 可以被 MoConVQ 原生
`MotionDataSet.add_bvh_with_character()` 读取并构造 GPT cache。初版 `vec6d`
导出把 HumanML3D/T2M 局部 6D rotation 直接映射到 MoConVQ BVH rigid-body frame，
视觉上会出现倒置和肢体翻转；当前默认改为 `--rotation-source joints_ik`，用
`new_joints` 的骨骼方向估计 BVH 局部旋转，并保留 `--rotation-source vec6d`
用于复现实验对比。两条 smoke 样本上，IK 路径把 depth0 cache top fraction 从
约 `0.34` 降到约 `0.069`，native-retarget observation p99 `|z|` 从约 `12.06`
降到约 `7.89`，且 MP4 视觉检查不再出现大面积倒置。但 max `|z|` 仍约 `59.4`，
复杂动作仍有夸张弯折，因此扩大使用前仍必须做更大样本 token/observation 分布诊断、
质量过滤和人工视频检查。

`diagnose_bvh_character_retarget.py --per-file` 和
`summarize_bvh_retarget_quality.py` 已经把这一质量检查做成可复现筛选表。10 条
train split smoke 中，临时阈值接受 5 条、拒绝 5 条；拒绝原因包括高
observation z-score、短序列、以及低 z-score 但 depth0 token collapse。过滤后的
5 条样本 cache 有 18 个窗口、1440 个有效 token，depth0 top fraction 约 `0.039`，
并通过 head-only 训练 smoke。该结果只证明筛选闭环可跑，不是最终训练集规模或模型质量结论。

后续 50/100/500 条 train split 诊断已经验证相同闭环可以扩展到更大的
processed-HumanML3D 样本批次。batch50 临时阈值接受 10 条、拒绝 40 条；
accepted-only cache 有 39 个窗口、3120 个有效 token，depth0 top fraction 约
`0.041`。batch100 临时阈值接受 16 条、拒绝 84 条；accepted-only cache 有 61 个
窗口、4880 个有效 token，depth0 top fraction 约 `0.023`，并能进入
`train_real_text_gpt.py --train-scope head --smoke`。batch500 临时阈值接受 90 条、
拒绝 410 条；accepted-only cache 有 90 个 50-token 窗口、15804 个有效 token、
90 个唯一序列，depth0 top fraction 约 `0.038`。`split_bvh_quality_summary.py`
进一步将 batch500 accepted rows 按 seed 13 划分为 72 train / 18 val，并分别构建
train/val cache；MP4 审计后，`apply_bvh_quality_overrides.py` 生成 filtered-v2
summary：剔除 `013481` floor/prone accepted 样本，恢复 `010684` 与 `M012928`
两个可视上较合理的 walking/turning rejected 样本；v2 cache 有 73 train / 18 val
窗口，并通过 1 epoch head-only train/val 路径检查。主要拒绝原因是短序列、depth0 token collapse、tokens 少和少量 retarget
后 observation z-score 异常。这说明 HumanML3D 主线没有被放弃，但正式训练不能再
盲目使用早期 hand-written retarget cache；
`make_bvh_contact_sheet.py` 可把 quality summary 中的 accepted/rejected BVH 抽帧成
静态审计图，便于在正式训练前检查是否存在倒置、爆肢、明显脚滑或错误拒绝。
当前更可靠的主线是：

```text
processed HumanML3D new_joints/new_joint_vecs
  -> export_humanml3d_to_bvh.py --rotation-source joints_ik
  -> MoConVQ native MotionDataSet.add_bvh_with_character()
  -> per-file observation/token quality filter
  -> optional MP4-audit manual override
  -> deterministic accepted train/val quality split
  -> accepted-only GPT cache
  -> conservative MoConGPT fine-tuning
```

## 6. 还需要完成的工作

### 6.1 当前效果问题定位

当前最大问题不是脚本跑不通，而是生成结果质量不好。后续应优先定位以下问题：

- HumanML3D 22-joint skeleton 到 MoConVQ 20-body state 的 retarget 是否足够准确；
- `state2ob()` 之后再经过 `agent.encode_seq_all()` 得到的 latent/RVQ indices 是否能被 MoConVQ decoder 稳定还原；
- 合成长序列的拼接边界是否引入不自然速度、朝向或脚部状态突变；
- `--caption-mode window` 下每个 50-token motion window 对应的局部 caption 是否真正匹配该窗口动作；
- GPT 只在 50-token window 上训练，而推理时滚动生成更长序列，是否出现分布外累积误差；
- 修复后的训练 loss/accuracy 是否能在不发生 token collapse 的情况下改善生成；
- 现有评估主要依赖 BVH 视觉检查，还需要更系统的指标，例如生成长度、脚滑、root drift、重建误差、caption-action 对齐人工评分。

建议下一步先做一个最小闭环诊断：

```text
HumanML3D joints
  -> MoConVQ state/observation
  -> encode_seq_all()
  -> RVQ latent
  -> MoConVQ decoder/generate BVH
```

先不训练 GPT，只检查单条真实动作经过转换和 MoConVQ encode/decode 后是否还能生成合理动作。如果这一步质量差，问题主要在 retarget/cache；如果这一步质量可以，再继续查 GPT 训练和长文本生成。

### 6.2 生成效果评估与修复

当前已经能生成 BVH，但效果不好。建议保留以下对比目录作为问题样例：

```text
stage1_artifacts/generated_bvh_compare/real_stage1_epoch5_vs_baseline/
```

后续修复方向：

- 先用修复后的训练代码重新训练，不再使用旧 `real_stage1` checkpoint 作有效评估；
- 对比 baseline、修复后 epoch checkpoint、best_val、last 的同一组 prompt；
- 检查修复后模型是否仍然只是避免早停，但动作内容重复或语义不对；
- 对生成 BVH 做逐 prompt 人工记录，例如“是否转身”“是否跳跃”“是否蹲下”“是否明显脚滑”；
- 优先使用默认 `--generation-mode auto`，或显式 `--generation-mode segmented`，将长文本按 `" then "` 拆成局部 caption 分段生成；
- 后续评估必须使用当前 top-p / nucleus sampling 代码路径重新生成结果；旧 greedy 或固定 top-k 视频只能作为历史诊断，不能作为当前模型结论；
- 再尝试不同 `--chunk-size`、`--context-size`、`--temperature`、`--top-p`、`--top-k`，避免单一 decoding 策略固化坏模式；
- 将训练 cache 中的若干 window 反解成 BVH，检查训练目标本身是否可信；
- 若 retarget 问题明显，优先重做 HumanML3D 到 MoConVQ character 的转换，而不是继续调 GPT。

### 6.3 数据规模和训练配置

当前已经完成一次真实链路，但还需要系统复现实验。建议按规模逐步扩大和记录：

```text
10 samples -> 50 samples -> 100 samples -> larger accepted-only train/val cache
```

每一步确认：

- failure log 是否为空或可接受；
- cache 的 window 数是否合理；
- indices 是否在合法范围内；
- accepted/rejected contact sheet 或视频中是否出现明显 false positive/false negative；
- 训练 loss 是否下降。

同时不要只看 token accuracy。token accuracy 高但 BVH 差时，应优先检查数据转换和生成策略。

### 6.4 Backup Plan: LLM In-Context Motion Token Planning

如果后续确认 HumanML3D 直接拼接后的长序列质量不足，或者修复训练目标后 GPT 仍然在长文本上重复/语义错位，可以采用论文中 MoConVQ LLM integration 的备选路线：把 MoConVQ RVQ indices 当作紧凑动作表示，构建“文本描述 -> token 序列”的 example bank，让大模型通过 in-context learning 规划和重组 token，再用本地 MoConVQ decoder/controller 输出 BVH。

详细方案见：

```text
STAGE1_BACKUP_PLAN.md
```

### 6.5 T5 模型下载和缓存

`build_real_moconvq_gpt_cache.py` 默认使用：

```text
t5-large
```

如果机器上没有 HuggingFace 缓存，会在首次运行时下载。可能需要：

- 网络连接；
- HuggingFace cache 空间；
- `sentencepiece`；
- 足够 GPU/CPU 内存。

如果下载失败，不应自动退回 hash encoder，因为真实实验要求和原 MoConVQ text-to-motion 逻辑一致。

当前机器上已有本地 T5：

```text
/home/chenjie/cc/robotics/hf_models/t5-large
```

后续推荐直接传本地路径，减少联网依赖：

```bash
--text-model /home/chenjie/cc/robotics/hf_models/t5-large
```

### 6.6 Retarget 质量检查

当前 HumanML3D 到 MoConVQ 的 retarget 是确定性 kinematic 近似：

```text
HumanML3D 22 joints -> MoConVQ 20 bodies
```

当前映射已经按 HumanML3D 的左右肢定义修正：HumanML3D 右腿链 `2,5,8,11` 对应 MoConVQ `rUpperLeg/rLowerLeg/rFoot/rToes`，左腿链 `1,4,7,10` 对应左腿；上肢同理右臂 `17,19,21`、左臂 `16,18,20`。它已经经过 shape、MoConVQ encoder smoke test 和小样本 observation 转换测试，但还需要做系统的视觉质量评估。后续建议：

- 抽样保存 retarget 后的 state/observation；
- 通过 MoConVQ decoder 或 tracking 生成 BVH；
- 人眼检查拼接边界和身体姿态；
- 检查脚滑、朝向突变、手臂异常等问题。

如果视觉检查发现明显脚滑、左右肢异常或朝向错误，下一步应考虑更严格的 BVH/SMPL 到 MoConVQ character retarget。

### 6.7 Val cache 和评估指标

当前脚本支持 `--val-cache`，并已经跑通过真实 val cache。若需要重新构建，可运行：

```bash
python Script/stage1/synthesize_long_humanml3d.py \
  --humanml-root ../HumanML3D/HumanML3D \
  --split val \
  --num-sequences 200 \
  --min-clips 2 \
  --max-clips 4 \
  --seed 1 \
  --candidate-pool 256 \
  --transition-max-score 0.35 \
  --blend-frames 5 \
  --output-dir stage1_artifacts/long_humanml3d/val
```

如果需要复现旧数据才加：

```bash
--allow-forced-transitions
```

正式实验默认不要加该参数；否则此前 60% 以上 transition forced 的问题会重新出现。

然后构建 val cache：

```bash
python Script/stage1/build_real_moconvq_gpt_cache.py \
  --long-h5 stage1_artifacts/long_humanml3d/val/long_sequences.h5 \
  --manifest stage1_artifacts/long_humanml3d/val/manifest.jsonl \
  --base-data moconvq_base.data \
  --text-model t5-large \
  --window-size 50 \
  --window-stride 25 \
  --rvq-depth 4 \
  --caption-mode window \
  --gpu 0 \
  --output stage1_artifacts/gpt_cache/val_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/val_failures.jsonl
```

### 6.8 长动作生成与展示

当前 `generate_long_motion.py` 可用于训练后生成 BVH。它默认使用 `T5Tokenizer + T5EncoderModel`，和真实 cache 构建路径一致；如果只想离线调试文本形状，可以显式传 `--text-encoder hash`。

生成脚本默认 `--generation-mode auto`。如果文本能按 `--segment-joiner` 拆成多段，例如默认的 `" then "`，脚本会自动使用 segmented generation；否则使用 fixed-context rolling generation。rolling 模式下，`--max-length` 控制总 latent token 数，`--context-size` 控制每个 chunk 最多回看多少历史 latent，`--chunk-size` 控制每次新采样多少 token。由于 GPT 的 `block_size=52` 还包含一个 condition token，每轮实际历史长度会自动裁剪到 `51 - 当前chunk长度`，避免超过 position/mask 长度。文本侧仍由 `--max-text-length` 控制，默认 256，超长 prompt 会被 T5 tokenizer 截断。

示例：

```bash
python Script/stage1/generate_long_motion.py \
  --checkpoint stage1_artifacts/checkpoints/real_stage1/best_val.pth \
  --text "a person walks forward then turns around and waves" \
  --output-bvh stage1_artifacts/generated/demo.bvh \
  --base-data moconvq_base.data \
  --text-encoder t5 \
  --text-model t5-large \
  --max-text-length 256 \
  --max-length 120 \
  --context-size 26 \
  --chunk-size 25 \
  --gpu 0 \
  --seed 0
```

生成 BVH 后建议同时跑工程指标脚本，记录早停、时长、root 轨迹和重复风险。它不是论文级 FID/R-precision 的替代品，但能避免只凭视频主观判断：

```bash
python Script/stage1/evaluate_bvh_metrics.py \
  'stage1_artifacts/generated_bvh_compare/<run_id>/*.bvh' \
  --sample-stride 6 \
  --lags 5,10,20,30 \
  --expected-min-frames 1200 \
  --output stage1_artifacts/generated_bvh_compare/<run_id>/summary_metrics_script.json
```

MoConVQ 论文的正式 Text2Motion 量化指标是 HumanML3D test set 上的 FID 和 R-precision，依赖兼容的 HumanML3D/SMPL motion feature extractor。当前本地已经具备 T2M evaluator source/checkpoint/glove assets，因此 Stage1 可以输出 approximate FID/R-precision/matching-score；但生成结果需要先经过 MoConVQ BVH -> approximate HumanML3D 22-joint feature adapter，长序列还会被 evaluator 截断到最多 196 帧，所以报告中必须标注为 approximate evaluator-adapter route，而不是论文原生 SMPL 评估。

当前 readiness 检查已显式支持 T2M-GPT/text-to-motion evaluator 布局。需要的最小源码和资源是：

```text
models/evaluator_wrapper.py
models/modules.py
utils/eval_trans.py
utils/word_vectorizer.py
options/get_eval_option.py
checkpoints/t2m/text_mot_match/model/finest.tar
checkpoints/t2m/text_mot_match/opt.txt
glove/our_vab_data.npy
glove/our_vab_words.pkl
glove/our_vab_idx.pkl
```

为避免 evaluator 资产准备步骤散落在终端历史中，当前提供了 helper：

```text
Script/stage1/prepare_t2m_evaluator_assets.py
tests/test_stage1_prepare_t2m_evaluator_assets.py
```

它支持：

```bash
# 检查一个 evaluator root 是否具备所需源码和资源
python Script/stage1/prepare_t2m_evaluator_assets.py \
  --root /tmp/stage1_t2m_evaluator_assets

# 从已 clone 的 T2M-GPT 源码目录复制 evaluator 所需源码
python Script/stage1/prepare_t2m_evaluator_assets.py \
  --root /tmp/stage1_t2m_evaluator_assets \
  --source-root /tmp/T2M-GPT-stage1-inspect \
  --copy-sources

# 打印官方 Google Drive 下载、解压、readiness 检查命令
python Script/stage1/prepare_t2m_evaluator_assets.py \
  --root /tmp/stage1_t2m_evaluator_assets \
  --print-download-commands
```

当前 `/tmp/stage1_t2m_evaluator_assets` 已经包含 evaluator source files、checkpoint 和 glove assets；`check_evaluation_readiness.py` 对该目录返回 `paper_metrics_ready=true`。历史下载记录：无代理时 Google Drive 网络不可达；需要下载时应使用 `http_proxy=http://127.0.0.1:7898` 和 `https_proxy=http://127.0.0.1:7898`。

为了让 evaluator assets 一到就能跑正式指标，当前新增了 Stage1 paper-metric runner：

```text
Script/stage1/evaluate_t2m_paper_metrics.py
tests/test_stage1_t2m_paper_metrics.py
```

它的路线是：

```text
generated MoConVQ/base.bvh
  -> bvh_to_humanml3d_features.py 近似转成 HumanML3D 263-d feature
  -> 使用 HumanML3D Mean.npy/Std.npy 归一化
  -> T2M evaluator motion/text embeddings
  -> FID vs HumanML3D reference split
  -> R-precision / matching score for generated prompt-motion pairs
```

在 assets 缺失时可以先跑 check-only，确认 prompt suite、BVH 输入和缺失项：

```bash
python Script/stage1/evaluate_t2m_paper_metrics.py \
  /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/bvh/*.bvh \
  --prompts /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/prompts.tsv \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --evaluator-root /tmp/stage1_t2m_evaluator_assets \
  --output-dir /tmp/stage1_t2m_paper_metrics_check_20260613 \
  --summary /tmp/stage1_t2m_paper_metrics_check_20260613/summary_after_sources.json \
  --check-only
```

当前 check-only 结果：

```text
sources_ready = true
assets_ready  = false
planned prompts:
  circle_crouch_stand
  sidestep_kick_turn
  walk_jump_dance
  walk_turn_wave
planned models:
  baseline_top_p
  finetuned_top_p
missing assets:
  checkpoints/t2m/text_mot_match/model/finest.tar
  checkpoints/t2m/text_mot_match/opt.txt
  glove/our_vab_data.npy
  glove/our_vab_words.pkl
  glove/our_vab_idx.pkl
```

正式运行去掉 `--check-only` 即可；如果 assets 仍缺失，脚本会退出并写出缺项 summary，不会输出伪 FID/R-precision。注意：这个 runner 仍属于 approximate evaluator-adapter route，因为生成 BVH 先经过 MoConVQ/base.bvh 到 HumanML3D skeleton 的近似转换，并且 T2M evaluator 默认最多消费 196 帧，长序列会按 `--max-motion-length` 截断。

当前已经新增一个 MoConVQ BVH 到 HumanML3D feature 的近似 adapter：

```text
Script/stage1/bvh_to_humanml3d_features.py
tests/test_stage1_bvh_to_humanml3d_features.py
Script/stage1/calibrate_bvh_to_humanml3d_adapter.py
tests/test_stage1_bvh_to_humanml3d_calibration.py
```

它的路径是：

```text
generated MoConVQ/base.bvh
  -> BVH forward kinematics
  -> approximate MoConVQ 20-body positions to HumanML3D 22-joint positions
  -> HumanML3D scripts/generate_motion_representation.py::process_file()
  -> 263-d new_joint_vecs feature
```

Smoke test 已能把当前 best run 的 generated BVH 转成 263-d feature：

```text
input:
  /tmp/stage1_long_fixed_bvh_native_200_head_epoch5_compare_20260613/bvh/circle_crouch_stand__finetuned_top_p.bvh

output:
  /tmp/stage1_bvh_to_humanml3d_smoke/new_joint_vecs/circle_crouch_stand__finetuned_top_p.npy

source frames:    1656 at ~120 FPS
resampled joints: 276 at 20 FPS
feature shape:    275 x 263
```

这个 adapter 消除了“完全没有 BVH -> HumanML3D feature 转换”的硬缺口，但它仍然是骨架近似：直接对应的关节按 BVH node 名复制，HumanML3D 的 spine/neck/head 中间关节由 MoConVQ torso/head chain 插值得到。2026-06-13 已增加 roundtrip calibration：

```bash
python Script/stage1/calibrate_bvh_to_humanml3d_adapter.py \
  --humanml-root /home/chenjie/cc/robotics/HumanML3D \
  --split test \
  --limit 20 \
  --seed 13 \
  --output-dir /tmp/stage1_bvh_to_humanml3d_calibration_test20_20260613 \
  --summary /tmp/stage1_bvh_to_humanml3d_calibration_test20_20260613/summary.json
```

校准方式是把已知 HumanML3D `new_joints/new_joint_vecs` 样本导出成 MoConVQ/base.bvh，再通过当前 adapter 转回 HumanML3D 22-joint 和 263-d feature，最后和原始 HumanML3D 数据比较。20 个 test split 样本结果：

```text
avg feature MAE:      0.0722
max feature MAE:      0.1009
avg feature z RMSE:   0.5694
avg feature z p95 abs:1.3696
max feature z p95 abs:1.8286
avg joint MPJPE:      0.0796
max joint MPJPE:      0.0841
root position error:  ~4.8e-7
```

解释：root 轨迹几乎无误差，说明 BVH root translation 往返是稳定的；约 `0.08` 的平均关节误差和标准化 feature p95 约 `1.37` 主要来自 MoConVQ/base.bvh 与 HumanML3D 骨架不完全同构，尤其是 spine/neck/head 近似。因此这个 adapter 可以用于“生成 BVH 能否转成 HumanML3D feature”的工程检查和 evaluator 接入前校准，但即使 evaluator assets 补齐，也必须在报告中把 FID/R-precision 标为 approximate/evaluator-adapter route，不能把它当成原生 HumanML3D/SMPL 无偏评估。

为了避免每次实验只比较单一路线，当前推荐把最终 Stage1 对比实验收敛到统一 suite：

```bash
python Script/stage1/run_stage1_model_suite.py \
  --run-id suite_filtered_stage1 \
  --finetuned-checkpoint stage1_artifacts/checkpoints/filtered_stage1_20260612_181802/best_val.pth \
  --backup-cache stage1_artifacts/gpt_cache_filtered_cache_stage1_20260612_174908/train_cache.pt \
  --base-data moconvq_base.data \
  --motion-dataset simple_motion_data.h5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --max-length 120 \
  --context-size 30 \
  --chunk-size 20 \
  --top-p 0.95 \
  --progress-scale 0.5 \
  --expected-min-frames 1200 \
  --gpu 0
```

这个脚本会在同一个 `run_id` 下生成：

```text
baseline_top_p BVH
finetuned_top_p BVH
backup_retrieval BVH
optional backup_llm BVH
summary_metrics.json
suite_summary.json
```

默认 prompt set 覆盖 walk-turn-wave、circle-crouch-stand、walk-jump-dance、sidestep-kick-turn 四类多阶段长文本。也可以用 `--prompts <tsv>` 指定正式报告 prompt。若要评估真实 LLM in-context token planning，把外部 LLM 的 JSON response 保存成文件，并用 `--llm-response-map responses.json` 提供 prompt 名到 response 文件的映射；suite 会调用 `llm_token_planning.py validate` 和 `decode-bvh` 得到 `backup_llm` BVH。

如果从临时 worktree 或仓库外部运行，建议显式传 `--motion-dataset /abs/path/to/simple_motion_data.h5`。MoConVQ 默认 config 里的 `motion_dataset` 是相对路径，显式传入可以避免 decode backup BVH 时依赖当前工作目录。retrieval-only backup 默认会截断超长连续重复 RVQ tuple run，并把 `repeat_repairs` 记录到 `retrieval_validation.json`；这是 decoder 可运行性修复，不应当被解释为语义成功。

长文本动作推荐使用分段生成，让每一段 motion 使用对应局部 caption，而不是每个 rolling chunk 都看同一个完整长文本。分段生成会把上一段末尾的 latent 作为下一段开头的上下文，从而保留动作连续性，同时显式告诉模型当前执行的是哪一段文本。默认 `auto` 会在检测到多段文本时走这条路径；当前兼容模式是精确按 `--segment-joiner` 字符串拆分，默认 joiner 是英文小写、两侧带空格的 `" then "`。如果 prompt 中没有这个精确字符串，`auto` 会退回 rolling；如果要按中文“然后”或其他分隔符拆分，需要显式传 `--segment-joiner`。为了严格匹配 HumanML3D 合成数据里的 clip-caption 边界，推荐在正式 prompt TSV 中增加第三列 JSON segment list，或直接给生成脚本传 `--segments-json`；这样即使某个原始 caption 内部有 `then`，也不会被误拆。如果没有显式传 `--segment-lengths` 或 `--segment-length`，脚本会把 `--max-length` 自动分配到各文本段：

```bash
python Script/stage1/generate_long_motion.py \
  --checkpoint stage1_artifacts/checkpoints/real_stage1_fixed/best_val.pth \
  --text "a person walks forward then turns around then waves both arms" \
  --output-bvh stage1_artifacts/generated/demo_segmented.bvh \
  --base-data moconvq_base.data \
  --text-encoder t5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --generation-mode auto \
  --segment-joiner " then " \
  --segment-lengths 25,25,20 \
  --context-size 26 \
  --chunk-size 25 \
  --gpu 0 \
  --seed 0
```

显式 segment 示例：

```bash
python Script/stage1/generate_long_motion.py \
  --checkpoint /tmp/stage1_segment_aligned_bvh_native_200_head_seed13_5ep_20260614/checkpoint_epoch_3.pth \
  --text "a person reaches down then walks forward" \
  --segments-json '["a person reaches down then picks something up", "a person walks forward"]' \
  --segment-lengths 35,40 \
  --output-bvh stage1_artifacts/generated/demo_explicit_segments.bvh \
  --base-data moconvq_base.data \
  --text-encoder t5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --generation-mode auto \
  --context-size 30 \
  --chunk-size 20 \
  --gpu 0 \
  --seed 0
```

TSV 第三/第四列格式：

```text
sample_name<TAB>long prompt text<TAB>["segment caption 1", "segment caption 2"]<TAB>[35, 40]
```

2026-06-13 的 best checkpoint 上做过一次只改变 generation mode 的 ablation，说明当前 `" then "` 分段不是单纯的代码偏好，而是对长 prompt 指标有实际帮助。相同 prompt、seed、top-p、checkpoint 下：

| generation mode | model | avg frames | early stop rate | avg pose velocity | avg pose variance |
| --- | --- | ---: | ---: | ---: | ---: |
| auto / then-segmented | baseline | 1062 | 0.75 | 14.052 | 141.194 |
| auto / then-segmented | head fine-tuned | 1194 | 0.50 | 19.133 | 190.011 |
| forced rolling | baseline | 768 | 0.75 | 37.226 | 380.162 |
| forced rolling | head fine-tuned | 930 | 0.50 | 39.383 | 374.359 |

Per-prompt frame counts:

| prompt | segmented baseline | segmented fine-tuned | rolling baseline | rolling fine-tuned |
| --- | ---: | ---: | ---: | ---: |
| walk_turn_wave | 816 | 864 | 432 | 432 |
| circle_crouch_stand | 1176 | 1656 | 816 | 1416 |
| walk_jump_dance | 1392 | 1392 | 1416 | 1416 |
| sidestep_kick_turn | 864 | 864 | 408 | 456 |

因此当前 Stage1 长文本生成应保留 `--generation-mode auto` 或显式 `--generation-mode segmented`。为了复现 plain-`" then "` Val8 最好结果，可以继续使用精确的 `" then "` joiner；但如果 prompt 来自 HumanML3D clip captions，或需要严格匹配训练 clip 边界，应优先使用 TSV 第三列 / `--segments-json` 显式传入 segment list，避免句内 `then` 被误拆。

### 6.9 数据问题修复记录

2026-05-29 的评估显示，旧模型虽然能生成更长 BVH，但后半段容易重复或循环。进一步检查发现主要问题在数据侧：

- 旧合成集的 transition score 在候选 clip 还没有平移/yaw 对齐前计算，因此大量本来可对齐的 clip 被误判为差 transition，旧 train manifest 中 `forced_transitions=1261/2016`，约 62.5%。
- 旧 cache 按整条长序列滑窗，导致约 85% 的训练 window 跨越 clip 边界，约 67% 的 train window 碰到 forced transition。这会把不连续拼接处也当成 GPT 的监督目标。
- 旧 cache 即使使用 `caption_mode=window`，跨边界窗口仍会把多个局部 caption 拼在一起，模型无法明确知道当前 50-token 窗口应该执行哪一段动作。

当前代码已修复：

- `synthesize_long_humanml3d.py` 在 transition scoring 前先把候选 clip 对齐到前一段末帧，再评价 root velocity、yaw、foot height/velocity。
- 同一条合成长序列内优先避免重复使用同一个 HumanML3D sample，减少训练数据里天然循环同一动作的情况。
- `real_moconvq_cache.py` 默认 `--window-policy clip`，即每个训练 window 只来自单个 clip 内部；跨 clip 边界不再直接喂给 GPT。
- `real_moconvq_cache.py` 支持 `--forced-transition-margin`，如果 manifest 中仍有 forced transition，可以裁掉边界两侧若干 latent token。
- `real_moconvq_cache.py` 默认 `--rotation-calibration rest`，用 MoConVQ world rest pose 校准 HumanML3D 位置重定向得到的 body quaternion。前 20 条 fixed train 序列的 observation p99 `|z|` 从未校准的 `19.39` 降到 `5.06`，说明原先存在明显静态 body-frame mismatch。

因此，旧目录里的 cache 不建议继续训练：

```text
stage1_artifacts/gpt_cache/train_cache.pt
stage1_artifacts/gpt_cache/val_cache.pt
```

建议重建到新目录，避免和旧实验混用：

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
  --output-dir stage1_artifacts/long_humanml3d_fixed/train

python Script/stage1/synthesize_long_humanml3d.py \
  --humanml-root ../HumanML3D/HumanML3D \
  --split val \
  --num-sequences 200 \
  --min-clips 2 \
  --max-clips 4 \
  --seed 1 \
  --candidate-pool 256 \
  --transition-max-score 0.35 \
  --blend-frames 5 \
  --caption-joiner " then " \
  --output-dir stage1_artifacts/long_humanml3d_fixed/val
```

构建新 cache：

```bash
python Script/stage1/build_real_moconvq_gpt_cache.py \
  --long-h5 stage1_artifacts/long_humanml3d_fixed/train/long_sequences.h5 \
  --manifest stage1_artifacts/long_humanml3d_fixed/train/manifest.jsonl \
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
  --output stage1_artifacts/gpt_cache_fixed/train_cache.pt \
  --failure-log stage1_artifacts/gpt_cache_fixed/train_failures.jsonl

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
  --output stage1_artifacts/gpt_cache_fixed/val_cache.pt \
  --failure-log stage1_artifacts/gpt_cache_fixed/val_failures.jsonl
```

新训练请使用重新构建且 config 中包含 `rotation_calibration=rest` 的 cache。不要复用旧 cache。

### 6.10 Fixed Dataset 实验结果

2026-05-29 已使用上述 fixed 数据链路重新跑完一次 20 epoch 真实实验，run id：

```text
fixed_dataset_stage1_20260529_135401
```

合成阶段现在会同时写两类日志：

```text
stage1_artifacts/long_humanml3d_fixed/train/synthesize.log
stage1_artifacts/long_humanml3d_fixed/train/synthesize_progress.jsonl
stage1_artifacts/long_humanml3d_fixed/val/synthesize.log
stage1_artifacts/long_humanml3d_fixed/val/synthesize_progress.jsonl
```

`synthesize.log` 适合直接看终端式摘要，`synthesize_progress.jsonl` 适合后续脚本统计。事件包括 `start`、`sequence_written`、`skip_sequence` 和 `summary`。本次 fixed 合成统计如下：

```text
train:
  sequences: 1000
  transitions: 1945
  avg_clips: 2.945
  avg_frames: 416.593
  forced_transitions: 0
  duplicate_sequences: 0
  failed/skipped attempts: 3
  accepted transition score mean/p50/p95/max:
    0.020231 / 0.007033 / 0.075246 / 0.297312

val:
  sequences: 200
  transitions: 398
  avg_clips: 2.990
  avg_frames: 410.200
  forced_transitions: 0
  duplicate_sequences: 0
  failed/skipped attempts: 1
  accepted transition score mean/p50/p95/max:
    0.018912 / 0.006431 / 0.068812 / 0.279766
```

这说明当前拼接逻辑作为“过滤后的实验数据构造”已经比旧版干净很多：没有 forced transition，也没有同一长序列内重复使用同一个 sample。被拒绝的少量样本主要是 foot height / foot velocity 不连续，说明阈值确实在过滤局部接触不自然的边界。但这不等价于拼接逻辑已经能生成自然转场。它只是一个局部边界筛选器，不会合成真正的过渡动作，因此训练 GPT 时仍然默认使用 `--window-policy clip`，不要把跨 clip hard join 当成监督目标。

fixed cache：

```text
stage1_artifacts/gpt_cache_fixed/train_cache.pt
  latents:       (2958, 50, 768)
  indices:       (2958, 50, 4)
  text_features: (2958, 256, 1024)
  text_masks:    (2958, 256)
  caption_mode:  window
  window_policy: clip

stage1_artifacts/gpt_cache_fixed/val_cache.pt
  latents:       (598, 50, 768)
  indices:       (598, 50, 4)
  text_features: (598, 256, 1024)
  text_masks:    (598, 256)
  caption_mode:  window
  window_policy: clip
```

20 epoch 微调输出：

```text
checkpoint:
  stage1_artifacts/checkpoints/fixed_dataset_stage1_20260529_135401/

log:
  stage1_artifacts/logs/fixed_dataset_stage1_20260529_135401.log

training curves:
  stage1_artifacts/figures/fixed_dataset_stage1_20260529_135401/loss_curve.png
  stage1_artifacts/figures/fixed_dataset_stage1_20260529_135401/loss_accuracy_curve.png
  stage1_artifacts/figures/fixed_dataset_stage1_20260529_135401/loss_accuracy_curve_data.csv
```

训练指标：

```text
epoch 1:
  train loss: 3.7948
  val loss:   2.7816
  train acc:  0.2505
  val acc:    0.3485

epoch 20 / best val:
  train loss: 1.6198
  val loss:   1.7807
  train acc:  0.5569
  val acc:    0.5236
```

对比生成已经导出 BVH 和 MP4：

```text
BVH:
  stage1_artifacts/generated_bvh_compare/fixed_dataset_stage1_20260529_135401/

MP4:
  stage1_artifacts/generated_bvh_compare/fixed_dataset_stage1_20260529_135401_mp4/
```

Prompt 包括：

```text
walk_turn_wave        a person walks forward then turns around then waves both arms
circle_crouch_stand   a person walks in a circle then crouches down then stands up
walk_jump_dance       a person walks forward then jumps then dances
sidestep_kick_turn    a person sidesteps to the left then kicks with the right foot then turns around
```

Baseline 生成时仍然容易提前输出 end token，因此对比文件名使用 `baseline_early`。finetuned best 对 4 个 prompt 都生成了 2160-frame BVH，对应 MP4 约 18 秒；baseline_early 视频长度约 4.0-7.0 秒。这个结果说明 fixed 训练改善了 baseline 早停问题，但“生成更长”不等于“动作语义更好”。最终结论仍需人工检查 MP4 中是否存在后半段重复、动作语义错位、脚滑和姿态异常。

2026-05-29 后续排查确认：该 run 的 checkpoint 仍不能作为有效模型。虽然它在 fixed cache 上的 token loss 明显低于 baseline，但旧训练上下文 latent 与推理 latent 空间不一致，并且全量微调覆盖了 baseline 运动先验，导致视频质量比 baseline 差。修复后的代码已改为从 RVQ indices 重建 GPT 推理同构的 4-layer latent，并增加 `--train-scope base_head/head`。下一轮对比应重新训练，不要继续使用：

```text
stage1_artifacts/checkpoints/fixed_dataset_stage1_20260529_135401/best_val.pth
```

如果 fixed GPT 仍然出现循环或局部动作重复，优先排查顺序：

1. 查看 MP4，区分是 GPT token 重复，还是 HumanML3D -> MoConVQ retarget/decode 后动作质量差。
2. 统计生成 RVQ token 的重复率、end token 位置和 per-depth 分布。
3. 使用 segmented generation，让每段文本单独编码并继承上一段 motion context，而不是全程用同一个长 prompt rolling。
4. 若需要真正自然跨段过渡，增加 transition retrieval/library，而不是把 hard join 边界交给 GPT 学。
5. 如果拼接路线仍不稳定，转向 `STAGE1_BACKUP_PLAN.md` 中的 LLM in-context motion token planning。


## 7. 当前状态一句话总结

Stage1 的旧 fixed dataset 工程链路已经完整跑通，但旧 finetuned checkpoint 已判定无效：它改善了早停，却因训练/推理 latent 空间不一致、全量微调覆盖先验、以及 HumanML3D retarget/cache 质量不足导致视频质量差。当前代码已修复训练上下文 latent 重建，并把 HumanML3D 主线推进到 `long_sequences.h5` -> `joints_ik` BVH export -> MoConVQ-native character retarget -> per-file quality filter -> deterministic accepted train/val split -> accepted-only GPT cache。

当前推荐的 Stage1 engineering/metric 结果是 2026-06-14 artifact label 的 segment-aligned native-retarget run：73 train / 18 val long sequences 形成 476 train windows 和 117 val windows。head-only 5 epoch 训练能让 loss 下降并在 plain-`" then "` protocol 下取得 partial improvement，但严格按 HumanML3D clip boundary 推理时仍暴露出容量不足。新的 `base_head` 3 epoch 微调把 val loss 从 `14.912 -> 10.731`，并在 explicit segment + scaled segment lengths 的 Held-out Val8 上取得当前最好严格协议结果：平均长度 `1182 -> 1197`，早停率持平 `0.50`，root path `1.682 -> 2.074`，approximate FID `20.279 -> 14.985`，R-precision@1 持平 `0.375`，R-precision@2 `0.500 -> 0.750`，R-precision@3 `0.625 -> 0.875`，matching score `4.813 -> 4.384`。完整 Val18 strict prompt 复现中，`checkpoint_epoch_3` 的 approximate FID `13.726 -> 12.633`、R@1 `0.222 -> 0.278`、matching score `4.880 -> 4.609`、平均长度和 root path 继续优于 baseline，但 R@2 下降、R@3 和早停率持平；`checkpoint_epoch_2` 更保守，approximate FID `13.726 -> 13.060`、R@3 `0.444 -> 0.500`、matching score `4.880 -> 4.789`，同时 R@1/R@2 持平。降低到 `top_p=0.90,temp=0.8` 是负结果，finetuned 反而低于 baseline。contact sheet 未见空帧、整体倒置或爆炸，`train_000057` 和 `train_000077` 已生成 side-by-side MP4。

因此当前结论是：长 HumanML3D 合成可用；旧 hand-written HumanML3D-to-state cache 不适合正式结论；BVH-to-character native retarget 是目前真正 work 的主线；训练和推理必须使用一致的 segment 语义。默认 `" then "` 分段可作为历史对照，但会误拆含句内 `then` 的 HumanML3D caption；正式长文本评估应优先使用第三列 JSON segments 和第四列 segment lengths。`base_head` 结果说明，修复数据映射、segment 边界和段长后，适度扩大微调范围可以在 approximate paper-style FID/R-precision 上超过 baseline 的主要指标，但仍不是完全解决：不同 checkpoint 在 R@1/R@2/R@3 之间有 trade-off，pose variance 和重复率也仍需人工视频审计。论文指标相关资产已经具备，但 MoConVQ BVH 到 HumanML3D 263-d feature 的近似 adapter 带有骨架适配误差，且 evaluator 会截断长序列，因此当前 FID/R-precision 必须标为 approximate evaluator-adapter route，不能替代原生 HumanML3D/SMPL 无偏评估。
