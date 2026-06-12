# Stage1 Backup Plan: LLM In-Context Motion Token Planning

本文档是 Stage1 的备选实验路线。主线仍然是使用 HumanML3D 合成长序列，转换成 MoConVQ `latent_vq + RVQ indices + T5 text feature` cache，并微调仓库内的 `Text2Motion_Transformer`。如果后续确认“HumanML3D clip 直接拼接后 retarget 到 MoConVQ”带来的运动边界质量不足，或者微调后的 GPT 仍然在长动作上出现重复/语义错位，则采用本 backup plan 作为保底路线。

## 1. 当前 Cache 是否能复用

结论分两层：

- 从文件 schema 看，旧 cache 的 `latents/indices/text_features/text_masks` 能被修复后的 `train_real_text_gpt.py` 读取。
- 从实验有效性看，旧 cache 不建议继续作为主实验数据，因为它来自旧合成集，包含大量 forced transition 和跨 clip 边界窗口。

因此真实实验请使用 fixed cache：

```text
stage1_artifacts/gpt_cache_fixed/train_cache.pt
  latents:       (2958, 50, 768), float32
  indices:       (2958, 50, 4), int64
  text_features: (2958, 256, 1024), float32
  text_masks:    (2958, 256), bool
  caption_mode:  window
  window_policy: clip
  valid indices: 0..511, padding index 513

stage1_artifacts/gpt_cache_fixed/val_cache.pt
  latents:       (598, 50, 768), float32
  indices:       (598, 50, 4), int64
  text_features: (598, 256, 1024), float32
  text_masks:    (598, 256), bool
  caption_mode:  window
  window_policy: clip
  valid indices: 0..511, padding index 513
```

原因：

- 最近的关键代码修改是训练目标对齐方式，而不是 cache schema。
- 旧训练错误包括训练/推理 latent 空间不一致：cache 的 `latent_vq` 是 MoConVQ encoder 8 层 RVQ 总和，而 Text2Motion GPT 推理只采样前 4 层 RVQ token 并求和作为下一步上下文。修复后训练脚本在 batch 内从 cache 的 `indices` 动态重建前 4 层 `context_latent`，target 仍然是 cache 中原有的 `indices`。
- fixed cache 是 `caption_mode=window + window_policy=clip`，每个 50-token motion window 对应单个 clip 的局部 caption，符合修复后的长文本定位策略。
- fixed cache 的合成源没有 forced transition，且同一长序列内没有重复 sample；这比旧 cache 更适合作为主实验输入。

只有以下情况才需要重建 cache：

- 重新合成了更干净的 `long_sequences.h5`；
- 修改了 HumanML3D 到 MoConVQ state/observation 的 retarget 逻辑；
- 修改了 `window-size/window-stride/rvq-depth/max-text-length`；
- 从 `caption_mode=window` 切回或改成其他文本对齐策略；
- 换了 T5 模型或 text feature 生成方式；
- 发现当前合成数据本身边界不自然，需要重新过滤 forced transitions。

可直接重训的命令：

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd /home/chenjie/cc/robotics/MoConVQ

python Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache_fixed/train_cache.pt \
  --val-cache stage1_artifacts/gpt_cache_fixed/val_cache.pt \
  --init-checkpoint text_generation_GPT.pth \
  --base-data moconvq_base.data \
  --output-dir stage1_artifacts/checkpoints/fixed_dataset_stage1_next \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --train-scope base_head \
  --gpu 0 \
  --seed 0 \
  --save-every 1 \
  --num-workers 4
```

## 2. 为什么需要 Backup Plan

当前主线的风险点有两个：

1. 数据风险：HumanML3D 的短 clip 即使经过根位置/yaw 对齐，直接拼成长序列仍可能在速度、脚部接触、身体姿态和语义边界上不自然。这样的数据会让 GPT 学到“局部合理但跨段重复/断裂”的模式。
2. 模型风险：MoConGPT 的 motion context 长度有限。论文中也说明 MoConGPT 接收的 motion sequence 长度受限，长序列需要 sliding window 方式续生成。即使训练目标修复，单纯靠一个长 prompt 和 rolling context 也不能保证模型理解“现在执行到第几段文本”。

因此 backup plan 不再把“拼接后的长动作”作为唯一监督信号，而是把 MoConVQ 的离散 token 作为动作技能库，让大模型负责高层动作分解、检索和重组，MoConVQ 负责低层物理动作解码。

## 3. 论文依据

MoConVQ 论文中给出了两条与 backup plan 直接相关的依据：

- Text2Motion-MoConGPT：论文用 HumanML3D 训练 text-conditioned MoConGPT，使其从自然语言生成 MoConVQ motion indices，再由 physics-based decoder/controller 生成动作。
- LLM Integration：论文额外展示了用 ChatGPT/Claude 这类大模型做 in-context learning。具体做法是给 LLM 大量“文本描述 + MoConVQ index sequence”的样例，让 LLM 在不微调的情况下学习 index 表示，并对新文本输出新的 index 序列。论文强调 LLM 可以重组已有动作片段，例如把 walk、kick、dance 等动作 token 拼成一个长动作。

这说明 backup plan 是论文已有思路的工程化复现/缩小版：不需要先训练一个新的长序列 GPT，而是通过 example retrieval + LLM in-context planning 生成 MoConVQ token，再使用现有 MoConVQ decoder。

## 4. Backup Plan 总体目标

建立一条独立于“长序列拼接训练质量”的保底链路：

```text
HumanML3D / existing cache examples
  -> build text-index example bank
  -> retrieve examples for each sub-action
  -> prompt LLM to output planned MoConVQ token sequence
  -> validate / repair token sequence
  -> convert indices to latent via MoConVQ codebooks
  -> decode with MoConVQ decoder/controller
  -> BVH output
```

核心思路：

- 不让 LLM 直接生成连续 323-d observation 或 768-d latent。
- 让 LLM 只处理紧凑的离散 motion tokens。
- 只在较高层做动作顺序、重复次数、组合关系规划。
- token 合法性、长度、去重、平滑和 BVH 生成仍由本地脚本控制。

## 5. 数据准备

### 5.1 Example Bank

从现有 `train_cache.pt` 或从短 clip cache 中抽取样例，建立 `stage1_artifacts/llm_backup/example_bank.jsonl`。

每行格式：

```json
{
  "example_id": "train_000123_w000",
  "caption": "a person walks forward",
  "sequence_id": "train_000123",
  "window_range": [0, 50],
  "indices": [[297, 12, 43, 88], [471, 18, 40, 91]],
  "indices_depth0": [297, 471, 246, 463, 463],
  "source": "train_cache.pt"
}
```

第一版可以只让 LLM 生成 `depth0` token，贴近论文里为了简化而只考虑 VQ index 的设置。生成 BVH 时有两种实现路线：

- 路线 A：只输出 depth0，其他 RVQ depth 用 0 或检索样例中的对应 depth 补齐。优点是 prompt 简短；缺点是动作细节可能差。
- 路线 B：让 LLM 输出 4-depth token tuple，例如 `[297,12,43,88]`。优点是能直接还原 RVQ latent；缺点是 prompt 更长，LLM 更容易格式出错。

建议第一版用路线 B 做真实解码，因为当前 MoConVQ 的 latent 是 4 层 RVQ embedding 求和；同时在 prompt 中严格要求 JSON array，便于解析和检查。

### 5.2 Retrieval

对输入长文本先分段：

```text
"walk forward for a long time then kick then dance"
-> ["walk forward for a long time", "kick", "dance"]
```

每段从 example bank 检索 top-k 样例：

- 简单版本：使用 T5 text feature 或 sentence-transformer embedding 做 cosine similarity；
- 无额外依赖版本：用 caption token overlap / BM25；
- 调试版本：手动指定 examples。

每段至少取 3-5 个样例，并优先选择短而动作明确的 caption。

## 6. LLM Prompt 设计

Prompt 要避免让 LLM 把单个 token 当成完整动作，也要避免直接复制一个长样例。建议结构：

```text
You are controlling a simulated humanoid through MoConVQ RVQ token tuples.
Each motion is a sequence of 4-integer tuples [d0,d1,d2,d3].
Each integer must be in [0, 511].
Do not output explanations in the final answer.
Do not invent non-integer tokens.
A short action should contain multiple tuples, not a single tuple.
For "for a long time", repeat a short stable locomotion subsequence.
For compound prompts, concatenate sub-action sequences in order.

Examples:
Caption: a person walks forward
Tokens: [[...], [...], ...]

Caption: a person kicks with the right foot
Tokens: [[...], [...], ...]

Question:
Generate tokens for: a person walks forward for a long time then kicks then dances.

Return only JSON:
{"tokens": [[d0,d1,d2,d3], ...]}
```

对于长文本，推荐分段调用 LLM：

1. 第一次让 LLM 输出 plan：

```json
{
  "segments": [
    {"text": "walk forward for a long time", "target_tokens": 30},
    {"text": "kick", "target_tokens": 10},
    {"text": "dance", "target_tokens": 30}
  ]
}
```

2. 再逐段检索 examples 并生成 token。
3. 最后本地脚本拼接 token，并做合法性检查。

这样可以显式解决“模型不知道执行到第几段文本”的问题。

## 7. Token 校验与修复

LLM 输出不能直接信任，必须经过本地 validator：

- 必须是 JSON；
- `tokens` 必须是 list；
- 每个 token 必须是 4 个整数；
- 每个整数必须在 `[0, 511]`；
- 总长度必须在指定范围内；
- 相邻完全相同 tuple 的连续重复不能超过阈值，例如 5；
- 如果非法 token 少，可以用最近合法 token 或检索样例 token 替换；
- 如果整体非法，重新 prompt，加入错误原因。

可记录：

```text
stage1_artifacts/llm_backup/runs/<run_id>/prompt.txt
stage1_artifacts/llm_backup/runs/<run_id>/raw_response.txt
stage1_artifacts/llm_backup/runs/<run_id>/tokens.json
stage1_artifacts/llm_backup/runs/<run_id>/validation.json
stage1_artifacts/llm_backup/runs/<run_id>/output.bvh
```

## 8. Token 到 BVH

给定 4-depth RVQ token sequence 后，可以直接使用 `moconvq_base.data` 中的 RVQ codebook embedding 重建 latent：

```text
latent[t] = codebook0[idx[t,0]]
          + codebook1[idx[t,1]]
          + codebook2[idx[t,2]]
          + codebook3[idx[t,3]]
```

然后复用现有生成路径：

```text
latent sequence
  -> agent.posterior.decoder.decode_dynamic()
  -> agent.act_tracking()
  -> CharacterTOBVH
  -> output.bvh
```

这部分可以复用 `generate_long_motion.py` 中的 BVH 写出逻辑，只是把 GPT sampling 换成 LLM token input。

## 9. 对照实验

Backup plan 不是为了替代主线，而是作为对照和保底。建议至少比较三组：

```text
Baseline:
  text_generation_GPT.pth + generate_long_motion.py

Main:
  修复后的 train_real_text_gpt.py checkpoint + segmented/auto generation

Backup:
  LLM-generated MoConVQ tokens + MoConVQ decoder/controller
```

Prompt 组：

```text
1. a person walks forward for a long time then kicks
2. a person walks forward then turns around then waves both arms
3. a person circles around then crouches down then stands up
4. a person walks forward then jumps then dances
5. a person walks in a square trajectory
```

评估指标：

- 是否按顺序完成每个子动作；
- 生成长度是否满足要求；
- 后半段是否 token collapse；
- 相邻重复 token tuple 比例；
- root drift 是否异常；
- 是否明显脚滑；
- 人工视觉评分。

## 10. 最小实现任务

### Task A: 导出 example bank

新增脚本：

```text
Script/stage1/export_llm_motion_examples.py
```

输入：

```text
--cache stage1_artifacts/gpt_cache/train_cache.pt
--output stage1_artifacts/llm_backup/example_bank.jsonl
--max-examples 1600
--max-tokens-per-example 50
```

输出 JSONL，每行包含 caption、window_range、indices。

### Task B: 检索 examples

新增脚本：

```text
Script/stage1/retrieve_llm_motion_examples.py
```

输入一个 query segment，从 example bank 中取 top-k captions。第一版可用 BM25/token overlap，不引入新依赖。

### Task C: LLM prompt runner

新增脚本：

```text
Script/stage1/run_llm_motion_planner.py
```

第一版可以不接 API，只生成 prompt 文件，由人工复制到 ChatGPT/Claude/其他大模型，再把 response 保存回来。

原因：课程环境和网络/API key 不稳定时，手工模式最稳，也符合论文中上传 examples 文件给 Claude 的实验方式。

### Task D: token validator

新增脚本：

```text
Script/stage1/validate_llm_motion_tokens.py
```

检查 JSON、shape、范围、重复率，并输出清洗后的 `tokens.json`。

### Task E: tokens to BVH

新增脚本：

```text
Script/stage1/generate_bvh_from_rvq_tokens.py
```

输入：

```text
--tokens stage1_artifacts/llm_backup/runs/<run_id>/tokens.json
--base-data moconvq_base.data
--output-bvh stage1_artifacts/llm_backup/runs/<run_id>/output.bvh
```

功能：从 codebook embedding 重建 latent，复用 MoConVQ decoder/controller 写 BVH。

## 11. 预期优缺点

优点：

- 不依赖拼接长动作数据的训练质量；
- 更适合处理长文本、抽象任务和显式动作顺序；
- 可以复用当前 MoConVQ decoder/controller；
- 能和论文的 LLM integration 对齐，报告中有合理依据；
- 可作为主线微调失败时的可展示结果。

缺点：

- LLM 输出 token 可能格式错误，需要 validator；
- LLM 可能复制相似样例，而不是真正组合；
- token 级拼接仍可能有边界不平滑；
- 如果只用 depth0，动作细节会差；如果用 4-depth tuple，prompt 会更长；
- 评估会更依赖人工视觉检查。

## 12. 建议结论写法

如果主线训练效果仍不好，报告中可以这样表述：

```text
We implemented the intended fine-tuning pipeline for T2M-MoConGPT, but observed that long motion synthesis from directly concatenated HumanML3D clips can introduce boundary and semantic alignment noise. As a backup, we follow MoConVQ's LLM integration idea: using MoConVQ RVQ indices as a compact action representation, an LLM can perform in-context motion planning by recombining retrieved text-index examples. This route keeps MoConVQ responsible for low-level physics-based decoding while using the LLM only for high-level long-horizon composition.
```
