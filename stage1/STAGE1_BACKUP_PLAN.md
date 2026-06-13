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

## 10. 当前实现状态

上述 backup route 现在由一个统一脚本实现：

```text
Script/stage1/llm_token_planning.py
```

它包含以下子命令：

```text
export-bank      从 GPT cache 导出 caption -> RVQ token example bank
retrieve         用轻量 token-overlap / IDF 检索相关 examples
build-prompt     为外部 LLM / 手动 ICL 生成 prompt 和检索记录
validate         解析和校验 LLM JSON response，输出 tokens.json
retrieval-plan   无 LLM API 时的 retrieval-only token baseline
decode-bvh       将 4-depth RVQ token sequence 解码成 BVH
```

这种合并式入口替代了最初计划中的多个小脚本，便于在课程环境中复现和测试。

### Task A: 导出 example bank

命令：

```text
python Script/stage1/llm_token_planning.py export-bank \
  --cache stage1_artifacts/gpt_cache_filtered_cache_stage1_20260612_174908/train_cache.pt \
  --output stage1_artifacts/llm_backup/example_bank_filtered_200.jsonl \
  --max-examples 200 \
  --max-tokens-per-example 32 \
  --min-tokens-per-example 8
```

输出 JSONL，每行包含 caption、window_range、indices、indices_depth0 等字段。

### Task B: 检索 examples

命令：

```text
python Script/stage1/llm_token_planning.py retrieve \
  --bank stage1_artifacts/llm_backup/example_bank_filtered_200.jsonl \
  --query "a person kicks with the right foot" \
  --top-k 5
```

第一版使用轻量 token-overlap / IDF scoring，不引入额外依赖。它不是 semantic retrieval 的最终形态，但足够支撑离线 prompt 构造和 retrieval baseline。

### Task C: LLM prompt runner

```text
python Script/stage1/llm_token_planning.py build-prompt \
  --bank stage1_artifacts/llm_backup/example_bank_filtered_200.jsonl \
  --text "a person walks forward then kicks with the right foot then dances" \
  --top-k 3 \
  --segment-token-count 12 \
  --max-tokens-per-example 12 \
  --output-prompt stage1_artifacts/llm_backup/runs/<run_id>/prompt.txt \
  --output-json stage1_artifacts/llm_backup/runs/<run_id>/retrieval.json
```

第一版不接外部 API，只生成 prompt 文件。使用者可以把 prompt 复制到 ChatGPT/Claude/其他 LLM，再把 response 保存回来。这符合 MoConVQ 论文中用商业 LLM 做 in-context learning 的设置，也避免 API key 和网络依赖。

### Task D: token validator

```text
python Script/stage1/llm_token_planning.py validate \
  --response-file stage1_artifacts/llm_backup/runs/<run_id>/raw_response.txt \
  --output-tokens stage1_artifacts/llm_backup/runs/<run_id>/tokens.json \
  --validation-json stage1_artifacts/llm_backup/runs/<run_id>/validation.json \
  --min-length 20 \
  --max-consecutive-repeat 5
```

validator 检查 JSON、tuple depth、整数范围、长度和连续重复 tuple。必要时可用 `--repair` 对少量越界值做 clamp，但正式实验应优先保留原始 response 和 validation report。

### Task E: retrieval-only baseline

```text
python Script/stage1/llm_token_planning.py retrieval-plan \
  --bank stage1_artifacts/llm_backup/example_bank_filtered_200.jsonl \
  --text "a person walks forward then kicks with the right foot then dances" \
  --top-k 3 \
  --segment-token-count 12 \
  --trim-repeat-runs \
  --output-tokens stage1_artifacts/llm_backup/runs/<run_id>/retrieval_tokens.json \
  --validation-json stage1_artifacts/llm_backup/runs/<run_id>/retrieval_validation.json
```

这条路线不调用 LLM，只把每段检索到的最佳 example token 复制/截断到目标长度。它不是最终 backup 质量上限，但能作为 deterministic lower bound，证明 token-to-BVH 解码路径是否可运行。`--trim-repeat-runs` 会截断超长连续相同 RVQ tuple，并把 `repeat_repairs` 写入 validation JSON；这是为了避免检索复制导致 token 文件不可解码，不应解释为语义质量提升。

### Task F: tokens to BVH

```text
python Script/stage1/llm_token_planning.py decode-bvh \
  --tokens stage1_artifacts/llm_backup/runs/<run_id>/tokens.json \
  --base-data moconvq_base.data \
  --motion-dataset simple_motion_data.h5 \
  --gpu 0 \
  --output-bvh stage1_artifacts/llm_backup/runs/<run_id>/output.bvh
```

功能：从 MoConVQ codebook embedding 重建 768-d latent，复用 MoConVQ decoder/controller 写 BVH。建议显式传 `--motion-dataset`，尤其是在临时 worktree 或仓库外运行时；MoConVQ 默认 config 使用相对路径 `./simple_motion_data.h5`。

## 10.1 Retrieval-only smoke result

Run id:

```text
llm_backup_smoke_20260613
```

Smoke prompt:

```text
a person walks forward then kicks with the right foot then dances
```

Inputs and outputs:

```text
example bank: stage1_artifacts/llm_backup/example_bank_filtered_200.jsonl
prompt:       stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/prompt.txt
tokens:       stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/retrieval_tokens.json
BVH:          stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/retrieval_output.bvh
metrics:      stage1_artifacts/llm_backup/runs/llm_backup_smoke_20260613/retrieval_bvh_metrics.json
```

Result:

| Item | Value |
|---|---:|
| exported examples | 200 |
| generated retrieval-only RVQ tuples | 36 |
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

- The backup engineering path is now a working minimal loop:

```text
GPT cache -> example bank -> retrieval/prompt -> validated RVQ tokens
-> MoConVQ decoder/controller -> BVH -> engineering metrics
```

- This smoke is not a semantic success claim. The retrieval-only baseline can copy and repeat examples, so the output must be compared visually and against baseline/finetuned GPT on the same prompts.
- The next backup experiment should use an actual LLM response through `build-prompt` + `validate`, then evaluate the generated BVH with the same Stage1 metric script and videos.

## 10.2 Unified suite retrieval-only smoke result

The backup route is now also wired into:

```text
Script/stage1/run_stage1_model_suite.py
```

This gives the backup path the same prompt set, artifact layout, and BVH metric
summary as baseline-vs-finetuned GPT comparison.

Smoke command:

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

| Prompt | Tuples after validation | Repeat repairs | BVH frames |
|---|---:|---:|---:|
| walk_turn_wave | 47 | 7 | 1128 |
| circle_crouch_stand | 50 | 4 | 1200 |
| walk_jump_dance | 51 | 3 | 1224 |
| sidestep_kick_turn | 54 | 0 | 1296 |

Model averages:

| Metric | backup_retrieval |
|---|---:|
| avg frames | 1212.0 |
| avg duration | 10.0996 s |
| avg root path | 2.5058 |
| avg root displacement | 0.6933 |
| avg pose velocity mean | 20.8206 |
| avg pose variance mean | 192.8391 |
| lag-20 repeat > 0.995 | 0.00 |
| early stop rate @ 1200 frames | 0.25 |

Interpretation:

- The backup path can now produce suite-compatible artifacts:
  `prompt.txt`, retrieval metadata, validated tokens, BVH, `summary_metrics.json`,
  and `suite_summary.json`.
- This is still retrieval-only, not actual LLM planning.  It proves the backup
  decoder/evaluation loop is reusable, but the next report-worthy backup result
  should use a real external LLM response via `--llm-response-map`.

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
