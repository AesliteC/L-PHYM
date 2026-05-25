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

## 2. 工作区结构

当前工作区在：

```text
/home/chenjie/cc/robotics/
```

主要包含两个重要目录：

```text
HumanML3D/   # 本地数据集与 HumanML3D 原始/中间处理文件，不建议上传 GitHub
MoConVQ/    # 主体代码仓库，Stage1 代码都放在这里，建议上传这个仓库中的代码改动
```

本项目默认使用以下本地路径：

```text
HumanML3D 数据根目录: ../HumanML3D/HumanML3D
MoConVQ 主仓库:       /home/chenjie/cc/robotics/MoConVQ
预训练 MoConVQ:       moconvq_base.data
预训练 GPT:           text_generation_GPT.pth
输出目录:             stage1_artifacts/
```

不要上传 `HumanML3D/` 数据目录。它包含 AMASS/HumanML3D 数据、body model、`pose_data`、`amass_data`、`new_joints`、`new_joint_vecs` 等大文件和受许可约束的数据。

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
  --gpu 0 \
  --output stage1_artifacts/gpt_cache/train_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/train_failures.jsonl
```

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
- 支持 padding token `513` 的 ignore；
- 记录 train/val loss、token accuracy、per-depth accuracy；
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
  tests.test_stage1_real_train -v
```

最近一次验证结果：

```text
Ran 14 tests in 9.900s
OK
```

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

注意：`t5-large` 的完整下载和正式编码尚未在大规模实验中验证。当前确认了 `transformers` 可 import，版本为 `4.46.3`。

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
Script/stage1/build_real_moconvq_gpt_cache.py
Script/stage1/train_real_text_gpt.py
```

## 6. 还需要完成的工作

### 6.1 正式数据规模实验

还没有完整跑完：

```text
1000+ 长序列合成
train/val cache 构建
20 epoch GPT 微调
生成结果评估
```

建议先从小规模逐步扩大：

```text
10 sequences -> 100 sequences -> 1000 sequences
```

每一步确认：

- failure log 是否为空或可接受；
- cache 的 window 数是否合理；
- indices 是否在合法范围内；
- 训练 loss 是否下降。

### 6.2 T5 模型下载和缓存

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

### 6.3 Retarget 质量检查

当前 HumanML3D 到 MoConVQ 的 retarget 是确定性 kinematic 近似：

```text
HumanML3D 22 joints -> MoConVQ 20 bodies
```

它已经经过 shape 和 encoder smoke test，但还没有做系统的视觉质量评估。后续建议：

- 抽样保存 retarget 后的 state/observation；
- 通过 MoConVQ decoder 或 tracking 生成 BVH；
- 人眼检查拼接边界和身体姿态；
- 检查脚滑、朝向突变、手臂异常等问题。

如果 retarget 质量不够，下一步应考虑更严格的 BVH/SMPL 到 MoConVQ character retarget。

### 6.4 Val cache 和评估指标

当前脚本支持 `--val-cache`，但还需要正式构建：

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
  --gpu 0 \
  --output stage1_artifacts/gpt_cache/val_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/val_failures.jsonl
```

### 6.5 长动作生成与展示

当前 `generate_long_motion.py` 是最小生成脚本。后续可以改进：

- 支持加载 `train_real_text_gpt.py` 产出的 `last.pth` 或 `best_val.pth`；
- 支持更长文本 prompt；
- 支持 rolling/chunked generation；
- 输出 BVH 到 `stage1_artifacts/generated/`；
- 保存 prompt、checkpoint、seed、生成长度等 metadata。

## 7. GitHub 上传建议

推荐只上传 `MoConVQ` 仓库中的 Stage1 代码，不上传 `HumanML3D` 数据。

应上传：

```text
.gitignore
STAGE1_README.md
MoConVQCore/Model/cross_trans_ori_fixsum.py
Script/stage1/
tests/test_stage1_*.py
```

不要上传：

```text
HumanML3D/
stage1_artifacts/
*.h5
*.pt
*.pth
__pycache__/
amass_data/
pose_data/
new_joints/
new_joint_vecs/
```

当前 `.gitignore` 已包含：

```text
stage1_artifacts/
*.h5
*.pth
```

建议后续也忽略：

```text
*.pt
__pycache__/
```

上传前建议检查：

```bash
git status --short --untracked-files=all
git diff --cached --name-only
```

只添加 Stage1 相关文件：

```bash
git add \
  .gitignore \
  STAGE1_README.md \
  MoConVQCore/Model/cross_trans_ori_fixsum.py \
  Script/stage1 \
  tests/test_stage1_gpt.py \
  tests/test_stage1_humanml3d.py \
  tests/test_stage1_motion_bridge.py \
  tests/test_stage1_real_cache.py \
  tests/test_stage1_real_synthesis.py \
  tests/test_stage1_real_train.py
```

提交：

```bash
git commit -m "docs: add stage1 handoff readme"
```

推送到自己的 GitHub fork 或新仓库：

```bash
git push -u origin stage1-real-pipeline
```

## 8. 给接手同学的最短运行顺序

```bash
cd /home/chenjie/cc/robotics/MoConVQ
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
```

先跑测试：

```bash
python -m unittest \
  tests.test_stage1_humanml3d \
  tests.test_stage1_motion_bridge \
  tests.test_stage1_gpt \
  tests.test_stage1_real_synthesis \
  tests.test_stage1_real_cache \
  tests.test_stage1_real_train -v
```

合成 train 长序列：

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
  --output-dir stage1_artifacts/long_humanml3d/train
```

构建 train cache：

```bash
python Script/stage1/build_real_moconvq_gpt_cache.py \
  --long-h5 stage1_artifacts/long_humanml3d/train/long_sequences.h5 \
  --manifest stage1_artifacts/long_humanml3d/train/manifest.jsonl \
  --base-data moconvq_base.data \
  --text-model t5-large \
  --window-size 50 \
  --window-stride 25 \
  --rvq-depth 4 \
  --gpu 0 \
  --output stage1_artifacts/gpt_cache/train_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/train_failures.jsonl
```

微调 GPT：

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
  --gpu 0 \
  --seed 0 \
  --save-every 1 \
  --num-workers 4
```

## 9. 当前状态一句话总结

Stage1 的代码框架、长序列合成、真实 MoConVQ encoder cache 构建、GPT 微调入口和测试都已经完成；下一步主要是跑正式规模实验、确认 T5 cache、检查 retarget 视觉质量，并用训练后的 checkpoint 做生成展示。
