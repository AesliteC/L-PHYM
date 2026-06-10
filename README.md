# L-PHYM Stage1: HumanML3D Long-Motion GPT Fine-Tuning

本仓库是在 MoConVQ 基础上实现的 Stage1 实验代码，目标是把 HumanML3D 的短动作-文本样本合成为较长的动作序列，并转换成 MoConVQ 文本条件 GPT 可以训练的格式，从而微调仓库内的 `Text2Motion_Transformer`。

这里的 GPT 不是 HuggingFace `CausalLM`。Stage1 使用的是 MoConVQ 原仓库中的 motion-token GPT：模型读取文本特征和历史 motion latent，输出每帧的 RVQ codebook token，再由 MoConVQ decoder/controller 生成动作和 BVH。

## Stage1 目标

Stage1 的实验链路如下：

```text
HumanML3D short motion clips
  -> transition-filtered long motion-language sequences
  -> HumanML3D 22-joint motion
  -> MoConVQ 20-body state
  -> MoConVQ 323-d observation
  -> MoConVQ encoder latent + RVQ indices
  -> T5 text features
  -> fine-tune MoConVQ Text2Motion_Transformer
  -> generate BVH / render MP4 for evaluation
```

训练目标不是预测文本 token，而是预测 motion token：

```text
motion latent:  (B, T, 768)
RVQ indices:    (B, T, 4)
text features:  (B, L, 1024)
text mask:      (B, L)
clip feature:   (B, 512)

model output:   per-frame RVQ logits
training target: 4-depth RVQ indices
```

## 当前完成情况

Stage1 当前已经完成了端到端工程链路：

- HumanML3D 数据索引与 split 读取。
- 基于 transition score 的长动作序列合成。
- HumanML3D 22-joint motion 到 MoConVQ 20-body state 的转换。
- MoConVQ observation、latent、RVQ indices 和 T5 text feature cache 构建。
- MoConVQ `Text2Motion_Transformer` 微调脚本。
- 文本到 BVH 的生成脚本。
- BVH 到 MP4 的可视化脚本。
- Stage1 相关单元测试和 smoke tests。

当前代码已经可以从 HumanML3D 合成数据一路跑到 GPT 微调与 BVH 生成。最近一次修正后的诊断训练显示 token-level 指标可以正常学习：20 epoch 内 validation loss 和 CE 持续下降，token accuracy 明显提升。这说明数据转换、cache 构建和训练 objective 的工程链路是可用的。

但当前模型效果还不是最终结果。定性观察中，微调模型仍可能在长动作生成时出现动作重复、段落顺序不稳定或后半段语义弱化。后续工作重点应放在数据质量、文本-动作窗口对齐和长程生成策略上，而不是只继续堆训练 epoch。

## 目录结构

```text
MoConVQ/
  Script/stage1/
    humanml3d.py
    synthesize_long_humanml3d.py
    convert_humanml3d_to_moconvq_observation.py
    real_moconvq_cache.py
    build_real_moconvq_gpt_cache.py
    train_real_text_gpt.py
    generate_long_motion.py
    render_bvh_to_mp4.py
    intermediate_motion_format.py
    export_baseline_intermediate.py

  tests/
    test_stage1_*.py

  stage1_artifacts/
    long_humanml3d/
    gpt_cache/
    checkpoints/
    generated_bvh_compare/
    logs/
```

`stage1_artifacts/` 是默认实验输出目录。重新合成数据、构建 cache、训练 checkpoint、生成 BVH 和渲染视频都会写到这个目录下。

## 项目代码结构

```text
.
├── MoConVQCore/                      # MoConVQ 主模型、环境和工具代码
│   ├── Model/                        # MoConVQ、RVQ、Text2Motion Transformer 等模型
│   ├── Env/                          # VCL/ODE 跟踪环境
│   └── Utils/                        # motion、quaternion、PyTorch 和数据工具
├── Script/                           # 数据处理、tokenize、训练和生成脚本
│   └── stage1/                       # Stage1 长动作数据合成、cache、训练和生成入口
├── tests/                            # Stage1 单元测试和 smoke tests
├── ModifyODESrc/                     # 修改版 ODE / VclSimuBackend C++/Cython 源码
├── diff-quaternion/                  # quaternion / rotation C++、CUDA、PyTorch 扩展
├── Data/                             # world、参数、贴图等运行数据
├── ThirdParty/                       # Eigen、GLEW、GL 等第三方依赖
└── stubs/                            # Python 类型 stub
```

## 运行环境

建议在已配置好的 MoConVQ 环境（见UPSTREAM_README.md）中运行：

```bash
conda activate moconvq
cd MoConVQ
```

默认假设以下资源可用：

```text
../HumanML3D/HumanML3D/    # HumanML3D 数据目录
moconvq_base.data          # MoConVQ 基础模型与 RVQ codebook
text_generation_GPT.pth    # MoConVQ 文本 GPT 初始化权重
```

HumanML3D 目录应至少包含：

```text
all.txt
train.txt
val.txt
test.txt
new_joints/
new_joint_vecs/
texts/
```

文本特征默认使用 `t5-large`。如果本地没有缓存，首次构建 cache 或生成动作时需要能够访问 HuggingFace，或者把 `--text-model` 指向本地 T5 模型目录。

## 快速开始

下面所有命令都假设当前目录是 `MoConVQ/`。

### 1. 合成长动作数据集

训练集示例：

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

验证集示例：

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
  --caption-joiner " then " \
  --output-dir stage1_artifacts/long_humanml3d/val
```

输出文件：

```text
manifest.jsonl       # 每条长序列的样本来源、caption、边界和过渡分数
long_sequences.h5    # joints_22、joint_vecs_263、clip boundaries 等数据
summary.json         # 合成参数和统计信息
```

合成逻辑会先采样一个 HumanML3D clip，再从候选池中选择 transition score 较低的后续 clip。score 会考虑根位置、根速度、朝向、脚部高度和脚部速度。拼接时会对后一个 clip 做根位置和 yaw 对齐，并在边界处做短窗口平滑。

默认不接受超过阈值的 forced transition。如果需要放宽合成条件，可以调大 `--transition-max-score`，或者显式加入 `--allow-forced-transitions` 做对照实验。

### 2. 可选：检查 MoConVQ observation 转换

如果只想检查 retarget 和 observation 是否能生成，可以先运行：

```bash
python Script/stage1/convert_humanml3d_to_moconvq_observation.py \
  --long-h5 stage1_artifacts/long_humanml3d/train/long_sequences.h5 \
  --manifest stage1_artifacts/long_humanml3d/train/manifest.jsonl \
  --output-h5 stage1_artifacts/long_humanml3d/train/moconvq_observations.h5 \
  --summary stage1_artifacts/long_humanml3d/train/moconvq_observations_summary.json
```

期望输出中的核心 shape：

```text
state_20x13:     (T, 20, 13)
observation_323: (T, 323)
```

### 3. 构建 GPT 训练 cache

训练集 cache：

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
  --window-policy sequence \
  --gpu 0 \
  --output stage1_artifacts/gpt_cache/train_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/train_failures.jsonl
```

验证集 cache：

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
  --window-policy sequence \
  --gpu 0 \
  --output stage1_artifacts/gpt_cache/val_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/val_failures.jsonl
```

cache 主要字段：

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

当前推荐使用 `--caption-mode window --window-policy sequence`。这样每个 50-token motion window 直接在整条合成长序列上滑窗，允许窗口跨过 clip 边界；同时 `caption-mode=window` 会根据窗口覆盖到的 clip 选择局部或跨段 caption，例如 `walk then turn`，避免把完整长文本无差别复制给所有局部窗口。若要做保守对照实验，可以显式传 `--window-policy clip`，它会回到按原始 clip 边界切窗口的行为。

### 4. 微调 MoConVQ GPT

```bash
python Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache/train_cache.pt \
  --val-cache stage1_artifacts/gpt_cache/val_cache.pt \
  --init-checkpoint text_generation_GPT.pth \
  --base-data moconvq_base.data \
  --output-dir stage1_artifacts/checkpoints/stage1_real \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --train-scope base_head \
  --depth-weights 1.0,1.0,0.7,0.5 \
  --baseline-kl-weight 0.05 \
  --kl-temperature 1.0 \
  --end-token-weight 0.05 \
  --gpu 0 \
  --seed 0 \
  --save-every 1 \
  --num-workers 4
```

输出内容：

```text
checkpoint_epoch_*.pth
best_val.pth
last.pth
train_log.jsonl
config.json
```

训练脚本当前使用 corrected autoregressive objective：用上一时刻的 motion latent 上下文预测当前时刻的 RVQ indices，并对 4 个 RVQ depth 分别计算 token loss。这个设置比直接把当前 latent 喂给当前 token prediction 更接近推理时的自回归条件。

### 5. 从文本生成 BVH

使用微调 checkpoint：

```bash
python Script/stage1/generate_long_motion.py \
  --checkpoint stage1_artifacts/checkpoints/stage1_real/best_val.pth \
  --text "a person walks forward then turns around then crouches down" \
  --output-bvh stage1_artifacts/generated_bvh_compare/stage1_real/example.bvh \
  --base-data moconvq_base.data \
  --text-encoder t5 \
  --text-model t5-large \
  --generation-mode segmented \
  --segment-length 30 \
  --context-size 51 \
  --chunk-size 25 \
  --gpu 0 \
  --seed 0
```

使用 baseline checkpoint 做对照时，把 `--checkpoint` 换成 `text_generation_GPT.pth` 即可。

对于包含 `" then "` 的复合文本，推荐使用 `--generation-mode segmented`。它会把长文本分成多个子动作段，每段分别编码文本，同时保留 motion latent prefix 作为上下文。这比“整段长 prompt + rolling generation”更容易表达当前应该执行到哪个动作段。

### 6. 渲染 BVH 为 MP4

```bash
python Script/stage1/render_bvh_to_mp4.py \
  --input stage1_artifacts/generated_bvh_compare/stage1_real \
  --output-dir stage1_artifacts/generated_bvh_compare/stage1_real/videos \
  --fps 30 \
  --width 960 \
  --height 720
```

## 测试

运行 Stage1 相关测试：

```bash
python -m unittest discover -s tests -p "test_stage1*.py" -v
```

也可以只跑核心链路测试：

```bash
python -m unittest \
  tests.test_stage1_humanml3d \
  tests.test_stage1_real_synthesis \
  tests.test_stage1_real_cache \
  tests.test_stage1_real_train \
  tests.test_stage1_real_generate \
  tests.test_stage1_render_bvh \
  -v
```

## 当前限制

1. **HumanML3D clip 拼接仍可能引入边界噪声。**
   即使做了根位置、yaw 和短窗口平滑，不同 clip 的速度、脚接触、姿态相位和动作语义仍可能不连续。边界附近的训练窗口可能影响 GPT 对长动作的学习。

2. **MoConVQ GPT 的 motion context 有长度限制。**
   `Text2Motion_Transformer` 的 block size 限制了单次可见的 motion latent 数量。长动作需要 rolling 或 segmented generation，而不是一次性喂入任意长度 motion context。

3. **长文本和局部动作窗口之间存在对齐问题。**
   如果每个 50-frame window 都使用整段长 caption，模型很难知道当前窗口对应的是第几个子动作。因此当前默认使用局部 caption cache，并在生成时使用 segmented mode。

4. **token-level 指标不能完全代表视觉质量。**
   validation loss 和 token accuracy 能说明模型在训练分布上学习到 RVQ token 预测，但还需要结合 BVH/MP4 定性观察，以及动作重复、动作顺序、root drift、foot sliding 等指标做评估。

## 后续工作

Stage1 后续处理以下方向：

- 改进长序列合成策略，减少跨 clip 边界不自然样本。
- 对边界窗口进行过滤、降权或单独标注，避免 GPT 学到拼接噪声。
- 进一步完善 segment-aware conditioning，让生成过程明确知道当前动作段。
- 建立更系统的 baseline 对比：原始 `text_generation_GPT.pth`、不同 caption mode、不同 window policy、不同 transition threshold。
- 增加长动作评估指标，包括重复率、动作顺序一致性、root trajectory 稳定性和 foot sliding。
- 保留 LLM in-context motion-token planning 作为备选路线：让大模型负责长文本分解和 MoConVQ token 级动作规划，再由 MoConVQ decoder/controller 负责物理动作生成。

## 参考入口

主线脚本：

```text
Script/stage1/synthesize_long_humanml3d.py
Script/stage1/build_real_moconvq_gpt_cache.py
Script/stage1/train_real_text_gpt.py
Script/stage1/generate_long_motion.py
Script/stage1/render_bvh_to_mp4.py
```

补充文档：

```text
TEXT_GPT_TRAINING.md
STAGE1_BACKUP_PLAN.md
```
