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
    diagnose_long_humanml3d_quality.py
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
  --drop-overlap-frames 1 \
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
  --drop-overlap-frames 1 \
  --caption-joiner " then " \
  --output-dir stage1_artifacts/long_humanml3d/val
```

输出文件：

```text
manifest.jsonl       # 每条长序列的样本来源、caption、边界和过渡分数
long_sequences.h5    # joints_22、joint_vecs_263、clip boundaries 等数据
summary.json         # 合成参数和统计信息
synthesize.log       # 合成日志
synthesize_progress.jsonl
```

合成逻辑会先采样一个 HumanML3D clip，再从候选池中选择 transition score 较低的后续 clip。score 会考虑根位置、根速度、朝向、脚部高度和脚部速度。拼接时会对后一个 clip 做根位置和 yaw 对齐，在边界处做短窗口平滑，并默认丢弃后一个 clip 的首个重叠帧，减少边界重复帧带来的监督噪声。

默认不接受超过阈值的 forced transition。如果需要放宽合成条件，可以调大 `--transition-max-score`，或者显式加入 `--allow-forced-transitions` 做对照实验。

### 2. 检查合成数据质量

正式构建 cache 之前，建议先检查拼接边界质量：

```bash
python Script/stage1/diagnose_long_humanml3d_quality.py \
  --long-h5 stage1_artifacts/long_humanml3d/train/long_sequences.h5 \
  --manifest stage1_artifacts/long_humanml3d/train/manifest.jsonl \
  --output-json stage1_artifacts/long_humanml3d/train/dataset_quality.json \
  --transition-jsonl stage1_artifacts/long_humanml3d/train/transition_quality.jsonl
```

这个诊断会统计每个拼接边界的 root gap、root velocity gap、yaw gap、脚高度差和脚速度差。它不是论文级评估指标，但可以提前发现明显坏的拼接数据，避免把错误监督喂给 GPT。

### 3. 可选：检查 MoConVQ observation 转换

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

### 4. 构建 GPT 训练 cache

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
  --window-policy clip \
  --sample-mode segment_prefix \
  --prefix-size 25 \
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
  --window-policy clip \
  --sample-mode segment_prefix \
  --prefix-size 25 \
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
target_masks:  (N, 50)
end_masks:     (N, 50)
segment_idxs:  (N,)
num_segments:  (N,)
segment_progress: (N,)
prefix_ranges / target_ranges / segment_ranges
sample_ids:    list[list[str]]
config:        dict
```

当前推荐使用 `--caption-mode window --window-policy clip --sample-mode segment_prefix`。这样每个训练样本包含“前序 motion prefix + 当前动作段 caption + 当前动作段目标 token”。loss 只监督当前动作段，prefix 只作为上下文。这个组织方式比单纯的 clip/window 训练更接近长文本分段生成时的使用方式。

### 5. 微调 MoConVQ GPT

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
  --train-scope temporal_base_head \
  --depth-weights 1.0,0.7,0.4,0.2 \
  --baseline-kl-weight 0.05 \
  --kl-temperature 2.0 \
  --end-token-weight 0.01 \
  --progress-conditioning auto \
  --teacher-progress-conditioning none \
  --progress-scale 1.0 \
  --context-size 51 \
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

训练脚本当前使用 segment-aware autoregressive objective：用上一时刻的 motion latent 上下文和当前 segment caption/progress 条件预测当前时刻的 RVQ indices，并对 4 个 RVQ depth 分别计算 token loss。对于 `segment_prefix` cache，前序 prefix token 不参与 CE/KL/accuracy，只作为上下文；当前 segment 的真实 token 才作为监督目标。

训练曲线：

```bash
python Script/stage1/plot_train_curves.py \
  --train-log stage1_artifacts/checkpoints/stage1_real/train_log.jsonl \
  --output-dir stage1_artifacts/figures/stage1_real
```

使用 baseline KL 时，默认让 teacher 使用原始 zero `clip_feature` 条件，即 `--teacher-progress-conditioning none`。student 可以使用 `--progress-conditioning auto` 学习分段进度；teacher 不接收 progress feature，避免用 baseline 在未训练条件下的输出约束 student。

### 6. 从文本生成 BVH

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
  --top-p 0.95 \
  --temperature 1.0 \
  --progress-conditioning auto \
  --gpu 0 \
  --seed 0
```

使用 baseline checkpoint 做对照时，把 `--checkpoint` 换成 `text_generation_GPT.pth` 即可。

对于包含 `" then "` 的复合文本，推荐使用 `--generation-mode segmented`。它会把长文本分成多个子动作段，每段分别编码文本，同时保留 motion latent prefix 作为上下文，并通过 progress conditioning 显式告诉模型当前是第几个动作段。这比“整段长 prompt + rolling generation”更容易表达当前应该执行到哪个动作段。

批量比较 baseline 和微调模型：

```bash
python Script/stage1/run_text_gpt_comparison.py \
  --run-id stage1_real_top_p \
  --prompts stage1_artifacts/prompts.tsv \
  --baseline-checkpoint text_generation_GPT.pth \
  --finetuned-checkpoint stage1_artifacts/checkpoints/stage1_real/best_val.pth \
  --base-data moconvq_base.data \
  --text-model t5-large \
  --text-encoder t5 \
  --max-text-length 256 \
  --max-length 120 \
  --generation-mode auto \
  --context-size 30 \
  --chunk-size 20 \
  --top-k 0 \
  --top-p 0.95 \
  --temperature 1.0 \
  --progress-conditioning auto \
  --baseline-progress-conditioning none \
  --progress-scale 0.5 \
  --seed 123 \
  --gpu 0
```

这里 baseline 和 finetuned 使用相同 top-p 采样、相同文本分段和相同长度设置；baseline 默认不使用 progress feature，因为原始 checkpoint 没有见过这个条件。当前实验里 `--progress-scale 0.5` 比 `1.0` 的 rollout 更稳，同时仍能缓解 baseline 容易早停的问题。

最终 Stage1 对比包建议使用统一 suite 脚本。它在同一组长文本 prompt 下收集 baseline GPT、微调 GPT、retrieval-only token backup，以及可选的外部 LLM response backup，并统一输出 BVH 工程指标：

```bash
python Script/stage1/run_stage1_model_suite.py \
  --run-id stage1_suite_real \
  --finetuned-checkpoint stage1_artifacts/checkpoints/stage1_real/best_val.pth \
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

主要输出位于 `stage1_artifacts/model_suite/<run_id>/`：

```text
prompts.tsv
bvh/*.bvh
bvh/*.log
summary_metrics.json
suite_summary.json
llm_backup/*/prompt.txt
llm_backup/*/retrieval_tokens.json
llm_backup/*/retrieval_validation.json
```

如果已经手动调用外部 LLM 得到 JSON response，可以把 prompt 名到 response 文件的映射写成一个 JSON object，并追加 `--llm-response-map responses.json`。suite 会先用本地 validator 校验 token，再解码成 `backup_llm` BVH。当前 `summary_metrics.json` 仍然只是 Stage1 工程诊断，不替代 MoConVQ 论文使用的 HumanML3D FID/R-precision。

`--motion-dataset` 建议显式传入，尤其是在临时 worktree 或脚本目录外运行时。backup retrieval 分支默认启用 `--backup-trim-repeat-runs`，用于截断检索复制带来的超长连续相同 RVQ tuple；修复数量会记录在每个 `retrieval_validation.json` 里。这是为了保证 token 文件可解码，不代表语义质量已经提升。

### 7. 渲染 BVH 为 MP4

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

5. **当前本地 HumanML3D 是 processed corpus ready，但不是 native BVH source ready。**
   `/home/chenjie/cc/robotics/HumanML3D` 下的 `all.txt/texts/new_joints/new_joint_vecs` 已对齐，可以用于 catalog、长 caption 合成和诊断；但当前没有 `pose_data/`、标准 AMASS motion `.npz` 或大规模 BVH exports。因此首选的 MoConVQ 原生 `MotionDataSet.add_bvh_with_character()` 数据路线还需要先恢复 source motion 或导出 BVH。

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
Script/stage1/check_stage1_data_readiness.py
Script/stage1/build_real_moconvq_gpt_cache.py
Script/stage1/build_bvh_character_gpt_cache.py
Script/stage1/train_real_text_gpt.py
Script/stage1/generate_long_motion.py
Script/stage1/run_stage1_model_suite.py
Script/stage1/render_bvh_to_mp4.py
```

补充文档：

```text
TEXT_GPT_TRAINING.md
STAGE1_BACKUP_PLAN.md
```
