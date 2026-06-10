# Stage1 长程数据与可视化实验总结

更新时间：2026-06-02

本文档总结当前 Stage1 长程文本到动作生成实验，包括实验动机、数据构造、cache 策略修改、训练结果、可视化对比结果、已发现问题以及后续建议。

## 1. 实验目标

本轮实验的核心目标是验证：

1. 是否可以构造更适合长程动作生成的 Stage1 训练数据。
2. 是否可以让 Stage1 训练窗口跨越动作片段边界，从而让模型学习动作过渡。
3. 使用长程 sequence-window 数据微调后的文本 GPT，是否比原版 `text_generation_GPT.pth` 更适合复合动作生成。
4. 可视化生成结果是否能支持 token 指标上的改进结论。

原版 Stage1 文本 GPT 主要面向较短动作片段。对于如下复合 prompt：

```text
a person walks forward then turns around then crouches down
```

如果训练数据和训练窗口主要停留在单个 clip 内，模型很难直接学习 `"walk" -> "turn" -> "crouch"` 之间的过渡。因此本轮实验重点放在长程序列构造和跨边界窗口训练上。

## 2. 长程数据构造

本次使用的是一版清理后的 long HumanML3D 数据：

```text
/home/chenjie/cc/robotics/MoConVQ/stage1_artifacts/long_humanml3d_fixed/train
/home/chenjie/cc/robotics/MoConVQ/stage1_artifacts/long_humanml3d_fixed/val
```

数据由多个 HumanML3D 动作片段拼接成较长序列，每条长序列平均包含约 3 个动作片段。

### 2.1 训练集统计

```text
sequences:            1000
avg clips/sequence:   2.945
avg frames/sequence:  416.593
transitions:          1945
forced transitions:   0
failed:               3
```

### 2.2 验证集统计

```text
sequences:            200
avg clips/sequence:   2.99
avg frames/sequence:  410.2
transitions:          398
forced transitions:   0
failed:               1
```

### 2.3 数据质量判断

这版数据有两个重要特点：

1. 每条序列平均接近 3 个 clip，因此能覆盖复合动作。
2. `forced transitions = 0`，说明没有强行插入不可控过渡点，数据相对干净。

因此它可以作为当前 Stage1 长程训练的基础数据集。

## 3. Stage1 Cache 构造方式修改

### 3.1 原始问题

原先 Stage1 cache 更偏向 clip 内窗口，也就是训练窗口通常不会跨越动作片段边界。这种方式对短动作建模是合理的，但对长程复合动作存在不足：

```text
clip A: walk forward
clip B: turn around
clip C: crouch down
```

如果窗口都限制在单个 clip 内，模型看到的大多是：

```text
walk forward
turn around
crouch down
```

而不是：

```text
walk forward -> turn around
turn around -> crouch down
```

这样模型训练时缺少直接学习动作过渡的样本。

### 3.2 当前修改

修改文件：

```text
Script/stage1/real_moconvq_cache.py
```

将 Stage1 cache 默认窗口策略改为：

```python
window_policy="sequence"
```

CLI 默认值也同步改为：

```python
parser.add_argument("--window-policy", choices=("sequence", "clip"), default="sequence")
```

含义：

- `sequence`：在整条合成长序列上滑动窗口，允许窗口跨越 clip 边界。
- `clip`：保留旧行为，只在单个 clip 内取窗口。

旧策略仍然可以显式使用：

```bash
--window-policy clip
```

这样后续可以做对照实验。

### 3.3 Cache 输出

本次训练实际使用的 cache 路径：

```text
stage1_artifacts/fixed_sequence_window/gpt_cache/train_cache.pt
stage1_artifacts/fixed_sequence_window/gpt_cache/val_cache.pt
```

### 3.4 Cache 统计

训练 cache：

```text
windows:           3590
cross-boundary:    395 / 3590 = 11.0%
unique captions:   2470
failures:          0
size:              4.1G
```

验证 cache：

```text
windows:           707
cross-boundary:    99 / 707 = 14.0%
unique captions:   467
failures:          0
size:              812M
```

### 3.5 Cache 结果判断

sequence-window 策略确实产生了跨动作边界的训练窗口。训练集中约 11.0% 的窗口跨边界，验证集中约 14.0% 的窗口跨边界。

这个比例说明当前实验已经能验证跨边界窗口的有效性，但比例仍然偏低。如果后续目标是显著提升动作过渡质量，建议进一步加入 boundary-centered oversampling，使跨边界窗口比例提升到 30% 或 50%。

## 4. Stage1 训练实验

### 4.1 训练输出

本次完整训练输出目录：

```text
stage1_artifacts/fixed_sequence_window/checkpoints/seqwin_kl_depth_20epoch_foreground
```

主要文件：

```text
best_val.pth       740M
last.pth           739M
config.json
train_log.jsonl
```

当前推荐使用的 checkpoint：

```text
stage1_artifacts/fixed_sequence_window/checkpoints/seqwin_kl_depth_20epoch_foreground/best_val.pth
```

### 4.2 训练配置

```text
epochs:             20
batch_size:         2
train_scope:        base_head
lr:                 1e-5
weight_decay:       0.01
depth_weights:      1.0, 0.7, 0.4, 0.2
baseline_kl_weight: 0.05
kl_temperature:     2.0
end_token_weight:   0.01
num_workers:        0
gpu:                0
```

训练基于原版文本 GPT 权重继续微调。

训练目标使用 corrected autoregressive objective：使用上一时刻 motion latent 上下文预测当前时刻 RVQ indices，并对 4 个 RVQ depth 分别计算 token loss。这比直接把当前 latent 喂给当前 token prediction 更接近真实推理过程。

### 4.3 最佳验证结果

最佳验证结果出现在 epoch 18：

```text
epoch:    18
val loss: 2.6115
val CE:   2.3509
val acc:  38.42%
```

### 4.4 最后一轮结果

最后一轮 epoch 19：

```text
train loss: 2.4253
train CE:   2.1439
train acc:  41.36%

val loss:   2.6130
val CE:     2.3555
val acc:    38.31%
```

最后一轮验证集按 RVQ depth 的 token accuracy：

```text
depth0: 35.76%
depth1: 48.92%
depth2: 37.71%
depth3: 30.85%
```

### 4.5 训练趋势

整体趋势：

```text
val loss: 约 3.17 -> 约 2.61
val acc:  约 30.82% -> 约 38.3%
```

说明长程 sequence-window 数据和当前训练目标能够有效提升验证集 token 预测效果。

不过从大约 epoch 12 开始，验证 loss 基本在：

```text
2.61 - 2.63
```

之间震荡，继续使用相同配置训练收益预计有限。

### 4.6 训练结论

本次训练是有效的：

1. 模型能够在长程 sequence-window cache 上稳定收敛。
2. 验证 token accuracy 有明显提升。
3. best checkpoint 可以作为当前长程 Stage1 的新基线。

但也存在限制：

1. 验证 loss 已经平台化。
2. 跨边界窗口比例只有约 11%。
3. token 指标不能完全代表动作可视化质量。

因此后续重点应该转向可视化对比和更有针对性的跨边界数据增强。

## 5. 可视化生成实验

### 5.1 对比模型

新模型：

```text
stage1_artifacts/fixed_sequence_window/checkpoints/seqwin_kl_depth_20epoch_foreground/best_val.pth
```

原版 baseline：

```text
/home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth
```

MoConVQ base data：

```text
/home/chenjie/cc/robotics/MoConVQ/moconvq_base.data
```

T5 本地模型：

```text
/home/chenjie/cc/robotics/hf_models/t5-large
```

生成时需要从旧 MoConVQ 资源目录运行，因为环境初始化依赖：

```text
/home/chenjie/cc/robotics/MoConVQ/simple_motion_data.h5
```

### 5.2 生成设置

正式可视化使用 segmented generation：

```text
generation-mode: segmented
segment-length:  30
context-size:    51
chunk-size:      25
text-encoder:    t5
gpu:             0
```

对于包含 `" then "` 的复合 prompt，`segmented` 会将 prompt 拆成多个子动作段，每段分别编码文本，同时保留前一段 motion latent 作为上下文。这比整段 prompt 的 rolling generation 更容易表达当前应该生成到哪个动作。

### 5.3 测试 Prompt

共测试 3 条复合长 prompt：

```text
1. a person walks forward then turns around then crouches down

2. a person runs forward then jumps up then walks slowly

3. a person walks sideways then waves both arms then sits down
```

每条 prompt 分别用新模型和原版 baseline 生成。

### 5.4 BVH 输出

新模型 BVH：

```text
stage1_artifacts/generated_visualization/seqwin_best/walk_turn_crouch_seed0.bvh
stage1_artifacts/generated_visualization/seqwin_best/run_jump_walk_seed1.bvh
stage1_artifacts/generated_visualization/seqwin_best/side_wave_sit_seed2.bvh
```

原版 baseline BVH：

```text
stage1_artifacts/generated_visualization/baseline_original/walk_turn_crouch_seed0.bvh
stage1_artifacts/generated_visualization/baseline_original/run_jump_walk_seed1.bvh
stage1_artifacts/generated_visualization/baseline_original/side_wave_sit_seed2.bvh
```

正式 BVH 统计：

```text
frames: 2160
fps:    120
时长:   约 18 秒
```

### 5.5 MP4 输出

新模型视频：

```text
stage1_artifacts/generated_visualization/videos/seqwin_best/walk_turn_crouch_seed0.mp4
stage1_artifacts/generated_visualization/videos/seqwin_best/run_jump_walk_seed1.mp4
stage1_artifacts/generated_visualization/videos/seqwin_best/side_wave_sit_seed2.mp4
```

原版 baseline 视频：

```text
stage1_artifacts/generated_visualization/videos/baseline_original/walk_turn_crouch_seed0.mp4
stage1_artifacts/generated_visualization/videos/baseline_original/run_jump_walk_seed1.mp4
stage1_artifacts/generated_visualization/videos/baseline_original/side_wave_sit_seed2.mp4
```

### 5.6 并排对比视频

为了方便直接观察差异，额外合成了左右并排视频：

```text
stage1_artifacts/generated_visualization/videos/compare/walk_turn_crouch_seqwin_vs_baseline.mp4
stage1_artifacts/generated_visualization/videos/compare/run_jump_walk_seqwin_vs_baseline.mp4
stage1_artifacts/generated_visualization/videos/compare/side_wave_sit_seqwin_vs_baseline.mp4
```

并排视频中：

```text
左边: seqwin_best
右边: baseline_original
```

三个并排视频的检查结果：

```text
duration:   18.000000 seconds
resolution: 1920 x 720
status:     非空视频，抽帧像素方差非零
```

抽帧检查：

```text
run_jump_walk_seqwin_vs_baseline.png     (1920, 720), pixel std = 17.3969
side_wave_sit_seqwin_vs_baseline.png     (1920, 720), pixel std = 16.7013
walk_turn_crouch_seqwin_vs_baseline.png  (1920, 720), pixel std = 16.8679
```

说明视频不是黑屏或空帧。

## 6. 生成阶段发现的问题

### 6.1 End Token 提前停止

长程生成时发现原始采样函数会在预测到 end token 后提前停止。对于固定长度长程可视化，这会导致 chunk 不满长，并报错：

```text
RuntimeError: GPT returned too few latents for chunk: expected 25, got 23
```

这个问题不是 checkpoint 加载失败，也不是 MoConVQ decoder 失败，而是生成采样逻辑本身允许提前结束。

### 6.2 新增推理参数

为了解决可视化时固定长度生成的问题，在以下文件中新增了一个推理专用参数：

```text
Script/stage1/generate_long_motion.py
```

新增参数：

```bash
--suppress-end-token
```

作用：

1. 只在推理生成时生效。
2. 屏蔽 end token。
3. 只从真实 RVQ code 范围采样。
4. 强制模型按指定长度生成完整动作段。

这个改动不影响训练，不修改 checkpoint，默认行为仍保持原始采样逻辑。

### 6.3 短测试样本

生成链路调试时还留下了一个短测试样本：

```text
stage1_artifacts/generated_visualization/seqwin_best/smoke_walk_turn.bvh
stage1_artifacts/generated_visualization/videos/seqwin_best/smoke_walk_turn.mp4
```

该样本时长约 3.2 秒，仅用于验证生成和渲染链路，不作为正式实验结论依据。

正式可视化结果应查看：

```text
stage1_artifacts/generated_visualization/videos/compare/
```

## 7. 测试与验证

### 7.1 Cache 相关测试

已更新：

```text
tests/test_stage1_real_cache.py
```

主要变化：

1. 默认测试改为验证 `sequence` window 可以跨边界。
2. 保留显式 `window_policy="clip"` 的测试，确保旧行为仍可用。
3. forced-transition-margin 相关测试显式使用 `window_policy="clip"`。

已通过的测试：

```bash
python -m unittest tests.test_stage1_real_cache
python -m unittest tests.test_stage1_real_cache tests.test_stage1_real_train
```

完整 Stage1 测试发现当前 repo 根目录缺少部分外部资源，例如：

```text
moconvq_base.data
/home/chenjie/cc/HumanML3D/HumanML3D
```

这是资源路径问题，不是本次代码修改导致的失败。

### 7.2 生成相关测试

已更新：

```text
tests/test_stage1_real_generate.py
```

新增 `--suppress-end-token` 后，保持旧接口兼容，并更新测试替身参数。

验证命令：

```bash
python -m py_compile Script/stage1/generate_long_motion.py Script/stage1/render_bvh_to_mp4.py
conda run -n moconvq python -m unittest tests.test_stage1_real_generate
```

结果：

```text
Ran 8 tests in 1.142s
OK
```

## 8. 当前整体结论

### 8.1 已经完成的实验

目前已经完成：

1. 构造并检查清理后的 long HumanML3D 数据。
2. 将 Stage1 cache 默认窗口策略从 clip 内窗口改为 sequence-level 窗口。
3. 基于 sequence-window cache 构造完整 train/val cache。
4. 使用该 cache 训练 Stage1 文本 GPT 20 epoch。
5. 生成新模型与原版 baseline 的长程动作 BVH。
6. 渲染单独 MP4 和左右并排对比 MP4。
7. 修复长程生成时 end token 提前停止导致的固定长度生成失败问题。
8. 补充相关测试并验证通过。

### 8.2 当前结果如何

从 token 指标看，结果是有效的：

```text
val loss: 约 3.17 -> 约 2.61
val acc:  约 30.82% -> 约 38.3%
best val loss: 2.6115
best val acc:  38.42%
```

从数据构造看，sequence-window 策略也有效：

```text
train cross-boundary windows: 395 / 3590 = 11.0%
val cross-boundary windows:   99 / 707 = 14.0%
```

从可视化链路看，已经可以稳定生成固定长度复合动作视频：

```text
3 个 prompt
2 个模型
6 个正式 BVH
6 个正式 MP4
3 个左右并排对比 MP4
每个正式视频 18 秒
```

因此当前实验可以作为 Stage1 长程训练的有效基线。

### 8.3 当前不足

目前仍有几个限制：

1. 跨边界窗口比例较低，训练集中只有约 11%。
2. 验证 loss 在 epoch 12 后平台化，继续按当前配置训练收益有限。
3. token accuracy 不能完全代表动作质量。
4. 可视化结果还需要人工逐条观察，尤其是动作切换点。
5. `--suppress-end-token` 适合固定长度可视化，但它改变了推理时的停止机制，因此正式评估时应说明是否使用该参数。

## 9. 后续建议

### 9.1 人工观察可视化

优先查看：

```text
stage1_artifacts/generated_visualization/videos/compare/
```

重点观察：

1. 动作是否按 prompt 顺序发生。
2. 动作 A 到动作 B 的过渡是否自然。
3. 是否出现突然抖动、漂移、摔倒或姿态崩坏。
4. 后半段是否重复或退化。
5. 新模型是否比原版 baseline 更稳定。

### 9.2 做 boundary-centered oversampling

当前跨边界窗口比例只有约 11%。建议下一步构造更偏向过渡点的 cache：

```text
normal sequence windows + boundary-centered windows
```

目标比例可以设为：

```text
cross-boundary windows: 30% - 50%
```

这样可以更直接训练动作过渡能力。

### 9.3 做系统消融

建议比较以下模型：

```text
1. 原版 text_generation_GPT.pth
2. clip-window 微调模型
3. sequence-window 微调模型
4. boundary-oversampled sequence-window 微调模型
```

这样可以回答：

1. 提升是否来自长程数据。
2. 提升是否来自跨边界窗口。
3. 提升是否来自普通微调。
4. boundary oversampling 是否进一步改善过渡质量。

### 9.4 调整训练策略

当前训练已经平台化。后续如果继续训练，不建议直接用相同配置硬训。更合理的方向包括：

1. 降低学习率继续 fine-tune。
2. 增加 boundary window 比例。
3. 比较 `base_head` 与更大范围参数训练。
4. 调整 end token loss 或推理停止策略。
5. 加入更直接的 transition-focused validation set。

## 10. 关键文件与路径

### 10.1 代码

```text
Script/stage1/real_moconvq_cache.py
Script/stage1/generate_long_motion.py
Script/stage1/render_bvh_to_mp4.py
Script/stage1/train_real_text_gpt.py
```

### 10.2 测试

```text
tests/test_stage1_real_cache.py
tests/test_stage1_real_generate.py
tests/test_stage1_real_train.py
```

### 10.3 数据与 cache

```text
/home/chenjie/cc/robotics/MoConVQ/stage1_artifacts/long_humanml3d_fixed/train
/home/chenjie/cc/robotics/MoConVQ/stage1_artifacts/long_humanml3d_fixed/val

stage1_artifacts/fixed_sequence_window/gpt_cache/train_cache.pt
stage1_artifacts/fixed_sequence_window/gpt_cache/val_cache.pt
```

### 10.4 Checkpoint

```text
stage1_artifacts/fixed_sequence_window/checkpoints/seqwin_kl_depth_20epoch_foreground/best_val.pth
stage1_artifacts/fixed_sequence_window/checkpoints/seqwin_kl_depth_20epoch_foreground/last.pth
stage1_artifacts/fixed_sequence_window/checkpoints/seqwin_kl_depth_20epoch_foreground/train_log.jsonl
```

### 10.5 可视化结果

```text
stage1_artifacts/generated_visualization/seqwin_best/
stage1_artifacts/generated_visualization/baseline_original/
stage1_artifacts/generated_visualization/videos/seqwin_best/
stage1_artifacts/generated_visualization/videos/baseline_original/
stage1_artifacts/generated_visualization/videos/compare/
```

最推荐查看：

```text
stage1_artifacts/generated_visualization/videos/compare/walk_turn_crouch_seqwin_vs_baseline.mp4
stage1_artifacts/generated_visualization/videos/compare/run_jump_walk_seqwin_vs_baseline.mp4
stage1_artifacts/generated_visualization/videos/compare/side_wave_sit_seqwin_vs_baseline.mp4
```
