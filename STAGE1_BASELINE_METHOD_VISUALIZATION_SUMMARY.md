# Stage1 使用 Baseline 推理方法的可视化实验总结

更新时间：2026-06-10

本文档总结本轮新增的可视化消融实验：在我们已经训练好的 Stage1 long sequence-window 模型上，不再使用之前的长程分段强制生成策略，而是改用原版 baseline 的推理方式重新生成可视化结果。该实验的目的不是重新训练模型，而是隔离“模型本身”和“生成策略”两个因素，判断目前视觉效果不佳是否主要来自推理分布不匹配。

## 1. 实验背景

前面已经完成了一版 Stage1 长程数据训练实验，使用的 checkpoint 是：

```text
stage1_artifacts/fixed_sequence_window/checkpoints/seqwin_kl_depth_20epoch_foreground/best_val.pth
```

该模型来自 sequence-window 训练，即训练窗口可以在整条合成长序列上滑动，允许一部分窗口跨越动作片段边界。该模型在 token-level 验证指标上优于原始无拼接数据训练结果：

```text
seqwin long-data best val loss: 2.6115
seqwin long-data best val acc:  38.42%
```

但是之前做可视化时，使用的是为了长程复合动作设计的强制生成策略：

```text
generation-mode: segmented
segment-length:  30
greedy:          yes
suppress EOS:    yes
prompt split:    by " then "
```

以 3 段 prompt 为例：

```text
a person walks forward then turns around then crouches down
```

旧策略会将 prompt 分成 3 段，每段强制生成 30 个 latent，总共生成：

```text
30 x 3 = 90 latent steps
```

对应 BVH：

```text
2160 frames
120 fps
duration = 2160 / 120 = 18.0 seconds
```

这个策略和原版 MoConVQ baseline 的推理方式差异很大，因此之前的可视化结果不能直接回答“模型是否真的变差”。它同时混入了以下因素：

1. fine-tuned checkpoint 和原始 checkpoint 的差异。
2. greedy sampling 和 categorical sampling 的差异。
3. suppress EOS 和允许 EOS 的差异。
4. segmented 多段强制生成和整句一次生成的差异。
5. 90 latent 长程 rollout 和原版 50 左右 latent rollout 的差异。

本轮实验专门去掉这些推理策略上的额外干预，只保留我们的 `seqwin` checkpoint，使用 baseline 风格的生成方法。

## 2. 本轮实验目标

本轮实验要验证：

1. 在同一个 `seqwin` fine-tuned 模型上，换成原版 baseline 推理方式后，可视化结果是否更稳定。
2. 之前视觉效果较差是否可能主要由 `segmented + greedy + suppress EOS + 90 latent` 的推理分布不匹配导致。
3. 为后续问题定位建立一个更公平的对照：同一个模型，不同推理方式。

本轮实验不是验证长程能力的最终实验，因为 baseline 推理方式本身只生成约 50 个 latent，时长约 9.8 秒，不能覆盖之前 18 秒长程 rollout 的完整长度。它更适合作为推理策略消融实验。

## 3. Baseline 推理方法定义

原版 MoConVQ 文本生成脚本中，核心调用是：

```python
cur_embedding, _ = gpt.sample(clip_feature, bert_feature, bert_mask)
```

对应模型默认参数：

```python
def sample(..., if_categorial=True, max_length=50, ...)
```

因此原版 baseline 推理不是 greedy，而是：

```text
top-k categorical sampling
top-k = 50
allow EOS
whole prompt once
max_length = 50
```

模型 `sample(max_length=50)` 内部最终返回：

```python
return ls[:, :-1, :]
```

所以实际最多返回 49 个 motion latent。为了在当前 Stage1 可视化脚本里尽量等价复现原版行为，本轮使用：

```text
generation-mode:   rolling
max-length:        49
chunk-size:        49
context-size:      51
greedy:            no
suppress-end-token:no
allow-early-stop:  yes
```

其中 `chunk-size=49` 很重要。之前如果使用默认 `chunk-size=25`，虽然也是 rolling，但实际会分两次采样，仍然不是原版 `gpt.sample(max_length=50)` 的一次性生成行为。因此本轮把 `chunk-size` 设为 49，使其更接近整句一次采样。

## 4. 本轮实验改动

本轮没有修改训练代码，也没有重新训练模型。实际改动是推理/可视化设置：

### 4.1 保持不变的部分

```text
checkpoint:
stage1_artifacts/fixed_sequence_window/checkpoints/seqwin_kl_depth_20epoch_foreground/best_val.pth

base-data:
/home/chenjie/cc/robotics/MoConVQ/moconvq_base.data

text encoder:
T5

text model:
/home/chenjie/cc/robotics/hf_models/t5-large

GPU:
0
```

### 4.2 改变的部分

从之前的长程强制生成：

```text
generation-mode: segmented
segment-length:  30
greedy:          yes
suppress EOS:    yes
latent length:   90 for 3-segment prompts
duration:        18.0 seconds
```

改为本轮 baseline 风格推理：

```text
generation-mode:   rolling
max-length:        49
chunk-size:        49
greedy:            no
suppress EOS:      no
allow early stop:  yes
latent length:     up to 49
duration:          9.8 seconds if full length
```

### 4.3 为什么必须使用 `--allow-early-stop`

原版 baseline 允许 EOS。当前 Stage1 rolling 包装如果不开 `--allow-early-stop`，当模型提前生成 EOS、返回 latent 数少于固定 chunk 长度时，会报错：

```text
RuntimeError: GPT returned too few latents for chunk
```

这不是模型失败，而是当前包装逻辑默认把“提前结束”当成异常。为了复现原版 baseline 行为，本轮必须打开：

```text
--allow-early-stop
```

本次实际生成的 3 条样本都输出了 1176 BVH frames，即都达到了 49 latent 的完整长度，没有提前短于 49 latent。

## 5. 实验命令

运行目录需要使用原始 MoConVQ 目录：

```text
/home/chenjie/cc/robotics/MoConVQ
```

原因是 MoConVQ 的配置里仍然包含相对路径：

```text
./simple_motion_data.h5
```

如果从 `/home/chenjie/cc/robotics_rmw` 直接运行，会出现 `simple_motion_data.h5` 找不到的问题。

### 5.1 walk-turn-crouch

```bash
conda run -n moconvq env PYTHONPATH=/home/chenjie/cc/robotics_rmw \
  python /home/chenjie/cc/robotics_rmw/Script/stage1/generate_long_motion.py \
  --checkpoint /home/chenjie/cc/robotics_rmw/stage1_artifacts/fixed_sequence_window/checkpoints/seqwin_kl_depth_20epoch_foreground/best_val.pth \
  --text "a person walks forward then turns around then crouches down" \
  --output-bvh /home/chenjie/cc/robotics_rmw/stage1_artifacts/generated_visualization/seqwin_baseline_method/walk_turn_crouch_seed0.bvh \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --text-encoder t5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --generation-mode rolling \
  --max-length 49 \
  --context-size 51 \
  --chunk-size 49 \
  --allow-early-stop \
  --gpu 0 \
  --seed 0
```

### 5.2 run-jump-walk

```bash
conda run -n moconvq env PYTHONPATH=/home/chenjie/cc/robotics_rmw \
  python /home/chenjie/cc/robotics_rmw/Script/stage1/generate_long_motion.py \
  --checkpoint /home/chenjie/cc/robotics_rmw/stage1_artifacts/fixed_sequence_window/checkpoints/seqwin_kl_depth_20epoch_foreground/best_val.pth \
  --text "a person runs forward then jumps up then walks slowly" \
  --output-bvh /home/chenjie/cc/robotics_rmw/stage1_artifacts/generated_visualization/seqwin_baseline_method/run_jump_walk_seed1.bvh \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --text-encoder t5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --generation-mode rolling \
  --max-length 49 \
  --context-size 51 \
  --chunk-size 49 \
  --allow-early-stop \
  --gpu 0 \
  --seed 1
```

### 5.3 side-wave-sit

```bash
conda run -n moconvq env PYTHONPATH=/home/chenjie/cc/robotics_rmw \
  python /home/chenjie/cc/robotics_rmw/Script/stage1/generate_long_motion.py \
  --checkpoint /home/chenjie/cc/robotics_rmw/stage1_artifacts/fixed_sequence_window/checkpoints/seqwin_kl_depth_20epoch_foreground/best_val.pth \
  --text "a person walks sideways then waves both arms then sits down" \
  --output-bvh /home/chenjie/cc/robotics_rmw/stage1_artifacts/generated_visualization/seqwin_baseline_method/side_wave_sit_seed2.bvh \
  --base-data /home/chenjie/cc/robotics/MoConVQ/moconvq_base.data \
  --text-encoder t5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --generation-mode rolling \
  --max-length 49 \
  --context-size 51 \
  --chunk-size 49 \
  --allow-early-stop \
  --gpu 0 \
  --seed 2
```

### 5.4 渲染命令

```bash
conda run -n moconvq python /home/chenjie/cc/robotics_rmw/Script/stage1/render_bvh_to_mp4.py \
  --input /home/chenjie/cc/robotics_rmw/stage1_artifacts/generated_visualization/seqwin_baseline_method \
  --output-dir /home/chenjie/cc/robotics_rmw/stage1_artifacts/generated_visualization/videos/seqwin_baseline_method \
  --fps 30 \
  --width 960 \
  --height 720
```

## 6. 输出目录

BVH 输出目录：

```text
stage1_artifacts/generated_visualization/seqwin_baseline_method/
```

MP4 输出目录：

```text
stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/
```

## 7. 本轮实验结果汇总

### 7.1 BVH 结果

| 样本 | Prompt | Seed | BVH 文件 | 文件大小 | Frames | FPS | 时长 |
|---|---|---:|---|---:|---:|---:|---:|
| walk_turn_crouch | `a person walks forward then turns around then crouches down` | 0 | `stage1_artifacts/generated_visualization/seqwin_baseline_method/walk_turn_crouch_seed0.bvh` | 819221 bytes | 1176 | 120 | 9.8s |
| run_jump_walk | `a person runs forward then jumps up then walks slowly` | 1 | `stage1_artifacts/generated_visualization/seqwin_baseline_method/run_jump_walk_seed1.bvh` | 819353 bytes | 1176 | 120 | 9.8s |
| side_wave_sit | `a person walks sideways then waves both arms then sits down` | 2 | `stage1_artifacts/generated_visualization/seqwin_baseline_method/side_wave_sit_seed2.bvh` | 819338 bytes | 1176 | 120 | 9.8s |

三条 BVH 的 header 都显示：

```text
Frames: 1176
Frame Time: 0.008333
```

由于 `Frame Time = 1 / 120`，所以每条时长为：

```text
1176 / 120 = 9.8 seconds
```

这也说明本轮 3 条样本都生成到了 49 latent 的完整长度：

```text
1176 frames / 49 latent = 24 frames per latent
```

### 7.2 MP4 结果

| 样本 | MP4 文件 | 文件大小 | 渲染状态 |
|---|---|---:|---|
| walk_turn_crouch | `stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/walk_turn_crouch_seed0.mp4` | 438745 bytes | 成功 |
| run_jump_walk | `stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/run_jump_walk_seed1.mp4` | 466850 bytes | 成功 |
| side_wave_sit | `stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/side_wave_sit_seed2.mp4` | 388938 bytes | 成功 |

渲染脚本成功完成：

```text
render .../run_jump_walk_seed1.bvh -> .../run_jump_walk_seed1.mp4
render .../side_wave_sit_seed2.bvh -> .../side_wave_sit_seed2.mp4
render .../walk_turn_crouch_seed0.bvh -> .../walk_turn_crouch_seed0.mp4
```

随后使用 `moconvq` 环境中的 ffmpeg 对三个 MP4 做了解码校验，命令均正常退出且无错误输出：

```bash
conda run -n moconvq ffmpeg -v error -i <video>.mp4 -f null -
```

因此本轮 MP4 输出是有效视频文件，不是空文件或损坏文件。

## 8. 与之前可视化策略的对比

### 8.1 推理设置对比

| 项目 | 之前的 seqwin_best 可视化 | 本轮 seqwin_baseline_method 可视化 |
|---|---|---|
| checkpoint | `seqwin_kl_depth_20epoch_foreground/best_val.pth` | `seqwin_kl_depth_20epoch_foreground/best_val.pth` |
| prompt 使用方式 | 按 `" then "` 切成多段 | 整句 prompt 一次输入 |
| generation mode | `segmented` | `rolling` |
| sampling | greedy / top1 | top-k categorical sampling |
| EOS | suppress EOS | allow EOS |
| 每段长度 | 30 latent | 不分段 |
| 总 latent 长度 | 90 latent | 49 latent |
| BVH frames | 2160 | 1176 |
| 视频时长 | 18.0s | 9.8s |
| 主要用途 | 测试强制长程复合生成 | 复现 baseline 风格推理，做推理策略消融 |

### 8.2 输出长度对比

| 样本 | 之前 seqwin_best frames | 本轮 baseline-method frames | 之前时长 | 本轮时长 |
|---|---:|---:|---:|---:|
| walk_turn_crouch | 2160 | 1176 | 18.0s | 9.8s |
| run_jump_walk | 2160 | 1176 | 18.0s | 9.8s |
| side_wave_sit | 2160 | 1176 | 18.0s | 9.8s |

长度变短是预期现象，不代表生成失败。原因是本轮刻意采用 baseline 风格的短 horizon 推理，最多生成 49 latent，而不是之前强制生成 90 latent。

### 8.3 文件大小对比

| 样本 | 之前 seqwin_best MP4 | 大小 | 本轮 baseline-method MP4 | 大小 |
|---|---|---:|---|---:|
| walk_turn_crouch | `stage1_artifacts/generated_visualization/videos/seqwin_best/walk_turn_crouch_seed0.mp4` | 466671 bytes | `stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/walk_turn_crouch_seed0.mp4` | 438745 bytes |
| run_jump_walk | `stage1_artifacts/generated_visualization/videos/seqwin_best/run_jump_walk_seed1.mp4` | 496697 bytes | `stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/run_jump_walk_seed1.mp4` | 466850 bytes |
| side_wave_sit | `stage1_artifacts/generated_visualization/videos/seqwin_best/side_wave_sit_seed2.mp4` | 494456 bytes | `stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/side_wave_sit_seed2.mp4` | 388938 bytes |

MP4 文件大小不能直接表示动作质量，但可以作为输出完整性的辅助检查。本轮三个视频大小均正常，并且解码校验通过。

## 9. 与原始 baseline_original 的关系

需要特别注意：之前目录名里的 `baseline_original` 容易造成误解。

已有目录：

```text
stage1_artifacts/generated_visualization/videos/baseline_original/
```

该目录使用的是原始 pretrained checkpoint：

```text
/home/chenjie/cc/robotics/MoConVQ/text_generation_GPT.pth
```

但它当时的生成策略仍然是我们后续可视化脚本里的强制长程策略：

```text
segmented + greedy + suppress EOS + segment-length=30
```

因此它不是严格意义上的“原版 baseline 推理”。它更准确地说是：

```text
original checkpoint + current hard long-horizon visualization strategy
```

而本轮结果是：

```text
seqwin fine-tuned checkpoint + original baseline-style inference strategy
```

这两个对照回答的问题不同：

| 目录 | checkpoint | 推理方法 | 回答的问题 |
|---|---|---|---|
| `baseline_original/` | 原始 pretrained GPT | segmented + greedy + suppress EOS | 原始模型在强制长程策略下表现如何 |
| `seqwin_best/` | seqwin fine-tuned GPT | segmented + greedy + suppress EOS | 训练后模型在强制长程策略下表现如何 |
| `seqwin_baseline_method/` | seqwin fine-tuned GPT | baseline-style categorical sampling | 训练后模型在原版推理分布下表现如何 |

如果后续要做最完整的 2x2 对照，还应该补一组：

```text
original pretrained GPT + baseline-style categorical sampling
```

这样可以形成完整矩阵：

| 模型 | 强制长程策略 | baseline 推理策略 |
|---|---|---|
| original pretrained | 已有 `baseline_original/` | 建议补充 |
| seqwin fine-tuned | 已有 `seqwin_best/` | 本轮已完成 |

## 10. 对当前问题定位的意义

本轮实验的核心意义是：把“训练出来的模型不好”和“可视化生成策略太激进”拆开看。

之前的 poor visual quality 可能来自以下因素：

```text
segmented generation
+ greedy decoding
+ suppress EOS
+ fixed 90 latent rollout
+ long compound prompt per local segment
```

这些设置会让模型远离原版训练/推理分布。特别是前面 token 诊断已经观察到 fine-tuned 模型在 greedy 下存在 token diversity collapse：

```text
seqwin side_wave_sit:
repeat rate around 0.72
unique token count per depth very low
```

因此，如果本轮 baseline-style sampling 可视化明显更自然、更少停滞，那么可以说明：

```text
主要问题不是 seqwin checkpoint 完全不可用，
而是之前的长程强制推理策略放大了 token 重复和分布偏移。
```

如果本轮 baseline-style sampling 结果仍然明显不好，那么问题就更可能来自：

```text
1. fine-tuning 后 token 分布本身退化；
2. caption-window 对齐噪声；
3. sequence-window 合成数据仍有语义噪声；
4. token-level 指标提升没有转化为 motion-level 质量提升。
```

因此本轮实验是定位问题来源的重要中间步骤。

## 11. 当前结论

本轮已经完成以下事项：

1. 使用 `seqwin` fine-tuned checkpoint 重新生成了 3 条 baseline-style 可视化。
2. 推理方式改为整句 prompt、categorical sampling、允许 EOS、不使用 greedy、不 suppress EOS。
3. 每条输出 1176 BVH frames，对应 49 latent、约 9.8 秒。
4. 三条 BVH 均成功生成，三条 MP4 均成功渲染。
5. MP4 解码校验通过，输出文件有效。
6. 本轮结果与之前 `seqwin_best/` 的差异主要来自推理策略，而不是 checkpoint。

当前最重要的判断是：

```text
之前的 18 秒可视化结果不能直接代表 seqwin 模型本身的能力。
它同时受到了 segmented、greedy、suppress EOS、固定 90 latent 长程 rollout 的影响。
```

本轮 `seqwin_baseline_method/` 结果提供了一个更接近原版推理分布的对照，可以用来判断视觉问题是否主要来自生成策略。

## 12. 后续建议

建议下一步按照以下顺序继续定位问题：

1. 人工观看并对比：

```text
stage1_artifacts/generated_visualization/videos/seqwin_best/
stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/
```

重点看：

```text
motion 是否更自然
是否仍然重复/停滞
是否有明显语义动作
动作幅度是否过小
是否存在不稳定或异常姿态
```

2. 补齐 original pretrained checkpoint 的 baseline-style 推理结果：

```text
original checkpoint + rolling + categorical sampling + allow EOS
```

这一步可以补完整 2x2 对照矩阵，判断 fine-tuning 本身是否损害了原版短 horizon 生成能力。

3. 针对两种推理策略做 token 诊断：

```text
generated latent length
EOS step
repeat rate
unique token count per RVQ depth
top-k entropy
top-1/top-2 margin
```

4. 如果 baseline-style 结果明显好于 segmented-greedy 结果，后续长程生成应该优先修改推理策略：

```text
avoid greedy
allow EOS or softly control EOS
reduce segment length
use stochastic sampling
avoid forcing 90 latent in one visualization
```

5. 如果 baseline-style 结果仍然差，再回到数据和训练侧排查：

```text
caption-window alignment
end token loss weight
KL regularization strength
train-scope 是否过窄
synthetic transition quality
oracle token reconstruction quality
```

## 13. 本轮实验文件清单

```text
BVH:
stage1_artifacts/generated_visualization/seqwin_baseline_method/walk_turn_crouch_seed0.bvh
stage1_artifacts/generated_visualization/seqwin_baseline_method/run_jump_walk_seed1.bvh
stage1_artifacts/generated_visualization/seqwin_baseline_method/side_wave_sit_seed2.bvh

MP4:
stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/walk_turn_crouch_seed0.mp4
stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/run_jump_walk_seed1.mp4
stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/side_wave_sit_seed2.mp4

Summary:
STAGE1_BASELINE_METHOD_VISUALIZATION_SUMMARY.md
```
