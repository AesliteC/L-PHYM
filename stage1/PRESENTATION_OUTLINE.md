# 基于 MoConVQ 的长程文本驱动动作生成方法探索

本文档是项目展示用的逐页 PPT 大纲，适合 12-15 分钟课程 pre。主线是：

```text
问题背景 -> 方法设计 -> 工程实现 -> 实验结果 -> 当前问题 -> 后续计划
```

推荐标题：

```text
基于 MoConVQ 的长程文本驱动动作生成方法探索
HumanML3D Long-Horizon Text-to-Motion Fine-tuning on MoConVQ
```

---

## 第 1 页：标题页

### 标题

基于 MoConVQ 的长程文本驱动动作生成方法探索

### 页面内容

- 课程项目 / Stage1 实验
- 姓名、学号、日期
- 关键词：
  - Text-to-Motion
  - HumanML3D
  - MoConVQ
  - Motion Token
  - Long-Horizon Generation

### 视觉建议

放一张动作生成结果截图，或者使用生成视频中的一帧，例如：

```text
walk -> turn -> crouch
```

可以选择并排可视化视频中的截图，左侧为 fine-tuned 模型，右侧为 baseline。

---

## 第 2 页：研究背景

### 标题

从文本到人体动作生成

### 页面内容

Text-to-Motion 的目标是根据自然语言描述生成人体动作序列。例如：

```text
Input:  "a person walks forward"
Output: corresponding human motion sequence
```

该任务可以应用于：

- 角色动画生成
- 虚拟人控制
- 游戏动作制作
- 机器人动作规划
- 人机交互与仿真

当前很多方法对短动作生成效果较好，例如：

```text
a person walks forward
a person sits down
a person jumps up
```

但真实应用中经常需要复合动作，例如：

```text
a person walks forward then turns around then crouches down
```

### 讲述重点

本项目关注的不是单个短动作，而是多个语义动作连续组成的长程动作生成。

---

## 第 3 页：问题定义与挑战

### 标题

长程动作生成的核心挑战

### 页面内容

长程 Text-to-Motion 不只是生成更长的视频或 BVH，而是要求长时间序列仍然满足：

```text
语义正确
动作顺序正确
过渡自然
生成稳定
```

主要挑战可以分成三类：

| 挑战 | 说明 |
|---|---|
| Semantic Challenge | 复合 prompt 中包含多个动作阶段，模型需要理解动作顺序 |
| Transition Challenge | 不同动作片段之间需要自然过渡 |
| Generation Challenge | 长时间 rollout 容易出现重复、停滞、提前结束或语义弱化 |

### 例子

输入：

```text
a person walks forward then turns around then crouches down
```

理想输出：

```text
walk forward -> turn around -> crouch down
```

常见失败情况：

```text
只生成 walking
动作中途停止
动作重复
turn / crouch 语义不明显
动作过渡不自然
```

### 讲述重点

长程动作生成的难点不只是长度，而是长序列中的语义、顺序和运动连续性。

---

## 第 4 页：基础模型 MoConVQ 简介

### 标题

MoConVQ：基于离散 Motion Token 的动作生成框架

### 页面内容

MoConVQ 将连续人体动作编码为离散 RVQ token，再通过 decoder / controller 还原为动作。

本项目使用的是 MoConVQ 原仓库中的 `Text2Motion_Transformer`。这里的 GPT 不是 HuggingFace 的 `CausalLM`，它预测的不是文字 token，而是 motion token。

### 简化结构

```text
Text Prompt
   ↓
T5 Text Encoder
   ↓
Text Feature
   ↓
Text2Motion Transformer
   ↓
RVQ Motion Tokens
   ↓
MoConVQ Decoder / Controller
   ↓
BVH / MP4
```

### 训练目标

模型输入：

```text
motion latent:  (B, T, 768)
RVQ indices:    (B, T, 4)
text features:  (B, L, 1024)
text mask:      (B, L)
clip feature:   (B, 512)
```

模型输出：

```text
per-frame RVQ logits
```

训练目标：

```text
predict 4-depth RVQ indices for each motion frame
```

### 讲述重点

本项目不是训练文本生成模型，而是微调一个文本条件的 motion-token GPT。

---

## 第 5 页：项目总体目标

### 标题

项目目标：构造长程训练数据并微调 Text2Motion Transformer

### 页面内容

本项目目标是将 HumanML3D 的短动作样本合成为长程动作-文本序列，并转换成 MoConVQ 可以训练的格式，最终微调 `Text2Motion_Transformer`。

### 总体 Pipeline

```text
HumanML3D short motion clips
  -> transition-filtered long motion-language sequences
  -> HumanML3D 22-joint motion
  -> MoConVQ 20-body state
  -> MoConVQ 323-d observation
  -> MoConVQ encoder latent + RVQ indices
  -> T5 text features
  -> fine-tune Text2Motion_Transformer
  -> generate BVH / render MP4 for evaluation
```

### 项目任务

- 构造长程 HumanML3D 动作序列
- 实现 HumanML3D 到 MoConVQ 格式转换
- 构建 GPT 训练 cache
- 微调 MoConVQ 文本条件 GPT
- 生成 BVH / MP4 并进行可视化对比

### 讲述重点

本项目的核心不是重新设计一个新模型，而是把已有的 MoConVQ 系统扩展到长程复合动作生成场景。

---

## 第 6 页：数据来源与预处理

### 标题

HumanML3D 数据集与短动作样本

### 页面内容

HumanML3D 提供文本描述和对应的人体动作序列。每个样本通常是一个较短动作片段。

项目中读取的数据包括：

```text
texts/
new_joints/
new_joint_vecs/
split files
```

当前验证过的数据规模：

```text
all:       29228
train:     23384
val:        1460
test:       4384
train_val: 24844
```

### 示例

```text
Caption: a person walks forward
Motion:  22-joint sequence
```

### 讲述重点

HumanML3D 原始样本主要是短动作，因此需要进一步合成长程复合动作数据。

---

## 第 7 页：长程动作序列合成方法

### 标题

从短 Clip 到长程 Sequence

### 页面内容

长程数据不是随机拼接得到的，而是通过 transition score 选择相对自然的动作连接。

合成流程：

```text
sample initial clip
  -> search candidate next clips
  -> compute transition score
  -> choose low-score transition
  -> align root position and yaw
  -> blend boundary frames
  -> concatenate captions with "then"
```

### Transition Score

transition score 主要考虑：

- 根关节位置差
- 根关节速度差
- 面向方向 yaw 差
- 脚部高度差
- 脚部速度差

### 拼接示意

```text
clip A: walk forward
        ↓ transition filtering + alignment
clip B: turn around
        ↓ transition filtering + alignment
clip C: crouch down

caption:
a person walks forward then turns around then crouches down
```

### 讲述重点

使用过渡约束可以减少不自然拼接，使合成长序列更适合作为训练数据。

---

## 第 8 页：长程数据统计

### 标题

合成数据集统计

### 页面内容

训练集统计：

```text
sequences:            1000
avg clips/sequence:   2.945
avg frames/sequence:  416.593
transitions:          1945
forced transitions:   0
failed:               3
```

验证集统计：

```text
sequences:            200
avg clips/sequence:   2.99
avg frames/sequence:  410.2
transitions:          398
forced transitions:   0
failed:               1
```

### 数据质量判断

- 每条序列平均接近 3 个 clip，可以覆盖复合动作。
- `forced transitions = 0`，说明没有强制插入不可控过渡点。
- 当前数据可以作为 Stage1 长程训练的基础数据集。

### 视觉建议

可以用表格或柱状图展示：

- sequence 数量
- 平均 clip 数
- 平均 frames 数
- transition 数量

### 讲述重点

这版数据具备长程复合动作的基本结构，同时避免了明显质量差的强制拼接。

---

## 第 9 页：MoConVQ 训练 Cache 构建

### 标题

将长动作转换为 GPT 可训练格式

### 页面内容

MoConVQ 的 Text2Motion Transformer 不能直接使用 HumanML3D 原始 joints，需要先转换为 MoConVQ 内部格式。

转换流程：

```text
HumanML3D joints:       (T, 22, 3)
  -> MoConVQ state:     (T, 20, 13)
  -> observation:       (T, 323)
  -> encoder latent:    (T_latent, 768)
  -> RVQ indices:       (T_latent, 4)
```

文本侧使用 T5 编码：

```text
text_features: (L, 1024)
text_mask:     (L)
```

### Cache 字段

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

### 讲述重点

微调时真正预测的是每个 latent step 对应的 4 层 RVQ codebook token，而不是直接回归人体关节坐标。

---

## 第 10 页：Sequence-Window 训练策略

### 标题

让训练窗口跨越动作边界

### 原始问题

如果训练窗口只来自单个 clip，模型看到的大多是：

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

这样模型缺少学习动作过渡的样本。

### 当前策略

本项目将 Stage1 cache 默认窗口策略改为：

```text
window_policy = "sequence"
```

含义：

- `sequence`：在整条合成长序列上滑动窗口，允许窗口跨越 clip 边界。
- `clip`：只在单个 clip 内取窗口，保留旧行为作为对照。

窗口配置：

```text
window size:   50
window stride: 25
caption mode:  window
```

### Cache 统计

训练 cache：

```text
windows:           3590
cross-boundary:    395 / 3590 = 11.0%
unique captions:   2470
failures:          0
```

验证 cache：

```text
windows:           707
cross-boundary:    99 / 707 = 14.0%
unique captions:   467
failures:          0
```

### 讲述重点

sequence-window 的核心作用是让模型真正看到动作之间的过渡，而不是只看到孤立动作。

---

## 第 11 页：模型微调设置

### 标题

Text2Motion Transformer 微调配置

### 页面内容

微调模型：

```text
MoConVQ Text2Motion_Transformer
```

初始化权重：

```text
text_generation_GPT.pth
```

训练目标：

```text
predict per-frame 4-depth RVQ token indices
```

训练输入：

```text
motion latent:  (B, T, 768)
RVQ indices:    (B, T, 4)
text features:  (B, L, 1024)
text mask:      (B, L)
```

### 训练配置

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
gpu:                0
```

### Objective 修正

训练使用 corrected autoregressive objective：

```text
previous motion latent context -> current RVQ indices
```

这样比直接用当前 latent 预测当前 token 更接近真实推理过程。

### 讲述重点

我们尽量让训练过程接近推理过程，使用历史 motion latent 上下文预测当前 token。

---

## 第 12 页：训练结果

### 标题

Token-Level 指标显示模型能够学习长程数据

### 最佳验证结果

最佳验证结果出现在 epoch 18：

```text
epoch:    18
val loss: 2.6115
val CE:   2.3509
val acc:  38.42%
```

### 最后一轮结果

epoch 19：

```text
train loss: 2.4253
train CE:   2.1439
train acc:  41.36%

val loss:   2.6130
val CE:     2.3555
val acc:    38.31%
```

按 RVQ depth 的 token accuracy：

```text
depth0: 35.76%
depth1: 48.92%
depth2: 37.71%
depth3: 30.85%
```

### 训练趋势

```text
val loss: 约 3.17 -> 约 2.61
val acc:  约 30.82% -> 约 38.3%
```

### 结论

- 模型可以在 sequence-window cache 上稳定收敛。
- 验证 token accuracy 有明显提升。
- best checkpoint 可以作为当前长程 Stage1 的新基线。
- 但 token-level 指标不等价于最终动作质量。

### 视觉建议

如果有训练日志图，可以放：

- train / val loss curve
- train / val accuracy curve

如果没有图，可以直接用数字变化：

```text
30.82% -> 38.42%
```

---

## 第 13 页：生成与可视化实验

### 标题

从文本生成 BVH / MP4

### 测试 Prompt

共测试 3 条复合长 prompt：

```text
1. a person walks forward then turns around then crouches down

2. a person runs forward then jumps up then walks slowly

3. a person walks sideways then waves both arms then sits down
```

### 对比模型

```text
baseline_original:
原始 pretrained GPT

seqwin_best:
sequence-window fine-tuned GPT
```

### 输出形式

- BVH 文件
- MP4 渲染视频
- 左右并排对比视频

正式长程可视化结果：

```text
frames:   2160
fps:      120
duration: 18.0 seconds
```

### 生成结果位置

```text
stage1_artifacts/generated_visualization/videos/compare/
```

并排视频：

```text
walk_turn_crouch_seqwin_vs_baseline.mp4
run_jump_walk_seqwin_vs_baseline.mp4
side_wave_sit_seqwin_vs_baseline.mp4
```

### 视觉建议

这一页建议放视频截图，并在现场播放其中一个视频片段。

---

## 第 14 页：可视化结果与现象

### 标题

模型改进与当前问题

### What Works

- 端到端生成链路已经跑通。
- 模型可以从长程文本 prompt 生成 BVH。
- Fine-tuned 模型比原始 baseline 更容易生成到目标长度。
- Sequence-window 训练提升了 token-level 指标。
- BVH 和 MP4 渲染链路可用。

### What Remains Hard

- 动作质量仍不稳定。
- 长程生成中仍可能出现重复或停滞。
- 后半段动作语义可能弱化。
- 动作阶段顺序不一定完全清晰。
- token accuracy 提升不一定能直接转化为 motion-level 质量提升。

### 讲述重点

当前最重要的结论不是模型已经完美解决长程生成，而是端到端链路已经打通，并且实验定位出了后续优化方向。

---

## 第 15 页：推理策略消融实验

### 标题

视觉效果受生成策略影响明显

### 背景

之前长程可视化使用的是强制长程生成策略：

```text
generation mode: segmented
sampling:        greedy
EOS:             suppress EOS
latent length:   90 for 3-segment prompts
duration:        18.0 seconds
```

该策略和原版 MoConVQ baseline 推理方式差异较大，可能放大 token 重复和分布偏移问题。

### Baseline-Style 推理

后续消融实验使用更接近原版 baseline 的推理方式：

```text
generation mode:   rolling
prompt:            whole prompt once
sampling:          top-k categorical sampling
EOS:               allow EOS
latent length:     up to 49
duration:          9.8 seconds
```

### 对比表

| 项目 | 强制长程可视化 | Baseline-style 可视化 |
|---|---|---|
| prompt 使用方式 | 按 `"then"` 切成多段 | 整句 prompt 一次输入 |
| generation mode | segmented | rolling |
| sampling | greedy / top1 | top-k categorical |
| EOS | suppress EOS | allow EOS |
| latent 长度 | 90 | 49 |
| 视频时长 | 18.0s | 9.8s |
| 主要用途 | 测试强制长程复合生成 | 推理策略消融 |

### 实验意义

该实验用于区分：

```text
模型本身是否不可用
```

和：

```text
长程推理策略是否过于激进
```

### 讲述重点

生成策略本身会显著影响视觉效果，因此不能只根据一个推理配置下的视频判断模型质量。

---

## 第 16 页：当前项目贡献总结

### 标题

项目完成内容

### 数据侧

- 构建 HumanML3D 长程动作-文本序列。
- 设计 transition score 过滤不自然拼接。
- 实现 root position / yaw 对齐和边界平滑。

### 工程侧

- 完成 HumanML3D 到 MoConVQ observation 的转换。
- 完成 MoConVQ encoder latent 与 RVQ token cache 构建。
- 集成 T5 text feature cache。

### 训练侧

- 微调 MoConVQ Text2Motion Transformer。
- 实现 sequence-window 跨边界训练。
- 记录 train / val loss、token accuracy 和 per-depth accuracy。

### 评估侧

- 实现文本到 BVH 的生成。
- 实现 BVH 到 MP4 的渲染。
- 完成 baseline 与 fine-tuned 模型的可视化对比。
- 分析推理策略对视觉效果的影响。

### 讲述重点

这个项目的主要贡献是把 MoConVQ 从短动作生成扩展到长程复合动作生成的实验链路，并完成第一轮训练与可视化诊断。

---

## 第 17 页：不足与后续工作

### 标题

Limitations and Future Work

### 当前不足

- 跨边界窗口比例只有约 11%-14%，过渡样本仍然偏少。
- 合成 caption 和局部 motion window 之间仍可能存在噪声。
- token-level accuracy 不能充分反映 motion-level 质量。
- 长程推理时仍容易重复、停滞或语义漂移。
- 目前还缺少完整的 2x2 对照实验。

### 后续工作

1. Boundary-centered oversampling

   提高跨边界训练窗口比例，例如提升到 30%-50%，让模型看到更多动作过渡样本。

2. Better text-motion alignment

   让每个训练 window 使用更精确的局部 caption，减少文本和动作窗口不匹配的问题。

3. Better decoding strategy

   减少 greedy decoding，使用 stochastic sampling、temperature 或 entropy control。

4. More complete evaluation

   补齐 original checkpoint + baseline-style inference，形成完整 2x2 对照。

5. Motion-level metrics

   加入物理合理性、轨迹平滑性、动作多样性和语义一致性评估。

### 讲述重点

后续优化重点不只是继续训练更久，而是提升数据质量、窗口对齐和推理策略。

---

## 第 18 页：总结页

### 标题

Conclusion

### 页面内容

本项目完成了：

```text
HumanML3D 长程动作数据构造
MoConVQ token cache 构建
Text2Motion Transformer 微调
BVH / MP4 可视化评估
```

核心结论：

1. 端到端链路已经成功跑通。

2. Sequence-window 训练使模型能够学习跨动作边界的 token 预测。

3. 验证集 token accuracy 从约 30.82% 提升到 38.42%。

4. 当前生成质量仍有提升空间，后续应重点优化跨边界数据增强、文本-动作窗口对齐和长程推理策略。

可以用下面这句话收尾：

```text
End-to-end pipeline works, but high-quality long-horizon motion generation remains challenging.
```

---

## 12 页压缩版本

如果展示时间较短，可以压缩为 12 页：

1. 标题
2. 背景与挑战
3. MoConVQ 简介
4. 项目目标与 pipeline
5. 长程数据合成
6. Cache 构建
7. Sequence-window 策略
8. 微调设置
9. 训练结果
10. 可视化结果
11. 问题分析与推理策略消融
12. 总结与未来工作

---

## 现场展示建议

建议至少展示 1-2 个视频，不要只展示表格。

可以优先使用：

```text
stage1_artifacts/generated_visualization/videos/compare/walk_turn_crouch_seqwin_vs_baseline.mp4
stage1_artifacts/generated_visualization/videos/seqwin_baseline_method/walk_turn_crouch_seed0.mp4
```

讲述时建议保持谨慎，不要说“效果已经很好”，而是说：

```text
我们已经验证了长程数据构造和微调链路是可行的；
token-level 指标有提升；
但 motion-level 质量还没有完全解决；
下一步需要围绕数据边界、文本对齐和推理策略继续优化。
```

这个表述最符合当前项目状态。

---

## 推荐汇报话术

可以用以下逻辑串联整场 pre：

```text
这个项目关注长程 Text-to-Motion。
现有模型更擅长短动作，但复合 prompt 需要模型生成多个连续动作阶段。

我们基于 MoConVQ 做扩展。
MoConVQ 使用离散 RVQ motion token，因此我们的训练目标不是预测文本 token，
而是根据文本特征和历史 motion latent 预测每一帧的 motion token。

为了让模型学习长程动作，我们从 HumanML3D 的短动作样本出发，
通过 transition score 选择可以自然拼接的 clip，
构造平均包含约 3 个动作片段的长程序列。

随后我们把 HumanML3D motion 转换成 MoConVQ observation，
通过 MoConVQ encoder 得到 latent 和 RVQ indices，
再结合 T5 text features 构建 GPT 训练 cache。

训练时我们使用 sequence-window 策略，
允许窗口跨越动作边界，
从而让模型能看到 walk-to-turn、turn-to-crouch 这样的过渡样本。

实验结果显示，模型在 token-level 上能够稳定收敛，
验证 token accuracy 从约 30.82% 提升到 38.42%。

不过可视化结果也说明，token 指标提升不等于动作质量完全解决。
长程生成仍然存在重复、停滞、语义弱化等问题。
进一步消融发现，推理策略本身也会明显影响视觉质量。

因此当前结论是：端到端链路已经跑通，长程训练是有效的，
但高质量长程动作生成仍需要继续优化数据边界、文本对齐和解码策略。
```
