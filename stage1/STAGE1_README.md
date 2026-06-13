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

但当前结果还不能作为“效果好”的结论。最新生成对比中，微调模型能比 baseline 更稳定地生成长序列，不容易提前输出 end token；不过从实际 BVH 视觉效果看，动作质量和文本语义对齐仍然不理想，存在待解决问题。当前阶段的结论应写成：

```text
测试链路已成功跑通，但生成效果仍不好，后续重点是定位并修复数据转换、retarget、训练目标和长文本条件对齐问题。
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

窗口长度默认是 50，因为 `Text2Motion_Transformer` 的 `block_size=52` 会在 motion latent 前额外加入一个 condition token；cache 构建脚本会拒绝 `window-size > 51`。默认 `--max-text-length 256` 会把 T5 文本特征固定为 `(256, 1024)`，过长 caption 会按 T5 tokenizer 截断；正式实验推荐配合 `--caption-mode window`，让每个 50-token motion window 使用对应局部 caption。

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
  --gpu 0 \
  --output stage1_artifacts/gpt_cache/train_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/train_failures.jsonl
```

`--caption-mode sequence` 会把整条长序列 caption 复制给每个 window；`--caption-mode window` 会根据 `clip_boundaries` 给每个训练 window 选择重叠 clip 的局部 caption。真实长序列实验更推荐 `window`，因为它减少“当前动作窗口和整段长文本不对应”的噪声。

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
- 已完成一次 `train_real_text_gpt.py` 微调，输出 checkpoint：
  - `stage1_artifacts/checkpoints/real_stage1/checkpoint_epoch_5.pth`
  - `stage1_artifacts/checkpoints/real_stage1/checkpoint_epoch_10.pth`
  - `stage1_artifacts/checkpoints/real_stage1/checkpoint_epoch_15.pth`
  - `stage1_artifacts/checkpoints/real_stage1/checkpoint_epoch_20.pth`
  - `stage1_artifacts/checkpoints/real_stage1/best_val.pth`
  - `stage1_artifacts/checkpoints/real_stage1/last.pth`
- 已使用 `checkpoint_epoch_5.pth` 和 baseline `text_generation_GPT.pth` 生成 BVH 对比：
  - `stage1_artifacts/generated_bvh_compare/real_stage1_epoch5_vs_baseline/`

`checkpoint_epoch_5.pth` 的日志指标：

```text
train loss:       0.09024
train token acc:  0.98169
val loss:         0.03926
val token acc:    0.99300
val depth acc:    0.99965 / 0.99044 / 0.99233 / 0.98957
```

这说明训练脚本能够拟合 cache 中的 RVQ token 预测任务，但它不等价于最终动作质量好。动作生成质量仍需要通过 BVH 视觉检查、物理合理性和文本语义一致性评估。

已生成的 epoch5 对比中，baseline 在长文本 prompt 上经常提前结束，而 epoch5 微调模型基本能生成到目标长度：

```text
walk_turn_return:      baseline 696 frames,  epoch5 2880 frames
walk_run_jump:         baseline 936 frames,  epoch5 2880 frames
circle_wave_crouch:    baseline 1176 frames, epoch5 2880 frames
sidestep_kick_turn:    baseline 408 frames,  epoch5 2880 frames
long_sequence_mixed:   baseline 1176 frames, epoch5 2880 frames
```

这只能说明微调后模型更愿意生成长序列，不能说明动作语义和运动质量已经合格。目前实际观察结论是：链路成功，但效果不好，问题待解决。

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
Script/stage1/build_real_moconvq_gpt_cache.py
Script/stage1/train_real_text_gpt.py
```

## 6. 还需要完成的工作

### 6.1 当前效果问题定位

当前最大问题不是脚本跑不通，而是生成结果质量不好。后续应优先定位以下问题：

- HumanML3D 22-joint skeleton 到 MoConVQ 20-body state 的 retarget 是否足够准确；
- `state2ob()` 之后再经过 `agent.encode_seq_all()` 得到的 latent/RVQ indices 是否能被 MoConVQ decoder 稳定还原；
- 合成长序列的拼接边界是否引入不自然速度、朝向或脚部状态突变；
- `--caption-mode window` 下每个 50-token motion window 对应的局部 caption 是否真正匹配该窗口动作；
- GPT 只在 50-token window 上训练，而推理时滚动生成更长序列，是否出现分布外累积误差；
- 微调 loss 很低但生成质量差，可能说明模型主要学到 token 分布/结束 token 行为，而没有真正提升长语义控制；
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

- 对比 baseline、epoch5、best_val、last 的同一组 prompt；
- 检查 finetuned 模型是否只是避免早停，但动作内容重复或语义不对；
- 对生成 BVH 做逐 prompt 人工记录，例如“是否转身”“是否跳跃”“是否蹲下”“是否明显脚滑”；
- 尝试不同 `--chunk-size`、`--context-size`、`--temperature`、`--top-k`，避免 greedy decoding 固化坏模式；
- 将训练 cache 中的若干 window 反解成 BVH，检查训练目标本身是否可信；
- 若 retarget 问题明显，优先重做 HumanML3D 到 MoConVQ character 的转换，而不是继续调 GPT。

### 6.3 数据规模和训练配置

当前已经完成一次真实链路，但还需要系统复现实验。建议按规模逐步扩大和记录：

```text
10 sequences -> 100 sequences -> 当前规模 -> 更大规模
```

每一步确认：

- failure log 是否为空或可接受；
- cache 的 window 数是否合理；
- indices 是否在合法范围内；
- 训练 loss 是否下降。

同时不要只看 token accuracy。token accuracy 高但 BVH 差时，应优先检查数据转换和生成策略。

### 6.4 T5 模型下载和缓存

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

### 6.5 Retarget 质量检查

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

### 6.6 Val cache 和评估指标

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

### 6.7 长动作生成与展示

当前 `generate_long_motion.py` 可用于训练后生成 BVH。它默认使用 `T5Tokenizer + T5EncoderModel`，和真实 cache 构建路径一致；如果只想离线调试文本形状，可以显式传 `--text-encoder hash`。

生成脚本已经支持 fixed-context rolling generation：`--max-length` 控制总 latent token 数，`--context-size` 控制每个 chunk 最多回看多少历史 latent，`--chunk-size` 控制每次新采样多少 token。由于 GPT 的 `block_size=52` 还包含一个 condition token，每轮实际历史长度会自动裁剪到 `51 - 当前chunk长度`，避免超过 position/mask 长度。文本侧仍由 `--max-text-length` 控制，默认 256，超长 prompt 会被 T5 tokenizer 截断。

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


## 7. 当前状态一句话总结

Stage1 的代码框架、长序列合成、真实 MoConVQ encoder cache 构建、GPT 微调入口、滚动生成入口、测试和一次真实训练/生成链路都已经跑通；但当前 BVH 视觉效果不好，不能作为最终实验效果，下一步应重点排查 retarget/cache 质量、长文本-window 对齐、生成策略和 GPT 微调目标之间的问题。
