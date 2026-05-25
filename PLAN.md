# Stage1 真实实验数据合成与 MoConVQ-GPT 微调 Plan

## 1. 总体目标

构建一条真实实验链路：从 `HumanML3D/HumanML3D` 读取短动作-文本样本，按“过渡约束”合成长动作序列，保存成本地可复现实验数据集；再把长序列转换成 MoConVQ-GPT 可训练的 `latent_vq + RVQ indices + T5 text feature` cache；最后微调 `MoConVQ` 仓库里的 `Text2Motion_Transformer`，初始化权重来自 `text_generation_GPT.pth`。

关键原则：

- 不再把 HumanML3D 的 263-d vector 直接硬拼/平铺成 768-d latent 作为真实实验主线。
- 严格对齐路线采用：`HumanML3D joints -> MoConVQ 20-body state -> MoConVQ 323-d observation -> agent.encode_seq_all() -> latent_vq/indexs`。
- 文本特征采用原仓库一致的 `T5Tokenizer + T5EncoderModel`，默认 `t5-large`，不使用 hash text feature 作为真实实验输入。
- 所有生成产物放在 `MoConVQ/stage1_artifacts/` 下，避免污染仓库主数据。

## 2. 数据合成脚本

新增或重写 Stage1 数据合成入口，建议命名为：

- `Script/stage1/synthesize_long_humanml3d.py`

输入来自：

- `../HumanML3D/HumanML3D/all.txt`
- `../HumanML3D/HumanML3D/train.txt`
- `../HumanML3D/HumanML3D/val.txt`
- `../HumanML3D/HumanML3D/test.txt`
- `../HumanML3D/HumanML3D/new_joints/*.npy`
- `../HumanML3D/HumanML3D/new_joint_vecs/*.npy`
- `../HumanML3D/HumanML3D/texts/*.txt`
- `../HumanML3D/index.csv`

命令行参数：

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

合成逻辑：

- 从指定 split 中采样第一个 clip。
- 后续每个 clip 不直接随机拼，而是在 `candidate-pool` 个候选里计算 transition score。
- transition score 使用 `new_joints` 的边界帧计算：
  - 根关节末帧/首帧位置差；
  - 根关节速度差；
  - 面向方向差，使用左右髋与肩部估计 yaw；
  - 脚部高度和脚部速度差；
  - 若所有候选超过阈值，选 score 最低者并记录 `transition_forced=true`。
- 拼接时对后一个 clip 做平移和 yaw 对齐，使首帧根位置/朝向接近前一个 clip 的末帧。
- `blend-frames` 用于边界处根平移/yaw 的短过渡平滑，不改原 clip 内部动作主体。
- 文本取每个 clip 的第一条 caption，用 `caption-joiner` 拼成长文本；同时保留每段 `clip_captions` 和边界帧位置，便于后续 debug。

输出格式：

- `manifest.jsonl`
  - 每行一个长序列；
  - 字段包括 `sequence_id`, `split`, `sample_ids`, `caption`, `clip_captions`, `clip_boundaries`, `transition_scores`, `source_paths`, `start_frames`, `end_frames`。
- `long_sequences.h5`
  - 每个 group 是一个 `sequence_id`；
  - dataset 包括 `joints_22`, `joint_vecs_263`, `clip_boundaries`, `transition_scores`；
  - attrs 包括 `caption`, `sample_ids`, `split`。
- `summary.json`
  - 记录总条数、平均 clip 数、平均帧数、失败/强制过渡数量、参数快照。

## 3. MoConVQ-GPT 训练 cache 构建

新增真实实验 cache 构建脚本，建议命名为：

- `Script/stage1/build_real_moconvq_gpt_cache.py`

命令示例：

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

转换流程：

- 读取每条长序列的 `joints_22`。
- 使用 HumanML3D 的 22-joint skeleton 定义和 MoConVQ `world.json` 的 20-body body order，构造确定性 retarget mapping。
- 生成 MoConVQ state：
  - body position 来自映射后的 HumanML3D joints；
  - body rotation 通过父子骨向量、左右肩/髋估计局部坐标系，再转 quaternion；
  - linear velocity 用相邻帧位置差分；
  - angular velocity 用相邻帧 quaternion 差分；
  - state shape 必须是 `(T, 20, 13)`。
- 调用 MoConVQ 的 `state2ob()` 得到 observation，shape 必须是 `(T, 323)`。
- 用 `build_agent(gpu)` 加载 `moconvq_base.data`，调用：
  - `agent.encode_seq_all(None, observation)`
- 从返回值取：
  - `latent_vq`: `(1, T_latent, 768)`；
  - `indexs`: 原始 RVQ indices，转成训练需要的 `(T_latent, 4)`。
- 使用 T5 编码长文本：
  - `text_features`: `(text_len, 1024)`；
  - `text_masks`: `(text_len,)`，语义与原 `text2motion_generation.py` 一致。
- 将长序列切成训练窗口：
  - 默认 `window-size=50`，匹配 GPT `block_size=52`；
  - 默认 `window-stride=25`；
  - 每个 window 共享该长序列 caption/T5 feature，并记录 `sequence_id`, `window_start`, `window_end`。
- 保存 `.pt` cache，字段固定为：
  - `latents`: `(N, 50, 768)`
  - `indices`: `(N, 50, 4)`
  - `text_features`: `(N, L, 1024)`
  - `text_masks`: `(N, L)`
  - `captions`: `list[str]`
  - `sequence_ids`: `list[str]`
  - `window_ranges`: `list[tuple[int, int]]`
  - `sample_ids`: `list[list[str]]`
  - `config`: 参数快照

失败处理：

- 单条序列 retarget、observation、encoder、T5 任一步失败，不中断全局构建；
- 写入 `failure-log`，记录 `sequence_id`, `reason`, `traceback_short`；
- 若失败率超过 `--max-failure-rate`，脚本以非零退出码结束；
- cache 构建结束时打印成功 window 数、失败序列数、平均 token 长度、indices 最大/最小值。

## 4. 微调脚本

保留现有 `Text2Motion_Transformer` 架构，但整理成真实实验训练入口，建议命名为：

- `Script/stage1/train_real_text_gpt.py`

命令示例：

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

训练逻辑：

- 构建 MoConVQ GPT：
  - `num_vq=512`
  - `embed_dim=768`
  - `clip_dim=512`
  - `block_size=52`
  - `num_layers=9`
  - `n_head=8`
  - `fc_rate=2`
- 从 `moconvq_base.data` 读取 RVQ codebook embedding，并拼接两个 zero special embeddings，保持原仓库 GPT 初始化习惯。
- 加载 `text_generation_GPT.pth`，兼容 `module.` 前缀。
- batch 输入：
  - `latent`: `(B, 50, 768)`
  - `indices`: `(B, 50, 4)`
  - `clip_feature`: `(B, 512)`，默认全零，和原 `text2motion_generation.py` 一致；
  - `text_feature`: `(B, L, 1024)`
  - `text_mask`: `(B, L)`
- forward 后取：
  - `logits[:, :, 1:, :]` 对应 4 层 RVQ token；
  - 用 `F.cross_entropy` 预测 `indices`；
  - ignore 掉 padding/special token，如果 cache 中未来加入 padding，则使用 `ignore_index=513`。
- 输出：
  - `checkpoint_epoch_{k}.pth`
  - `best_val.pth`
  - `last.pth`
  - `train_log.jsonl`
  - `config.json`
- 每个 epoch 记录：
  - train loss；
  - val loss；
  - token accuracy；
  - per-depth accuracy；
  - learning rate；
  - elapsed time。

## 5. 测试与验收

最低验收命令：

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
cd /home/chenjie/cc/robotics/MoConVQ
python -m unittest tests.test_stage1_humanml3d tests.test_stage1_real_cache tests.test_stage1_real_train -v
```

测试覆盖：

- HumanML3D catalog：
  - split 数量正确；
  - 每个 sample id 能找到 `text/new_joints/new_joint_vecs`。
- 长序列合成：
  - `--num-sequences 3 --min-clips 2 --max-clips 2` 产出 3 条；
  - 每条 manifest 有 2 个 sample；
  - `long_sequences.h5` group 数、caption、clip boundaries 正确；
  - transition score 可复现，同 seed 输出一致。
- MoConVQ observation/cache：
  - 小样本 1 条序列能生成 `(T, 20, 13)` state；
  - `state2ob()` 输出最后一维为 323；
  - `agent.encode_seq_all()` 能返回 finite `latent_vq` 和合法 `indices`；
  - cache 中 `latents.shape[-1] == 768`，`indices.shape[-1] == 4`。
- T5 文本特征：
  - 同一 caption 重复编码结果 shape 一致；
  - mask dtype 为 bool；
  - feature dim 为 1024。
- 训练 smoke：
  - 用 1-2 条序列构造 cache；
  - 跑 `--epochs 1 --batch-size 1 --smoke`；
  - 能完成 forward/backward/save；
  - 输出 `last.pth` 和 `train_log.jsonl`。

默认假设：

- 第一版严格实验不使用旧的 `lift_motion_vec_to_latent()` 作为主路径，只可保留为 debug fallback。
- `moconvq` 环境用于所有命令。
- `t5-large` 若本地未缓存，允许运行时从 HuggingFace 下载；如果下载失败，脚本应明确报错，不自动退回 hash encoder。
- 训练窗口长度默认 50，因为现有 MoConVQ GPT 的 `block_size=52`。
- 长序列数据用于构造更多长语义条件下的 50-frame 训练窗口；真正超长生成可在下一阶段用 rolling context / chunked generation 单独实现。
