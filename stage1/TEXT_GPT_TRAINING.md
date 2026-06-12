# Text2Motion GPT 训练代码使用说明

本文档说明如何训练 `text_generation_GPT.pth` 对应的 MoConVQ 文本到动作生成模型。这里的 GPT 不是 HuggingFace `CausalLM`，而是仓库内自定义的 `Text2Motion_Transformer`：

```text
text features + motion latent prefix
  -> Text2Motion_Transformer
  -> 每帧 4 层 RVQ code logits
  -> RVQ codebook embedding 求和
  -> 768 维 motion latent
  -> MoConVQ decoder / controller 输出动作
```

核心训练目标不是预测文本 token，而是预测每个 motion frame 的 4 个 RVQ token。

## 1. 相关文件

| 文件 | 作用 |
| --- | --- |
| `MoConVQCore/Model/cross_trans_ori_fixsum.py` | 定义 `Text2Motion_Transformer`、temporal cross-attention、RVQ token transformer 和分类头 |
| `Script/stage1/train_text_gpt.py` | 旧版/轻量训练入口；使用已经做好的 cache，指标较少，仅建议 smoke/debug |
| `Script/stage1/train_real_text_gpt.py` | 推荐训练入口；支持真实 MoConVQ cache、padding ignore、train/val 指标、checkpoint 保存 |
| `Script/stage1/real_moconvq_cache.py` | 从合成长序列 H5 构建真实训练 cache |
| `Script/stage1/convert_humanml3d_to_moconvq_observation.py` | 独立导出 MoConVQ `state_20x13` 和 `observation_323`，用于 retarget 检查 |
| `Script/stage1/build_real_moconvq_gpt_cache.py` | `real_moconvq_cache.py` 的命令行入口 |
| `Script/stage1/synthesize_long_humanml3d.py` | 从 HumanML3D 短 clip 合成长 motion-language 序列 |
| `Script/stage1/generate_long_motion.py` | 使用训练后的 checkpoint 从文本生成 BVH |
| `Script/text2motion_generation.py` | 原始推理脚本，使用 `text_generation_GPT.pth` + T5 + MoConVQ decoder |

推荐主线是：

```text
HumanML3D
  -> synthesize_long_humanml3d.py
  -> convert_humanml3d_to_moconvq_observation.py  # optional inspection
  -> build_real_moconvq_gpt_cache.py
  -> train_real_text_gpt.py
  -> generate_long_motion.py
```

## 2. 运行环境

从仓库根目录运行命令：

```bash
cd /home/chenjie/cc/robotics/MoConVQ
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
```

训练前需要确认以下文件存在：

```text
moconvq_base.data          # MoConVQ 基础模型，提供 encoder/decoder 和 RVQ codebook
text_generation_GPT.pth    # 文本 GPT 初始化 checkpoint
../HumanML3D/HumanML3D     # HumanML3D 数据根目录，若要重新构建 cache
```

检查：

```bash
ls -lh moconvq_base.data text_generation_GPT.pth
ls ../HumanML3D/HumanML3D
```

## 3. 模型输入输出

训练脚本构建的模型来自：

```python
from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer
```

配置在 `Script/stage1/train_text_gpt.py`：

```python
num_vq = 512
embed_dim = 768
clip_dim = 512
block_size = 52
num_layers = 9
n_head = 8
drop_out_rate = 0.1
fc_rate = 2
```

注意：源码里 `num_layers=9` 没有直接控制总层数。实际结构硬编码为：

```text
trans_temporal: 12 层 CrossBlock
trans_base:      4 层 Block
trans_head:      1 层 Block
```

训练时 `forward()` 的输入：

```text
latents:      (B, T, 768)   # MoConVQ encoder 得到的 latent_vq
indices:      (B, T, 4)     # 每帧 4 层 RVQ token
clip_feature: (B, 512)      # 当前训练脚本中默认为全 0
text_feature: (B, L, 1024)  # 文本编码特征，推荐 T5-large
text_mask:    (B, L)        # True 表示 padding / masked token
```

输出：

```text
logits:        (B, T, 5, 513)
projected:     (B, T, 768)
```

训练只使用：

```python
context_latents = latents[:, :-1, :]
logits, _ = model(context_latents, indices, clip_feature, text_feature, text_mask)
rvq_logits = logits[:, :, :4, :]
```

也就是用 `condition, latent[0], ..., latent[T-2]` 预测 `indices[0], ..., indices[T-1]`，并使用前 4 个 depth slot 对齐 4 层 RVQ token。不要使用旧写法 `logits[:, :, 1:, :]`；旧写法会让 RVQ depth 发生 teacher-forcing 泄漏，导致 loss/accuracy 虚高、推理时 token collapse。

类别数 `513` 的含义：

```text
0..511: 512 个 RVQ code
512:    结束/特殊 token
513:    padding ignore index，只出现在 target indices 中，不是模型输出类别
```

## 4. 数据 cache 格式

`train_real_text_gpt.py` 期望 `--train-cache` 和 `--val-cache` 是 `torch.save()` 出来的 dict，至少包含：

```text
latents:       Tensor/array, shape (N, T, 768), float32
indices:       Tensor/array, shape (N, T, 4), int64
text_features: Tensor/array, shape (N, L, 1024), float32
text_masks:    Tensor/array, shape (N, L), bool
captions:      list[str]
```

真实 cache 还会包含：

```text
sequence_ids:  list[str]
window_ranges: list[tuple[int, int]]
sample_ids:    list[list[str]]
config:        dict
```

默认 window 设置：

```text
window-size:   50
window-stride: 25
rvq-depth:     4
pad_index:     513
max-text-length: 256
```

如果某个 motion 序列长度不足一个 window，cache 构建代码会补齐到 `window-size`，补齐位置的 `indices` 为 `513`。训练 loss 中会用 `ignore_index=513` 跳过这些位置。

`Text2Motion_Transformer` 的 `block_size=52` 会在 motion latent 前额外加入一个 condition token，所以训练 window 的 motion token 数最多是 51；脚本会拒绝 `--window-size > 51`。文本侧默认用 T5 tokenizer 固定到 `--max-text-length 256`，更长 caption 会被截断。真实长序列实验当前默认使用 `--caption-mode window`，让每个 motion window 只绑定与该窗口重叠的局部 clip caption，降低长 caption 截断和语义不匹配的影响。

快速检查 cache：

```bash
python - <<'PY'
import torch
path = "stage1_artifacts/gpt_cache/train_cache.pt"
cache = torch.load(path, map_location="cpu")
for key in ["latents", "indices", "text_features", "text_masks"]:
    value = cache[key]
    print(key, tuple(value.shape), value.dtype)
print("captions", len(cache["captions"]))
print("first caption:", cache["captions"][0] if cache["captions"] else None)
valid = cache["indices"] != 513
print("valid token count", int(valid.sum()))
if valid.any():
    print("index min/max", int(cache["indices"][valid].min()), int(cache["indices"][valid].max()))
PY
```

期望大致类似：

```text
latents (N, 50, 768) torch.float32
indices (N, 50, 4) torch.int64
text_features (N, 256, 1024) torch.float32
text_masks (N, 256) torch.bool
index min/max 0 511
```

## 5. 从 HumanML3D 构建训练数据

如果已经有 `train_cache.pt` / `val_cache.pt`，可以跳过本节，直接看第 6 节。

### 5.1 合成长 motion-language 序列

训练真实 cache 前，需要先把 HumanML3D 短 clip 合成长序列：

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

输出：

```text
stage1_artifacts/long_humanml3d/train/manifest.jsonl
stage1_artifacts/long_humanml3d/train/long_sequences.h5
stage1_artifacts/long_humanml3d/train/summary.json
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--humanml-root` | HumanML3D 根目录 |
| `--split` | 使用哪个 split，通常是 `train` 或 `val` |
| `--num-sequences` | 合成长序列数量 |
| `--min-clips` / `--max-clips` | 每条长序列拼接多少个短 clip |
| `--candidate-pool` | 为每次拼接采样多少候选 clip 计算过渡分数 |
| `--transition-max-score` | 默认拒绝超过该分数的 transition，避免坏边界进入训练 |
| `--allow-forced-transitions` | 显式保留超过阈值的 transition，仅用于复现旧数据或 debug |
| `--blend-frames` | clip 边界过渡平滑帧数 |
| `--caption-joiner` | 多个短 caption 拼接成长 caption 的连接词 |
| `--output-dir` | 输出目录 |

建议也构建 val：

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

### 5.2a 可选：单独导出 MoConVQ observation 检查

构建 GPT cache 前，可以先把合成序列转换为 MoConVQ state/observation，单独检查 retarget 结果：

```bash
python Script/stage1/convert_humanml3d_to_moconvq_observation.py \
  --long-h5 stage1_artifacts/long_humanml3d/train/long_sequences.h5 \
  --manifest stage1_artifacts/long_humanml3d/train/manifest.jsonl \
  --output-h5 stage1_artifacts/long_humanml3d/train/moconvq_observations.h5 \
  --summary stage1_artifacts/long_humanml3d/train/moconvq_observations_summary.json
```

输出包括 `state_20x13: (T, 20, 13)` 和 `observation_323: (T, 323)`。

### 5.2 构建真实 MoConVQ GPT cache

把合成出的 `long_sequences.h5` 转成训练 cache：

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
  --fps 20 \
  --max-text-length 256 \
  --max-failure-rate 0.1 \
  --output stage1_artifacts/gpt_cache/train_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/train_failures.jsonl
```

构建 val cache：

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
  --fps 20 \
  --max-text-length 256 \
  --max-failure-rate 0.1 \
  --output stage1_artifacts/gpt_cache/val_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/val_failures.jsonl
```

这个步骤内部做了几件事：

```text
HumanML3D joints_22
  -> MoConVQ 20-body state
  -> state2ob() 得到 323 维 observation
  -> 加载 moconvq_base.data
  -> agent.encode_seq_all(None, observation)
  -> latent_vq + RVQ indices
  -> T5-large 编码 caption
  -> 切成 window
  -> torch.save(cache)
```

输出 JSON 中需要关注：

```text
windows:          生成了多少训练窗口
failed_sequences: 转换失败序列数量
failure_rate:     失败比例
index_min/max:    RVQ token 是否在合理范围内
```

如果 `failure_rate > --max-failure-rate`，脚本会退出并报错。失败详情保存在 `--failure-log` 指定的 jsonl 文件中。

## 6. 推荐训练入口：train_real_text_gpt.py

### 6.1 smoke test

第一次运行建议先做单 batch smoke test：

```bash
python Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache/train_cache.pt \
  --val-cache stage1_artifacts/gpt_cache/val_cache.pt \
  --init-checkpoint text_generation_GPT.pth \
  --base-data moconvq_base.data \
  --output-dir stage1_artifacts/checkpoints/real_stage1_smoke \
  --epochs 1 \
  --batch-size 2 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --gpu 0 \
  --seed 0 \
  --save-every 1 \
  --num-workers 0 \
  --smoke
```

smoke 模式中：

```text
只跑 1 个 train batch
如果提供 val cache，也只跑 1 个 val batch
训练后保存 last.pth 和 checkpoint_epoch_1.pth
```

如果 smoke test 都跑不通，不要直接开正式训练，先看第 11 节排错。

### 6.2 正式训练

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
  --train-scope base_head \
  --depth-weights 1.0,0.7,0.4,0.2 \
  --baseline-kl-weight 0.05 \
  --kl-temperature 2.0 \
  --end-token-weight 0.01 \
  --gpu 0 \
  --seed 0 \
  --save-every 1 \
  --num-workers 4
```

输出目录结构：

```text
stage1_artifacts/checkpoints/real_stage1/
  config.json
  train_log.jsonl
  checkpoint_epoch_1.pth
  checkpoint_epoch_2.pth
  ...
  best_val.pth
  last.pth
```

### 6.3 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--train-cache` | 必填 | 训练 cache 路径 |
| `--val-cache` | `None` | 验证 cache 路径；不传则只训练不验证 |
| `--init-checkpoint` | `text_generation_GPT.pth` | 初始化 GPT 权重 |
| `--base-data` | `moconvq_base.data` | MoConVQ 基础模型，用于读取 RVQ codebook |
| `--output-dir` | `stage1_artifacts/checkpoints/real_stage1` | checkpoint 和日志保存目录 |
| `--epochs` | `20` | 训练 epoch 数 |
| `--batch-size` | `8` | batch size |
| `--lr` | `1e-5` | AdamW 学习率 |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--train-scope` | `all` | 微调范围；真实实验优先用 `head` 或 `base_head`，避免破坏 baseline 运动先验 |
| `--depth-weights` | `None` | RVQ depth 加权 CE，例如 `1.0,0.7,0.4,0.2` 会更强调前两层主体动作 token |
| `--baseline-kl-weight` | `0.0` | baseline distillation 权重；大于 0 时加载冻结的 `--init-checkpoint` 作为 teacher，约束 student logits 不要偏离 baseline 太多 |
| `--kl-temperature` | `1.0` | KL distillation temperature；推荐从 `2.0` 开始 |
| `--end-token-weight` | `0.0` | padding 后第一步预测 end token 的辅助 loss；推荐小权重如 `0.01`，用于缓解乱早停/永不结束 |
| `--gpu` | `0` | 使用的 GPU id |
| `--seed` | `0` | `torch.manual_seed()` |
| `--save-every` | `1` | 每多少个 epoch 保存一次 `checkpoint_epoch_N.pth` |
| `--num-workers` | `4` | DataLoader worker 数 |
| `--smoke` | 关闭 | 只跑一个 batch，用于快速连通性测试 |

### 6.4 训练内部逻辑

`train_real_text_gpt.py` 的主流程：

```text
1. ptu.init_gpu(True, gpu_id=args.gpu)
2. 保存 config.json
3. cfg = gpt_config()
4. build_text_gpt_model(cfg, base_data_path=args.base_data)
5. _load_state_dict_flexible(model, args.init_checkpoint)
6. DataLoader(train_cache)
7. DataLoader(val_cache) 可选
8. AdamW(model.parameters(), lr, weight_decay)
9. 每个 epoch:
   - train: forward -> rvq logits -> cross entropy -> backward -> step
   - val: forward -> loss/accuracy
   - 保存 best_val.pth / checkpoint_epoch_N.pth / last.pth
   - 追加 train_log.jsonl
```

loss 计算：

```python
context_latent = latent[:, :-1, :]
logits, _ = model(context_latent, indices, clip_feature, text_feature, text_mask)
rvq_logits = logits[:, :, :indices.shape[-1], :]
loss = F.cross_entropy(
    rvq_logits.reshape(-1, rvq_logits.shape[-1]),
    indices.reshape(-1),
    ignore_index=513,
)
```

这个对齐是关键修复点。旧训练曾直接用 `latent` 与同一时刻 `indices` 对齐，并使用 `logits[:, :, 1:, :]`，等价于让模型在训练时看到当前帧 latent 和部分当前 depth token；该目标不是 MoConVQ 论文中的 autoregressive `p(I_k | I_<k)`，会造成验证指标虚高但生成阶段重复/塌缩。

`clip_feature` 当前固定为全 0：

```python
clip_feature = torch.zeros((latent.shape[0], 512), dtype=torch.float32, device=device)
```

因此这个训练入口主要依赖文本特征 `text_feature/text_mask`，不是 CLIP。

## 7. 日志解释

`train_log.jsonl` 每行是一个 epoch 的 JSON：

```json
{
  "epoch": 0,
  "train": {
    "loss": 1.23,
    "token_accuracy": 0.45,
    "valid_tokens": 123456,
    "depth_accuracy": [0.60, 0.48, 0.40, 0.32],
    "batches": 100
  },
  "val": {
    "loss": 1.50,
    "token_accuracy": 0.38,
    "valid_tokens": 23456,
    "depth_accuracy": [0.52, 0.39, 0.32, 0.25],
    "batches": 20
  },
  "lr": 1e-5,
  "elapsed_sec": 300.0
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `loss` | 所有非 padding RVQ token 的交叉熵 |
| `token_accuracy` | 所有深度、所有时间位置的 token top-1 accuracy |
| `valid_tokens` | 排除 padding index 513 后的 token 数 |
| `depth_accuracy` | 4 个 RVQ 深度各自的 token accuracy |
| `batches` | 实际跑过的 batch 数 |
| `best_val.pth` | 当前验证 loss 最低的模型 |
| `last.pth` | 最后一个 epoch 后的模型 |

查看日志：

```bash
tail -n 5 stage1_artifacts/checkpoints/real_stage1/train_log.jsonl
```

格式化最后一行：

```bash
python - <<'PY'
import json
path = "stage1_artifacts/checkpoints/real_stage1/train_log.jsonl"
last = None
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            last = json.loads(line)
print(json.dumps(last, indent=2))
PY
```

## 8. 继续训练 / 微调

继续从上次 `last.pth` 训练：

```bash
python Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache/train_cache.pt \
  --val-cache stage1_artifacts/gpt_cache/val_cache.pt \
  --init-checkpoint stage1_artifacts/checkpoints/real_stage1/last.pth \
  --base-data moconvq_base.data \
  --output-dir stage1_artifacts/checkpoints/real_stage1_continue \
  --epochs 10 \
  --batch-size 8 \
  --lr 5e-6 \
  --weight-decay 0.01 \
  --gpu 0 \
  --seed 0 \
  --save-every 1 \
  --num-workers 4
```

注意：当前脚本只保存和加载模型权重，不保存 optimizer/scheduler 状态。因此“继续训练”会重新初始化 AdamW 动量。严格复现实验时，应把 `config.json`、`train_log.jsonl` 和 checkpoint 一起保留。

## 9. 旧版训练入口：train_text_gpt.py

`Script/stage1/train_text_gpt.py` 是更简单的训练脚本：

```bash
python Script/stage1/train_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache/train_cache.pt \
  --init-checkpoint text_generation_GPT.pth \
  --output-dir stage1_artifacts/checkpoints/simple_stage1 \
  --batch-size 2 \
  --lr 1e-5 \
  --epochs 1 \
  --gpu 0 \
  --seed 0
```

它和推荐入口的区别：

| 对比项 | `train_text_gpt.py` | `train_real_text_gpt.py` |
| --- | --- | --- |
| val cache | 参数存在但当前没真正跑验证 | 支持验证 |
| padding ignore | 不使用 `ignore_index=513` | 使用 `ignore_index=513` |
| 指标 | 只记录 epoch loss | 记录 loss、token accuracy、depth accuracy |
| checkpoint | 只保存 `lphym_text_gpt.pth` | 保存 `last.pth`、`best_val.pth`、每 epoch checkpoint |
| 推荐程度 | 只适合 smoke / 兼容旧流程 | 推荐正式训练 |

除非你明确要跑旧实验，否则优先使用 `train_real_text_gpt.py`。

## 10. 训练后生成 BVH 验证

使用训练好的 checkpoint 生成一个 BVH：

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

这个脚本会：

```text
1. 加载 MoConVQ agent 和 moconvq_base.data
2. 加载 Text2Motion_Transformer checkpoint
3. 默认用 T5-large 生成 (1, L, 1024) 文本特征
4. 默认 `auto`：多段文本使用 segmented generation，单段文本使用 fixed-context rolling generation
5. 用 MoConVQ posterior decoder 解码 latent
6. 用 tracker/controller 写出 BVH
```

默认 `--text-encoder t5 --text-model t5-large`，与真实 cache 构建路径一致。只有调试离线文本 shape 时才建议显式使用 `--text-encoder hash`。

生成侧的长度处理：

```text
--max-length:      生成的总 latent token 数
--context-size:    每个 chunk 最多回看的历史 latent token 数
--chunk-size:      每轮新增 latent token 数
--max-text-length: T5 输入 token 上限，默认 256，过长 prompt 会截断
```

这解决的是 MoConVQ GPT 固定 `block_size=52` 的限制：长动作不是一次性把全部历史塞进 GPT，而是每次只保留最近一段 latent 作为上下文滚动生成。由于 block 里还要放一个 condition token，每轮实际历史长度会自动裁剪到 `51 - 当前chunk长度`，例如 `--chunk-size 25` 时最多回看 26 个历史 latent。长文本仍受 T5 tokenizer 的 `max_length` 约束；如果 prompt 很长，应在输入层面拆成更短的阶段描述，或提高 `--max-text-length` 后重新确认显存和速度。

长文本动作推荐使用分段生成；默认 `auto` 会在检测到多段文本时走这条路径：

```bash
python Script/stage1/generate_long_motion.py \
  --checkpoint stage1_artifacts/checkpoints/real_stage1_fixed/best_val.pth \
  --text "a person walks forward then turns around then waves both arms" \
  --output-bvh stage1_artifacts/generated/demo_segmented.bvh \
  --base-data moconvq_base.data \
  --text-encoder t5 \
  --text-model /home/chenjie/cc/robotics/hf_models/t5-large \
  --generation-mode auto \
  --segment-joiner " then " \
  --segment-lengths 25,25,20 \
  --context-size 26 \
  --chunk-size 25 \
  --gpu 0 \
  --seed 0
```

`segmented` 模式会把长文本按 joiner 切成局部 caption，每段单独编码文本并生成一段 motion latent；从第二段开始，脚本会把已生成序列末尾的 latent 作为 prefix/context 传给 GPT，只把新生成的 latent 追加到全局序列。`--segment-lengths` 可为每个子动作指定不同 latent token 数，数量必须和分段数一致；如果没有显式传 `--segment-lengths` 或 `--segment-length`，脚本会把 `--max-length` 自动分配到各文本段。它不是完整的高层 planner，但比“每个 rolling chunk 都看同一整段长文本”更符合 MoConGPT 的 50-code 短片段设计。

## 11. 常见问题

### 11.1 `ModuleNotFoundError: No module named 'torch'`

通常是没有激活环境：

```bash
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
```

### 11.2 `EnvironmentNameNotFound`

确认可用环境：

```bash
conda info --envs
```

本项目当前可用环境是：

```text
moconvq
```

### 11.3 checkpoint 加载出现 `module.` 前缀不匹配

训练代码已经处理：

```python
if any(k.startswith("module.") for k in state):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
```

这用于兼容 `nn.DataParallel` 保存的权重。

### 11.4 `logits/target shape mismatch`

检查 cache：

```bash
python - <<'PY'
import torch
cache = torch.load("stage1_artifacts/gpt_cache/train_cache.pt", map_location="cpu")
print(cache["latents"].shape)
print(cache["indices"].shape)
print(cache["text_features"].shape)
print(cache["text_masks"].shape)
PY
```

期望：

```text
latents:       (N, T, 768)
indices:       (N, T, 4)
text_features: (N, L, 1024)
text_masks:    (N, L)
```

如果 `indices` 最后一维不是 `4`，检查构建 cache 时的 `--rvq-depth`。

### 11.5 `index out of range in self`

通常是 `indices` 中存在非法 token。检查：

```bash
python - <<'PY'
import torch
cache = torch.load("stage1_artifacts/gpt_cache/train_cache.pt", map_location="cpu")
idx = cache["indices"]
valid = idx != 513
print("min", int(idx[valid].min()) if valid.any() else None)
print("max", int(idx[valid].max()) if valid.any() else None)
print("has below 0", bool((idx[valid] < 0).any()) if valid.any() else False)
print("has above 511", bool((idx[valid] > 511).any()) if valid.any() else False)
PY
```

训练 target 中允许：

```text
0..511: 正常 RVQ token
513:    padding ignore index
```

不应出现 `512` 作为普通 target；`512` 是模型输出中的特殊类。

### 11.6 显存不足

优先调小：

```text
--batch-size
```

例如：

```bash
--batch-size 2
```

如果仍然不足，可以减少 cache 构建阶段的 `--window-size`，但这会改变训练样本格式，需要重新构建 train/val cache。

### 11.7 DataLoader worker 报错或卡住

先用单进程 worker：

```bash
--num-workers 0
```

smoke test 建议始终先用 `--num-workers 0`。

### 11.8 训练 loss 不下降

按顺序检查：

```text
1. 是否加载了正确的 --init-checkpoint
2. --base-data 是否和 cache 构建时一致
3. cache 中 text_features 是否是训练期预期的编码方式
4. indices 的 min/max 是否合理
5. valid token 数是否过少
6. batch-size 是否过小导致指标抖动
7. lr 是否过大或过小
```

建议先确认单 batch 训练链路能跑通：

```bash
python Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache/train_cache.pt \
  --init-checkpoint text_generation_GPT.pth \
  --base-data moconvq_base.data \
  --output-dir stage1_artifacts/checkpoints/single_batch_debug \
  --epochs 1 \
  --batch-size 2 \
  --lr 1e-5 \
  --gpu 0 \
  --num-workers 0 \
  --smoke
```

如果单 batch 训练链路都跑不通，优先检查模型输入和 target 是否对齐。当前脚本没有内置“固定同一个 batch 训练多个 epoch”的 overfit 模式；需要时可以单独裁剪一个很小的 cache 再去掉 `--smoke` 做 overfit 调试。

## 12. 建议的完整命令模板

下面是一套从零开始的最小完整流程。

```bash
cd /home/chenjie/cc/robotics/MoConVQ
source /home/chenjie/miniconda3/etc/profile.d/conda.sh
conda activate moconvq
```

构建 train 长序列：

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

构建 val 长序列：

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
  --caption-mode window \
  --gpu 0 \
  --fps 20 \
  --max-text-length 256 \
  --max-failure-rate 0.1 \
  --output stage1_artifacts/gpt_cache/train_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/train_failures.jsonl
```

构建 val cache：

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
  --fps 20 \
  --max-text-length 256 \
  --max-failure-rate 0.1 \
  --output stage1_artifacts/gpt_cache/val_cache.pt \
  --failure-log stage1_artifacts/gpt_cache/val_failures.jsonl
```

先 smoke test：

```bash
python Script/stage1/train_real_text_gpt.py \
  --train-cache stage1_artifacts/gpt_cache/train_cache.pt \
  --val-cache stage1_artifacts/gpt_cache/val_cache.pt \
  --init-checkpoint text_generation_GPT.pth \
  --base-data moconvq_base.data \
  --output-dir stage1_artifacts/checkpoints/real_stage1_smoke \
  --epochs 1 \
  --batch-size 2 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --gpu 0 \
  --seed 0 \
  --save-every 1 \
  --num-workers 0 \
  --smoke
```

正式训练：

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

生成验证：

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

## 13. 最小代码阅读顺序

如果要改训练逻辑，建议按这个顺序看代码：

```text
1. Script/stage1/train_real_text_gpt.py
   - Dataset
   - compute_loss_and_metrics()
   - _run_epoch()
   - main()

2. Script/stage1/train_text_gpt.py
   - gpt_config
   - load_gpt_embeddings()
   - build_text_gpt_model()
   - _load_state_dict_flexible()

3. MoConVQCore/Model/cross_trans_ori_fixsum.py
   - Text2Motion_Transformer.forward()
   - CrossCondTransFeature
   - CrossCondTransBase
   - CrossCondTransHead

4. Script/stage1/real_moconvq_cache.py
   - build_cache_from_long_h5()
   - encode_observation_with_agent()
   - make_windows()
   - build_t5_text_encoder()
```

最容易影响训练正确性的代码点：

```text
1. forward 中 temporal feature 是否去掉额外 condition frame
2. `rvq_logits = logits[:, :, :4, :]` 是否和 indices 对齐
3. padding index 是否固定为 513
4. cache 构建时 rvq_depth 是否为 4
5. text feature 维度是否为 1024
6. base-data 是否和 cache/checkpoint 对应
```
