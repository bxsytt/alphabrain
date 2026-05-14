# RLAT TD3 训练 GPU 显存分析与超参数调整方案

## 1. 当前 GPU 显存状态

| GPU | 用途 | 已使用 | 总量 | 使用率 | 温度 | 备注 |
|-----|------|--------|------|--------|------|------|
| GPU 0 (RTX 3090) | **Rollout** (环境推理) | **17,620 MiB** | 24,576 MiB | **72%** | **82°C** 🔥 | 瓶颈 GPU |
| GPU 1 (RTX 3090) | **Training** (Actor-Critic 训练) | **10,179 MiB** | 24,576 MiB | **41%** | **40°C** ❄️ | 充裕余量 |

### 显存消耗明细

**GPU 0 (Rollout):**
- 冻结 VLA 模型 (Qwen2.5-VL-3B, bf16 ≈ 6GB + KV cache/activations ≈ 2-3GB) → ~9GB
- 20 个 LIBERO env 的 persistent env pool 显存开销 → ~3-4GB
- Rollout modules (encoder/actor/critic 副本 × 1) → 很小
- Eval modules (encoder/actor 副本) → 很小
- 其他开销 → ~1-2GB
- **总计: ~17.6GB**

**GPU 1 (Training):**
- 冻结 VLA 模型 (用于 eval) → ~9GB
- Actor (hidden_dim=512, chunk_len=4, action_dim=7) → < 2MB
- Critic (twin-Q, hidden_dim=512) → < 4MB
- Target networks (actor + critic 副本) → < 6MB
- Encoder (bottleneck_dim=256, 2 layers, 4 heads) → < 2MB
- Training activations (batch_size=256) → ~100-200MB
- **总计: ~10.2GB**

## 2. 关键发现

### GPU 0 温度问题 (82°C)
- NVIDIA RTX 3090 通常在 **83-85°C** 开始降频
- 当前 82°C 已经接近降频阈值，GPU 0 可能已经或即将降频
- GPU 0 作为 rollout GPU，温度偏高的原因是：
  - LIBERO 仿真大量使用 MuJoCo (EGL rendering)
  - Persistent env pool 持续运行 20 个环境
  - GPU 需要同时处理 VLA 推理 + EGL 渲染

### GPU 1 余量充裕 (41%, 40°C)
- 有约 **14GB 可用显存**
- 温度极低，说明计算负载远未饱和
- Actor/Critic 网络非常小（几百万参数级别）

## 3. 调整方案

### 方案 A：仅调大 Training GPU 的参数（安全，无需停机）

| 参数 | 当前值 | 建议值 | 说明 |
|------|--------|--------|------|
| `--td_batch_size` | 256 | **1024** | 4x 提升，更稳定的 Q 函数更新。GPU 1 显存完全够 |
| `--actor_hidden_dim` | 512 | **1024** | 2x 网络容量，可以更好地拟合动作分布 |
| `--critic_hidden_dim` | 512 | **1024** | 同上，提升 critic 的表达能力 |
| `--utd_ratio` | 3.0 | **5.0** | 每个新 data 做更多梯度步，加速学习 |
| `--bc_pretrain_steps` | 2000 | **5000** | 更充分的 BC warmup，避免 cold-start 问题 |
| `--beta` | 0.2 | **0.5** | BC 正则化权重，新版论文建议更高值稳定训练 |
| `--reward_coef` | 5.0 | 5.0 | 保持，已设为明显区分奖励信号 |
| `--target_noise_std` | 0.2 | 0.2 | 保持 TD3 标准值 |
| `--success_weight` | 2.0 | **3.0** | 更多采样成功轨迹，缓解稀疏奖励 |

**显存影响**: `td_batch_size=1024` 在 GPU 1 上大约增加 200-300MB 激活显存，远在安全范围内。`hidden_dim` 翻倍增加约 < 10MB 参数。

### 方案 B：同时微调 Rollout GPU 的参数（需确认 GPU 0 的温控状态）

| 参数 | 当前值 | 建议值 | 说明 |
|------|--------|--------|------|
| `--num_envs_per_task` | 2 | **4** | 翻倍并行环境数，需确认 GPU 0 显存和温度 |
| `--vla_extract_batch_size` | 4 | **8** | 仅 pretrain 阶段有用，当前在 offpolicy 阶段不生效 |
| `--G_per_task` | 60 | 60 | 保持，已合理 |

**注意**: `--num_envs_per_task` 从 2 → 4 会在 GPU 0 上增加约 2-3GB 显存（更多 env states + 更多 VLA 推理 KV cache），可能使 GPU 0 显存从 72% → 85%+，**不推荐**。

### 方案 C：引入 VLA Fine-tune（高级选项）

当前脚本有 `--finetune_vla` 选项但未启用。启用后：
- VLA 模型也会参与更新（梯度 checkpointing 开启）
- 会显著增加 GPU 1 显存和计算需求
- 适合训练后期收敛停滞时使用

## 4. 是否可以停训并续训？

### 当前进程状态
- Python PID 2322574 正在运行
- 两个子进程 PID 2323101/2323102 (LIBERO env workers)
- **训练正在进行中**

### Checkpoint 保存机制
- 每 `--save_interval 50` 次迭代保存一次 checkpoint
- 保存内容：`enc_dec`、`actor`、`q_critic` 状态字典
- 输出目录: `results/action_token_training_TD3/rlt_5traj_alltasks_release_*/rl_offpolicy/`

### 能否停训？
- **可以停训**，但需注意：
  - **Replay buffer 不可持久化** — 重启后会丢失所有 buffer 数据
  - `--warmup_iters 30` 次的 VLA rollout 数据会丢失
  - `--bc_pretrain_steps 2000` 的 BC 预训练结果会丢失
  - 如果当前在 TD3 训练早期（< 100 iter），丢失 buffer 影响相对可控
  - checkpoint 中保存的 actor/critic 权重可以在重启后恢复

### 如何续训？
当前脚本是线性执行的（Phase 1 pretrain → Phase 2 offpolicy），没有续训逻辑。如果要续训：
1. 需要修改 `train.py` 或 `train_rl_offpolicy.py`，添加从 checkpoint 恢复 actor/critic 和 encoder 的逻辑
2. 需要重建 replay buffer（重新收集）
3. 建议在训练脚本中增加 `--resume` 参数

**更实际的做法**：不停止当前训练，直接修改运行中的训练参数比较困难（参数在启动时通过命令行传入）。建议：
1. 如果是训练初期，**可以停掉并调整参数重新开始**（代价是损失 warmup 数据）
2. 如果是训练中后期（>100 iter），**不建议停**，因为 replay buffer 中的数据是积累了多次迭代的宝贵经验

## 5. 具体建议

### 推荐操作（按优先级）

1. **立即调整（修改配置文件后重启训练）：**
   - `--td_batch_size 256` → `1024`
   - `--actor_hidden_dim 512` → `1024`
   - `--critic_hidden_dim 512` → `1024`
   - `--utd_ratio 3.0` → `5.0`
   - `--beta 0.2` → `0.5`
   - `--success_weight 2.0` → `3.0`

2. **GPU 0 散热优化（不更改训练逻辑）：**
   - 降低 GPU 0 功率限制：`sudo nvidia-smi -i 0 -pl 250`（从 350W 降至 250W）
   - 检查散热风扇是否正常
   - 当前 82°C 接近降频阈值，注意监控

3. **额外的稳定训练建议：**
   - 增加 `--warmup_iters 30` → `50`（更充分的 buffer 预热）
   - 增加 `--bc_pretrain_steps 2000` → `5000`
   - 保持 `--max_iter 400` 不变

## 6. 修改后的完整命令（仅 Phase 2 部分）

```bash
python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --ckpt_path /home/zlb/embody_project/AlphaBrain/data/final_run \
    --encoder_path <pretrain_encoder_path>/checkpoints/pretrain_best/encoder.pt \
    --output_dir results/action_token_training_TD3/rlt_5traj_alltasks_release_v2/rl_offpolicy \
    --suite libero_goal \
    --all_tasks \
    --use_steplock \
    --rollout_gpus 0 \
    --train_gpu 1 \
    --bottleneck_dim 256 \
    --encoder_layers 2 \
    --encoder_heads 4 \
    --actor_hidden_dim 1024 \
    --critic_hidden_dim 1024 \
    --ref_dropout 0.5 \
    --fixed_std 0.3 \
    --G_per_task 60 \
    --group_size 1 \
    --num_envs_per_task 2 \
    --reward_coef 5.0 \
    --lr_actor 3e-4 \
    --lr_critic 3e-4 \
    --gamma 0.99 \
    --max_grad_norm 1.0 \
    --buffer_capacity 1000000 \
    --buffer_warmup 1024 \
    --warmup_iters 50 \
    --bc_pretrain_steps 5000 \
    --td_updates_per_iter 10000 \
    --utd_ratio 5.0 \
    --td_batch_size 1024 \
    --tau 0.005 \
    --beta 0.5 \
    --success_weight 3.0 \
    --actor_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --max_iter 400 \
    --eval_interval 20 \
    --eval_n_episodes 20 \
    --save_interval 50 \
    --save_video_interval 100 \
    --seed 42 \
    --use_wandb \
    --wandb_project AlphaBrain_RLT \
    --run_name rlt_5traj_alltasks_release_v2 \
    --log_interval 1
```
