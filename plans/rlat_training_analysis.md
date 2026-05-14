# RLAT (Reinforcement Learning Action Token) 训练诊断分析报告

## 一、整体流程回顾

```
Baseline VLA (qwengr00t_cl_lora_final.pt)
  │
  ▼
Phase 1: Encoder Pretrain ─── 训练 encoder-decoder，将 VLA hidden features (2048-dim)
  │                           压缩为 256-dim 瓶颈 token (rl_token)
  ▼
Phase 2: Off-Policy TD3 RL ── 冻结 VLA + encoder，训练 actor-critic 小网络
  │     ├── Iter 1-5:   VLA Warmup（纯 VLA rollout 填充 buffer，无 TD3 更新）
  │     └── Iter 6-500: TD3 Training（actor 参与 rollout，actor+critic 联合训练）
  ▼
输出: 训练好的 actor (ActionTokenActor) + critic (ActionTokenQCritic)
```

---

## 二、metrics.json 定量分析

### 2.1 全量数据概览

| 指标 | Iter 1-5 (Warmup) | Iter 6-100 | Iter 100-500 |
|------|:---:|:---:|:---:|
| **rollout success_rate** | 最高 12.5% | **0.0%** | **0.0%** |
| **eval_sr** | 0.0% | 0.0% | 0.0% |
| **best_success_rate** | 12.5% | 12.5% | 12.5% |
| **td_loss** | N/A (无 TD3) | 5.8 → 1.9 | 0.38 |
| **critic_loss** | N/A | 0.018 → 0.001 | **0.0001** |
| **bc_penalty** | N/A | 5.8 → 1.9 | **0.38** (≈ actor_loss) |
| **actor_loss** | N/A | 5.8 → 1.9 | **0.38** (≈ bc_penalty) |
| **q1_mean / q2_mean** | N/A | ~0.04 | **~0.0 → -0.006** |
| **buffer_size** | 320 → 1530 | 1530 → 31K | **100K (满)** |

### 2.2 关键发现

1. **actor_loss ≈ bc_penalty (严格相等)** — actor loss 中 Q 项 `-q_val.mean()` 几乎为 0，bc_penalty 占 100% 主导
2. **critic_loss 趋近于 0** — critic 学到了"无论输入什么 Q 值都是 0"
3. **q1_mean / q2_mean 最终为负** — 说明 critic 认为所有动作都会导致负回报
4. **buffer 被 failure trajectories 完全占据** — 100K 容量全部是失败轨迹

---

## 三、核心问题诊断

### 问题 1: Cold Start 死亡螺旋（致命问题）

```
Warmup 结束 (Iter 6)
  │
  ▼
Actor 随机初始化 → 生成随机 action
  │
  ▼
Rollout SR = 0.0% → Buffer 中塞满了失败轨迹
  │
  ▼
TD3 训练: Actor loss = -Q + β·BC
              ≈ 0 + 1.0 * BC(failure_actions) ← BC 正则化在拟合"失败动作"
  │
  ▼
Critic 从未见到 positive reward → Q(s,a) ≈ 0 for all
  │
  ▼
Actor 的 Q 项 = -Q.mean() ≈ 0 → 梯度完全来自 BC on failure
  │
  ▼
下一轮 actor → 依然生成失败动作 → 恶性循环 🔄
```

**根因**: Actor 缺少"冷启动"策略。从随机初始化直接介入 rollout 时，初期探索的成功概率极低，导致 buffer 中没有任何 positive 样本可供学习。

### 问题 2: β=1.0 的 BC 正则化在失败数据上适得其反

公式: `actor_loss = -Q(s, π(s)) + β * ||π(s) - a_VLA||²`

- 当 buffer 中全是失败轨迹时，`a_VLA` 是从 **失败 episode** 中采样的 VLA 动作
- β=1.0 强制 actor 去拟合"导致失败"的 VLA 动作
- 这完全违反了 BC 正则化的设计初衷 — BC 应该在 VLA **表现好** 时约束 actor 不偏离太远

### 问题 3: Warmup 数据太少且质量不足

启动脚本参数: `--warmup_iters 5`, `--G_per_task 30`

- 5 轮 warmup × 30 episodes = **150 episodes** (但只有 ~12.5% 成功，即 ~19 个成功 episode)
- 成功轨迹数量远不足以支撑后续 RL 训练
- VLA baseline 本身的成功率本身就不高（可能在 20-30%），buffer 中成功/失败比例严重失衡

### 问题 4: Encoder 在失败数据上的表示能力

Phase 1 的 encoder 是在随机 rollout 数据上通过 reconstruction 训练的。如果 Phase 1 的数据分布与 Phase 2 的分布（actor 产出）差异过大，encoder 的 bottleneck token 可能无法有效表示状态。

---

## 四、针对具体问题的回答

### 回答 1: 如何判断视频中是否加入了 RL Token 效果？

**判断方法**:
1. 视频文件名或路径中的 `eval_iter_XXXXX` 区分 warmup 阶段和 TD3 阶段
2. 查看 [`metrics.json`](results/action_token_training/metrics.json) 中对应 iter 的 `eval_sr` 字段
3. 观察视频中机械臂的动作是否"更平滑"、"更精准" — RL fine-tune 后的动作通常抖动更少
4. 对比 baseline VLA 的视频和 RL 视频中 **任务完成的关键帧时序**（是否更快完成任务）

**为什么 baseline 成功率更高？**

原因是上述 **Cold Start 死亡螺旋**。VLA baseline（纯行为克隆）已经在训练数据上拟合过，动作质量相对稳定。而 RL Token 的 actor 需要从零开始探索，但在当前配置下陷入了"全失败 → 无正信号 → 学不到"的循环。

### 回答 2: Batch size 和训练调度策略

#### Batch size

| 参数 | 你提到的值 | 脚本中的实际值 | 建议 |
|------|:---:|:---:|:---:|
| `td_batch_size` | 8 | **256** | ✅ 256 是合理的 |
| `utd_ratio` | — | 10.0 | ⚠️ 可能过高 |
| `td_updates_per_iter` | — | 10000 | ⚠️ cap 值，实际受 UTD 控制 |

**分析**: 实际启动脚本中 `--td_batch_size 256`，并不是 8。256 对于 actor-critic 小网络（~9M 参数）来说是合理的，可以保持。可能你之前尝试过 batch_size=8 的配置导致训练更差。

#### 训练调度策略

代码中 [`train_rl_offpolicy.py`](AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:732-744) 的逻辑是:

```
Warmup:   Iter 1 - 5  → 纯 VLA rollout，无 TD3 更新
                              │
Transition: Iter 6    → warmup_mode=False，actor 开始参与 rollout
                              │
TD3:      Iter 6 - 500 → actor + critic 联合训练
```

**不是"每 10 轮中前 5 轮 warmup"**，而是**全局 5 轮 warmup + 剩余全部 TD3**。

**UTD 机制**（基于 Update-To-Data ratio）:
```python
n_updates = max(1, int(n_new_transitions * args.utd_ratio / batch_sz))
n_updates = min(n_updates, args.td_updates_per_iter)  # capped at 10000
```
每轮收到新数据后，计算 UTD×new_data/batch_size 次更新。当 utd_ratio=10, batch_sz=256, 每轮 ~2560 新 transition 时:
```
n_updates = max(1, 2560 * 10 / 256) = 100 → 每轮 100 次 TD3 更新
```

---

## 五、优化方案

### 方案 A: 增加 Warmup + Actor 预训练（推荐，改动最小）

| 修改项 | 当前值 | 建议值 | 理由 |
|--------|:---:|:---:|:---:|
| `warmup_iters` | 5 | **20-50** | 收集更多高质量 VLA 轨迹 |
| `G_per_task` | 30 | **50-100** | 增加每轮数据量 |
| `beta` (BC weight) | 1.0 | **0.1-0.3** | 降低对失败数据的 BC 依赖 |
| `fixed_std` | 0.1 | **0.3-0.5** | 增加探索噪声 |

**关键补充**: 在 warmup 结束后、TD3 开始前，增加 **Actor BC Pretrain**:
- 从 buffer 中采样 VLA 数据，用 BC 预训练 actor N 步
- 这样 actor 在参与 rollout 前就已经学会模仿 VLA 动作
- 后续 RL 只需在 BC 基础上微调

### 方案 B: 分层 Buffer + 优先采样

**问题**: 均匀采样导致 99%+ 的 batch 都是失败数据。

**改进**:
1. 维护两个 buffer: `success_buffer` (只存成功轨迹) 和 `fail_buffer`
2. 采样时以 50:50 比例从两个 buffer 中抽取
3. 对成功轨迹做 oversampling

### 方案 C: 分阶段学习率 + 熵正则化

1. **Phase 2a (Iter 1-50)**: 高熵探索
   - `fixed_std = 0.5`, `beta = 0.1`
   - Actor loss 中增加熵奖励: `+ α * H(π)`
   - 目标是让 actor 在初期敢于探索不同动作

2. **Phase 2b (Iter 50-200)**: 正常 TD3
   - `fixed_std = 0.2`, `beta = 1.0`

3. **Phase 2c (Iter 200-500)**: 精细 tuning
   - `fixed_std = 0.05`, `beta = 2.0`

### 方案 D: 使用 On-Policy RL（PPO）替代 Off-Policy TD3

On-policy 方法对 off-policy 数据的分布偏移不那么敏感，适合这种 cold-start 场景。

当前你也有 on-policy 训练器 [`train_rl_onpolicy.py`](AlphaBrain/training/reinforcement_learning/trainers/train_rl_onpolicy.py)，可以考虑切换。

---

## 六、量化指标改进预期

| 指标 | 当前 (500 iter) | 优化后预期 (500 iter) |
|------|:---:|:---:|
| eval_sr | 0.0% | **15-25%** (接近 VLA baseline) |
| bc_penalty | ~0.38 (拟合 failure) | 0.1-0.2 (拟合 success) |
| q1_mean | -0.006 | **0.1-0.3** (正向 Q 值) |
| buffer 中 success 占比 | <0.1% | **5-10%** |

---

## 七、启用脚本参数对照

关键参数在 [`run_rlat_5traj_alltasks.sh`](scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh) 中:

```bash
--warmup_iters 5          # 建议改为 20-50
--td_batch_size 256       # 合理，保持
--utd_ratio 10.0          # 建议降到 2.0-5.0
--beta 1.0                # 建议降到 0.1-0.3
--fixed_std 0.1           # 建议升到 0.3-0.5
--G_per_task 30           # 建议升到 50-100
```
