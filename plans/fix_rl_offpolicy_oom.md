# 修复 Off-Policy RL 训练被 OOM Killer 终止的问题

## 问题诊断

### 现象
运行命令后，模型加载成功但在加载 `latest_model.pt` 时被 Linux OOM Killer 终止（"Killed"），无 Python traceback。

### 环境
- **CPU RAM**: 31GB（实际可用 ~9.9GB）
- **Swap**: 2GB（已用满）
- **GPU**: 2× RTX 3090（24GB VRAM each）
- **Checkpoint**: `data/final_run/latest_model.pt` = **7.6GB**

### 根本原因

你运行的命令没有指定 `--rollout_gpus` 和 `--train_gpu`，导致：
- 默认使用了 **全部 2 张 GPU** 作为 Rollout GPU
- 代码在 [`train_rl_offpolicy.py:67-74`](../AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:67) 中为每张 rollout GPU **各加载一份完整的 VLA 模型**

加载流程中的 CPU 内存峰值：

| 步骤 | CPU 内存开销 | 说明 |
|------|-------------|------|
| 基础系统占用 | ~20GB | 含已有进程 |
| 第一份 VLA 构建（build_framework） | ~6GB | 加载 Qwen2.5-VL-3B 模型结构 |
| 第一份 VLA 加载 latest_model.pt | +7.6GB | `torch.load()` 将 7.6GB state_dict 读入 CPU |
| 第一份 VLA 移至 GPU 后残留 | ~3-4GB | Python 对象尚未 GC |
| 第二份 VLA 构建 | +6GB | 再次加载 Qwen2.5-VL-3B |
| **峰值总计** | **~43GB** | **远超 31GB 物理内存** |

Swap 2GB 已用满，系统无法通过换页缓解，OOM Killer 触发。

### 与参考脚本的对比

参考脚本 [`run_rlat_5traj_alltasks.sh`](../scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh:76) 正确指定了 GPU 分配：
```bash
--rollout_gpus 0   # 只有 GPU 0 做 rollout（加载 1 份 VLA）
--train_gpu 1      # GPU 1 做训练
```

而你的命令缺少这两个参数，导致默认加载了 **2 份 VLA**。

## 修复方案

### 方案一（推荐）：按照参考脚本正确指定 GPU 分配

```bash
python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --ckpt_path data/final_run \
    --encoder_path results/rlt_training_TD3/5traj_alltasks_pretrain/pretrain/checkpoints/pretrain_best/encoder.pt \
    --suite libero_goal \
    --task_id 0 \
    --rollout_gpus 0 \
    --train_gpu 1
```

**原理**：仅 GPU 0 加载一份 VLA 做 rollout，GPU 1 专门做 TD3 训练。CPU 内存峰值降至 ~26GB，在 31GB 范围内。

### 方案二：仅使用单 GPU

```bash
CUDA_VISIBLE_DEVICES=0 python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --ckpt_path data/final_run \
    --encoder_path ... \
    --suite libero_goal \
    --task_id 0 \
    --rollout_gpus 0 \
    --train_gpu 0
```

**注意**：单 GPU 模式下，rollout 和训练串行执行，速度较慢。

### 方案三（长期优化）：转换为自包含 checkpoint 格式

减少 CPU 内存的关键在于让 checkpoint 包含 `vlm_pretrained/` 子目录，这样加载时可以使用 **meta device 初始化**，避免完整的 VLM 权重在 CPU 上重复驻留。

[`base_framework.py:134-153`](../AlphaBrain/model/framework/base_framework.py:134) 中已有此逻辑：
```python
# 如果 vlm_pretrained/ 存在，使用单次加载优化（meta device）
# 否则回退到两次读取（先加载 base VLM，再加载 latest_model.pt）
```

可以通过 `base_framework.convert_checkpoint_to_dir()` 方法转换。

## 实施步骤

1. **临时修复**：在命令中添加 `--rollout_gpus 0 --train_gpu 1`
2. **可选优化**：将 `data/final_run` 转换为带 `vlm_pretrained/` 的自包含格式
3. **验证**：确认训练正常启动

## 系统架构图

```mermaid
flowchart TD
    A[用户命令: --phase rl_offpolicy] --> B{指定 rollout_gpus?}
    B -->|未指定: 默认全部 GPU| C[Rollout GPUs: [0, 1]]
    B -->|指定: --rollout_gpus 0| D[Rollout GPUs: [0]]
    C --> E[加载 2份 VLA 到 GPU 0 和 GPU 1]
    E --> F[CPU RAM 峰值 ~43GB]
    F --> G[超过 31GB 限制]
    G --> H[OOM Killer: Killed]
    D --> I[加载 1份 VLA 到 GPU 0]
    I --> J[CPU RAM 峰值 ~26GB]
    J --> K[在 31GB 范围内 ✓]
    K --> L[训练正常启动]
```
