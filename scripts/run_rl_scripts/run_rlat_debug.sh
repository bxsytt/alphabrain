#!/bin/bash
# ============================================================
# RLActionToken Debug 脚本 — 快速验证训练流程
#
# 用途:
#   在完整训练前用最小配置跑通全流程，验证代码正确性。
#   相比正式训练脚本，此脚本:
#     - 使用单 GPU（rollout & train 同一块）
#     - 减少 iteration / episode / batch size
#     - 跳过 wandb（如需 wandb 日志可加 --use_wandb）
#     - 输出更详细的日志
#
# 用法:
#   bash scripts/run_rl_scripts/run_rlat_debug.sh
#
# 可选环境变量:
#   DEBUG_MODE=print    # 仅 print 调试信息（默认）
#   DEBUG_MODE=pdb      # 插入 pdb.set_trace()
#   DEBUG_MODE=anomaly  # 启用 autograd anomaly detection
# ============================================================

set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/home/zlb/miniconda3/envs/alphabrain/bin/python}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ── Debug 模式配置 ────────────────────────────────────────
DEBUG_MODE=${DEBUG_MODE:-print}  # print | pdb | anomaly

if [ "${DEBUG_MODE}" = "anomaly" ]; then
    export TORCH_DETECT_ANOMALY=1
    echo "[DEBUG] autograd anomaly detection ENABLED (slow, catches NaN gradients)"
fi

# ── 路径 ────────────────────────────────────────────────────
CKPT_PATH="data/final_run"
PRETRAIN_DIR="results/debug/pretrain"
ENCODER_PATH="${PRETRAIN_DIR}/checkpoints/pretrain_best/encoder.pt"
OUTPUT_DIR="results/debug/rl_offpolicy_debug_$(date +%m%d_%H%M)"
mkdir -p results/debug

RUN_NAME="debug_rlt_$(date +%m%d_%H%M)"

echo "============================================================"
echo " RLActionToken DEBUG MODE (${DEBUG_MODE})"
echo "   phase 1: encoder pretrain  -> ${PRETRAIN_DIR}"
echo "   phase 2: RL offpolicy      -> ${OUTPUT_DIR}"
echo "   suite:   libero_goal (task 0 only)"
echo "============================================================"

# ═══════════════════════════════════════════════════════════════
# Phase 1: Encoder Pretrain Debug
# ═══════════════════════════════════════════════════════════════
if [ ! -f "${ENCODER_PATH}" ]; then
    echo ""
    echo ">>> Phase 1: Encoder Pretrain (DEBUG) <<<"
    echo "     观测数: 100 | epoch: 5 | batch: 8"
    echo ""

    CUDA_VISIBLE_DEVICES=0 python AlphaBrain/training/reinforcement_learning/trainers/train.py \
        --phase pretrain \
        --ckpt_path ${CKPT_PATH} \
        --output_dir ${PRETRAIN_DIR} \
        --suite libero_goal \
        --task_id 0 \
        --bottleneck_dim 256 \
        --encoder_layers 2 \
        --encoder_heads 4 \
        --pretrain_n_obs 100 \
        --pretrain_steps_per_reset 5 \
        --pretrain_epochs 5 \
        --pretrain_lr 1e-4 \
        --pretrain_batch_size 8 \
        --vla_extract_batch_size 2 \
        --num_envs_per_task 1 \
        --seed 42

    # 强制降级保存 encoder（即使 pretrain 未收敛也能拿到 checkpoint）
    # 查找最新的 encoder checkpoint
    LATEST_ENC=$(find ${PRETRAIN_DIR} -name "encoder.pt" -type f 2>/dev/null | head -1)
    if [ -z "${LATEST_ENC}" ]; then
        echo "[WARNING] No encoder checkpoint found at ${ENCODER_PATH}"
        echo "          Script will proceed using dummy encoder (will crash if encoder not found)"
    else
        echo "[DEBUG] Encoder checkpoint found: ${LATEST_ENC}"
    fi
fi

# ═══════════════════════════════════════════════════════════════
# Phase 2: TD3 RL Debug
# ═══════════════════════════════════════════════════════════════
echo ""
echo ">>> Phase 2: TD3 RL (DEBUG) <<<"
echo "     GPU: 单卡 (rollout=0, train=0)"
echo "     iter: 5 | G_per_task: 4 | buffer_warmup: 128"
echo "     td_batch: 64 | warmup_iters: 2 | bc_steps: 50"
echo ""

# 如果开启 pdb 模式，PYTHON 会进入 pdb
if [ "${DEBUG_MODE}" = "pdb" ]; then
    export PYTHONBREAKPOINT=pdb.set_trace
    echo "[DEBUG] pdb.set_trace() enabled — add 'breakpoint()' in code to enter pdb"
fi

CUDA_VISIBLE_DEVICES=0 python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --ckpt_path ${CKPT_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal \
    --task_id 0 \
    --use_steplock \
    --rollout_gpus 0 \
    --train_gpu 0 \
    --bottleneck_dim 256 \
    --encoder_layers 2 \
    --encoder_heads 4 \
    --actor_hidden_dim 512 \
    --critic_hidden_dim 512 \
    --ref_dropout 0.5 \
    --fixed_std 0.3 \
    --G_per_task 4 \
    --group_size 1 \
    --num_envs_per_task 2 \
    --reward_coef 5.0 \
    --lr_actor 3e-4 \
    --lr_critic 3e-4 \
    --gamma 0.99 \
    --max_grad_norm 1.0 \
    --buffer_capacity 10000 \
    --buffer_warmup 128 \
    --warmup_iters 2 \
    --bc_pretrain_steps 50 \
    --td_updates_per_iter 100 \
    --utd_ratio 3.0 \
    --td_batch_size 64 \
    --tau 0.005 \
    --beta 0.2 \
    --success_weight 2.0 \
    --actor_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --max_iter 5 \
    --eval_interval 5 \
    --eval_n_episodes 4 \
    --save_interval 5 \
    --seed 42 \
    --log_interval 1

echo ""
echo "============================================================"
echo " DEBUG FINISHED"
echo "   output: ${OUTPUT_DIR}"
echo "   Run the following to check results:"
echo "     ls -la ${OUTPUT_DIR}/"
echo "     ls -la ${OUTPUT_DIR}/checkpoints/"
echo "     ls -la ${OUTPUT_DIR}/videos/  (if any)"
echo "============================================================"
