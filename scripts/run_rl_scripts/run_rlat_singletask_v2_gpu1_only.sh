#!/bin/bash
# ============================================================
# RLActionToken 单任务训练 v2 — 仅使用 GPU1（绕过 GPU0）
#
# 适用场景：GPU0 掉卡，仅 GPU1 可用
# 改动：
#   - CUDA_VISIBLE_DEVICES=1（只暴露 GPU1）
#   - rollout_gpus 和 train_gpu 都指向 GPU1
#   - 由于只有一张卡，rollout 和 training 串行执行
# ============================================================

RUN_NAME="rlt_singletask_task0_v2_gpu1only_$(date +%m%d_%H%M)"
CKPT_PATH="data/final_run"
ENCODER_PATH="results/rlt_training_TD3/phase1_pretrain/pretrain/checkpoints/pretrain_best/encoder.pt"
OUTPUT_DIR="results/action_token_training_TD3/${RUN_NAME}/rl_offpolicy"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate alphabrain

# ★ 关键：只暴露 GPU1，让 CUDA 认为它是唯一的 GPU（cuda:0）
export CUDA_VISIBLE_DEVICES=1
export LIBERO_PYTHON=/home/zlb/miniconda3/envs/libero/bin/python
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# ★ 开启 CUDA_LAUNCH_BLOCKING=1 以便在出错时获得精确的堆栈跟踪
export CUDA_LAUNCH_BLOCKING=1

echo "============================================================"
echo " RLActionToken Single-Task Training v2 (GPU1 Only)"
echo "   run_name:  ${RUN_NAME}"
echo "   ckpt:      ${CKPT_PATH}"
echo "   encoder:   ${ENCODER_PATH}"
echo "   output:    ${OUTPUT_DIR}"
echo "   suite:     libero_goal | task_id: 0"
echo "   GPU:       GPU1 only (CUDA_VISIBLE_DEVICES=1)"
echo "============================================================"

python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --ckpt_path "${CKPT_PATH}" \
    --encoder_path "${ENCODER_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
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
    --fixed_std 0.2 \
    --G_per_task 60 \
    --group_size 1 \
    --num_envs_per_task 6 \
    --reward_coef 5.0 \
    --lr_actor 3e-4 \
    --lr_critic 3e-4 \
    --gamma 0.99 \
    --max_grad_norm 1.0 \
    --buffer_capacity 1000000 \
    --buffer_warmup 10240 \
    --warmup_iters 20 \
    --bc_pretrain_steps 5000 \
    --td_updates_per_iter 50000 \
    --utd_ratio 10.0 \
    --td_batch_size 256 \
    --tau 0.005 \
    --beta 0.5 \
    --success_weight 5.0 \
    --actor_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --max_iter 200 \
    --eval_interval 20 \
    --eval_n_episodes 20 \
    --save_interval 20 \
    --save_video_interval 20 \
    --seed 42 \
    --use_wandb \
    --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_NAME}" \
    --log_interval 1
