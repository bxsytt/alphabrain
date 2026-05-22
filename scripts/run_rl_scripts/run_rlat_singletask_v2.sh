#!/bin/bash
# ============================================================
# RLActionToken 单任务训练 v2 — 修复 success_weight + 优化参数
#
# 主要改进：
#   1. 修复 ReplayBuffer.sample() 支持 success_weight（原来单任务不生效）
#   2. warmup_iters=20 → 更多纯 VLA 数据填充 buffer
#   3. bc_pretrain_steps=5000 → 更充分的 BC 预训练
#   4. utd_ratio=10.0 → 每次迭代 ~90 次 TD 更新（原来是 27）
#   5. beta=0.5 → 更强的 BC 正则防止 actor 漂移
#   6. success_weight=5.0 → 成功 transition 过采样率 5×
# ============================================================

RUN_NAME="rlt_singletask_task0_v2_$(date +%m%d_%H%M)"
CKPT_PATH="data/final_run"
ENCODER_PATH="results/rlt_training_TD3/phase1_pretrain/pretrain/checkpoints/pretrain_best/encoder.pt"
OUTPUT_DIR="results/action_token_training_TD3/${RUN_NAME}/rl_offpolicy"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate alphabrain

export CUDA_VISIBLE_DEVICES=0,1
export LIBERO_PYTHON=/home/zlb/miniconda3/envs/libero/bin/python
export PYTHONPATH="${PYTHONPATH}:${PWD}"

echo "============================================================"
echo " RLActionToken Single-Task Training v2"
echo "   run_name:  ${RUN_NAME}"
echo "   ckpt:      ${CKPT_PATH}"
echo "   encoder:   ${ENCODER_PATH}"
echo "   output:    ${OUTPUT_DIR}"
echo "   suite:     libero_goal | task_id: 0"
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
    --train_gpu 1 \
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
