#!/bin/bash
# 从 iter 100 checkpoint 恢复训练，继续训练到 iter 300（再训练 200 轮）
# 参数沿用原 run_rlat_singletask_v2.sh 的值
#
# 主要修正（对比旧版 run_rlat_resume_from20.sh）：
#   1. max_iter=300（原为 100 → 空循环 Bug，start_iter=101 > max_iter=100）
#   2. warmup_iters=0（resume 不需要 VLA warmup）
#   3. bc_pretrain_steps=0（resume 不需要 BC pretrain）
#   4. 参数对齐原 singletask_v2：success_weight=5.0, td_updates_per_iter=50000, eval_interval=10, save_interval=10
# ============================================================

RESUME_CKPT="/home/zlb/embody_project/AlphaBrain/results/action_token_training_TD3/singletask_resume_v2_0514_1122/rl_offpolicy/checkpoints/rl_offpolicy_iter_00200"
NEW_DIR="results/action_token_training_TD3/singletask_resume_v2_$(date +%m%d_%H%M)/rl_offpolicy"

python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --ckpt_path data/final_run \
    --resume "${RESUME_CKPT}" \
    --encoder_path results/rlt_training_TD3/5traj_alltasks_pretrain/pretrain/checkpoints/pretrain_best/encoder.pt \
    --output_dir "${NEW_DIR}" \
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
    --warmup_iters 0 \
    --bc_pretrain_steps 0 \
    --td_updates_per_iter 50000 \
    --utd_ratio 10.0 \
    --td_batch_size 256 \
    --tau 0.005 \
    --beta 0.5 \
    --success_weight 5.0 \
    --actor_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --max_iter 400 \
    --eval_interval 50 \
    --eval_n_episodes 50 \
    --save_interval 50 \
    --save_video_interval 50 \
    --seed 42 \
    --use_wandb \
    --wandb_project AlphaBrain_RLT \
    --run_name "rlt_singletask_task0_resume_from100_to300" \
    --log_interval 1
