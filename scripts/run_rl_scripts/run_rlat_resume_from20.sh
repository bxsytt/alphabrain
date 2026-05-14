#!/bin/bash
# 从 iter 20 checkpoint 恢复训练，继续训练 20 个 iter（到 iter 40）

RESUME_CKPT="/home/zlb/embody_project/AlphaBrain/results/action_token_training_TD3/singletask_test_0513_1012/rl_offpolicy/checkpoints/rl_offpolicy_iter_00020"
NEW_DIR="results/action_token_training_TD3/singletask_test_0513_$(date +%m%d_%H%M)/rl_offpolicy"

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
    --warmup_iters 5 \
    --bc_pretrain_steps 2000 \
    --td_updates_per_iter 10000 \
    --utd_ratio 3.0 \
    --td_batch_size 256 \
    --tau 0.005 \
    --beta 0.2 \
    --success_weight 2.0 \
    --actor_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --max_iter 100 \
    --eval_interval 20 \
    --eval_n_episodes 20 \
    --save_interval 20 \
    --save_video_interval 100 \
    --seed 42 \
    --use_wandb \
    --wandb_project AlphaBrain_RLT \
    --run_name "rlt_singletask_task0_resume_from20" \
    --log_interval 1
