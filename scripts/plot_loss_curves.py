#!/usr/bin/env python3
"""
绘制 Actor-Critic 训练损失曲线。

用法:
  # 方式1: 解析本地 metrics.json
  python scripts/plot_loss_curves.py \
      --metrics results/action_token_training_TD3/singletask_test_*/rl_offpolicy/metrics.json \
      --save figures/loss_curves.png

  # 方式2: 从 WandB 拉取数据（需要先 wandb login）
  python scripts/plot_loss_curves.py \
      --wandb_project AlphaBrain_RLT \
      --wandb_run rlt_singletask_task0_release \
      --save figures/wandb_loss_curves.png

依赖:
  pip install matplotlib numpy
  # WandB 拉取需要: pip install wandb
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_from_metrics_json(metrics_path: str, save_path: str = None):
    """从训练自动保存的 metrics.json 绘制曲线."""
    with open(metrics_path) as f:
        data = json.load(f)

    if not data:
        print(f"[错误] {metrics_path} 为空")
        return

    iters = [d.get("iter", i) for i, d in enumerate(data)]

    # ── 提取关键指标 ──
    def safe_get(key, default=0.0):
        return [d.get(key, default) for d in data]

    actor_loss = safe_get("actor_loss")
    critic_loss = safe_get("critic_loss")
    td_loss = safe_get("td_loss")
    bc_penalty = safe_get("bc_penalty")
    q1_mean = safe_get("q1_mean")
    q_actor_mean = safe_get("q_actor_mean")
    success_rate = safe_get("success_rate")
    running_sr = safe_get("running_avg_sr")
    eval_sr = safe_get("eval_sr")
    buffer_size = safe_get("buffer_size")
    total_env_steps = safe_get("total_env_steps")

    n_cols = 3
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    fig.suptitle("Actor-Critic Training Curves", fontsize=14, fontweight="bold")

    # 1. Actor Loss
    ax = axes[0, 0]
    valid = [i for i, v in enumerate(actor_loss) if v != 0.0]
    if valid:
        ax.plot([iters[i] for i in valid], [actor_loss[i] for i in valid], "b-", label="Actor Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Actor Loss  (-Q + β·BC)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Critic Loss
    ax = axes[0, 1]
    valid = [i for i, v in enumerate(critic_loss) if v != 0.0]
    if valid:
        ax.plot([iters[i] for i in valid], [critic_loss[i] for i in valid], "r-", label="Critic Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Critic Loss  (TD3 Twin-Q MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. TD Loss (combined)
    ax = axes[0, 2]
    valid = [i for i, v in enumerate(td_loss) if v != 0.0]
    if valid:
        ax.plot([iters[i] for i in valid], [td_loss[i] for i in valid], "g-", label="TD Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Combined TD Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. BC Penalty
    ax = axes[1, 0]
    valid = [i for i, v in enumerate(bc_penalty) if v != 0.0]
    if valid:
        ax.plot([iters[i] for i in valid], [bc_penalty[i] for i in valid], "m-", label="BC Penalty")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("BC Penalty")
    ax.set_title("BC Regularization  (||a - ã||²)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Q Values
    ax = axes[1, 1]
    valid = [i for i, v in enumerate(q1_mean) if v != 0.0]
    if valid:
        ax.plot([iters[i] for i in valid], [q1_mean[i] for i in valid], "c-", label="Q1 Mean")
    v2 = [i for i, v in enumerate(q_actor_mean) if v != 0.0]
    if v2:
        ax.plot([iters[i] for i in v2], [q_actor_mean[i] for i in v2], "y-", label="Q Actor Mean")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Q Value")
    ax.set_title("Q Value Estimates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Success Rate
    ax = axes[1, 2]
    ax.plot(iters, success_rate, "b-", alpha=0.5, label="Rollout SR")
    ax.plot(iters, running_sr, "b--", label=f"Running Avg SR (last={len([s for s in running_sr if s>0])})")
    valid_eval = [(iters[i], eval_sr[i]) for i in range(len(eval_sr)) if eval_sr[i] > 0]
    if valid_eval:
        e_iter, e_sr = zip(*valid_eval)
        ax.scatter(e_iter, e_sr, color="red", s=40, zorder=5, label="Eval SR")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. Buffer Size
    ax = axes[2, 0]
    ax.plot(iters, buffer_size, "orange", label="Buffer Size")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Transitions")
    ax.set_title("Replay Buffer Fill")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Actor Loss vs Critic Loss (combined, log scale)
    ax = axes[2, 1]
    valid_a = [i for i, v in enumerate(actor_loss) if v > 0]
    valid_c = [i for i, v in enumerate(critic_loss) if v > 0]
    if valid_a:
        ax.semilogy([iters[i] for i in valid_a], [actor_loss[i] for i in valid_a], "b-", label="Actor Loss")
    if valid_c:
        ax.semilogy([iters[i] for i in valid_c], [critic_loss[i] for i in valid_c], "r-", label="Critic Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Loss Comparison (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # 9. Eval Success Rate (if available)
    ax = axes[2, 2]
    if valid_eval:
        e_iter, e_sr = zip(*valid_eval)
        ax.plot(e_iter, e_sr, "r-o", label="Eval SR")
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.text(0.5, 0.5, "No eval data yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_title("Eval Success Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] 曲线已保存到: {save_path}")
    else:
        plt.show()


def plot_from_wandb(project: str, run_name: str, save_path: str = None):
    """从 WandB 拉取数据并绘制曲线."""
    try:
        import wandb
    except ImportError:
        print("[错误] 需要安装 wandb: pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    try:
        runs = api.runs(project)
        target_run = None
        for r in runs:
            if r.name == run_name:
                target_run = r
                break
        if target_run is None:
            print(f"[错误] 在项目 '{project}' 中找不到名为 '{run_name}' 的运行")
            sys.exit(1)
    except Exception as e:
        print(f"[错误] 无法连接 WandB: {e}")
        print("  请确保已运行: wandb login")
        sys.exit(1)

    print(f"[✓] 找到 WandB 运行: {target_run.name} (id={target_run.id})")
    history = target_run.history()
    print(f"  共 {len(history)} 行数据")

    if len(history) == 0:
        print("[错误] 该运行没有历史数据")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Actor-Critic Loss Curves (WandB: {run_name})", fontsize=14, fontweight="bold")

    # 列映射
    key_map = {
        "train/actor_loss": ("Actor Loss", axes[0, 0], "b-"),
        "train/critic_loss": ("Critic Loss", axes[0, 1], "r-"),
        "train/td_loss": ("TD Loss", axes[0, 2], "g-"),
        "train/bc_penalty": ("BC Penalty", axes[1, 0], "m-"),
        "train/q1_mean": ("Q1 Mean", axes[1, 1], "c-"),
        "rollout/success_rate": ("Rollout SR", axes[1, 2], "b-"),
    }

    extra_plots = {
        "train/q_actor_mean": ("Q Actor Mean", axes[1, 1], "y-"),
        "eval/success_rate": ("Eval SR", axes[1, 2], "r-o"),
    }

    for key, (label, ax, style) in {**key_map, **extra_plots}.items():
        if key in history.columns:
            vals = history[key].dropna()
            if len(vals) > 0:
                ax.plot(vals.values, style, label=label, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if "success_rate" in key or "SR" in key:
            ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] WandB 曲线已保存到: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="绘制 Actor-Critic 训练损失曲线")
    parser.add_argument("--metrics", type=str, default=None,
                        help="本地 metrics.json 的路径 (支持 glob 通配符)")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB 项目名 (例如 AlphaBrain_RLT)")
    parser.add_argument("--wandb_run", type=str, default=None,
                        help="WandB 运行名 (例如 rlt_singletask_task0_release)")
    parser.add_argument("--save", type=str, default=None,
                        help="保存图片路径 (例如 figures/loss_curves.png)")

    args = parser.parse_args()

    if args.metrics:
        # 支持 glob 通配符
        import glob
        paths = glob.glob(args.metrics)
        if not paths:
            print(f"[错误] 未找到匹配的文件: {args.metrics}")
            sys.exit(1)
        for p in sorted(paths):
            print(f"[✓] 读取: {p}")
            # 为每个文件生成独立的保存路径
            save_p = None
            if args.save:
                base, ext = os.path.splitext(args.save)
                if len(paths) > 1:
                    tag = os.path.basename(os.path.dirname(p))
                    save_p = f"{base}_{tag}{ext}"
                else:
                    save_p = args.save
            plot_from_metrics_json(p, save_p)
    elif args.wandb_project and args.wandb_run:
        plot_from_wandb(args.wandb_project, args.wandb_run, args.save)
    else:
        print("请提供 --metrics 或 --wandb_project + --wandb_run")
        print()
        print("示例:")
        print("  # 本地 metrics.json")
        print("  python scripts/plot_loss_curves.py --metrics 'path/to/metrics.json' --save figures/loss.png")
        print()
        print("  # WandB")
        print("  python scripts/plot_loss_curves.py --wandb_project AlphaBrain_RLT --wandb_run rlt_singletask_task0_release --save figures/wandb_loss.png")


if __name__ == "__main__":
    main()
