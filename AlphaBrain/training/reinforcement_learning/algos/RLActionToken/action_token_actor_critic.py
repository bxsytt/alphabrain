"""
ActionToken Actor-Critic, following the RL Token paper (Physical Intelligence)
closely on the actor/critic side (the deviations from the paper live in
action_token_encoder_decoder.py and action_token_trainer.py).

Key design choices from the paper:
  - Actor (Eq. 4): π_θ(a | x, ã) = N(μ_θ(x, ã), σ²I)
    The actor DIRECTLY outputs the action chunk, conditioned on (rl_token, vla_ref).
    VLA reference is an INPUT to the network, NOT a structural residual.
    BC regularization β‖a - ã‖² in the LOSS keeps actions close to VLA.
    Reference-action dropout (50%) prevents identity collapse.
  - Critic (Eq. 3): Q(s, a) — twin Q-networks (TD3-style).
"""

"""
信息瓶颈核心 — 包含 ActionTokenEncoder（将 VLA 的 action queries 压缩为 rl_token）和 ActionTokenDecoder（自回归重构 VLA tokens，仅预训练使用）
"""

import copy

import torch
import torch.nn as nn

"""
含义： 论文 Eq.4-5 的 Actor 网络 π_θ(a|x, ã)。它直接输出完整的动作序列 chunk，输入
是 RL token、VLA 参考动作和本体感知状态。VLA 参考动作不是残差连接，而是一个条件输入；
通过损失中的 BC 正则化项 β‖a−ã‖² 使输出接近 VLA 参考。参考动作 Dropout（默认 50%）防止策略退化为恒等映射。

    输入：
        rl_token: (B, 1, D) 或 (B, D) — 压缩后的 RL 潜在表示
        vla_action: (B, chunk_len, action_dim) — VLA 模型的参考动作
        prop_state: (B, prop_dim)（可选）— 本体感知状态（论文中为 eef_pos(3) + axisangle(3) + gripper(2)=8 维）
        deterministic: bool — 是否确定性输出
    输出：
        action: (B, chunk_len, action_dim) — 采样的动作序列
        log_prob: (B,) — 每个样本动作的对数概率（确定性模式为 None）
关键设计： 固定标准差 fixed_std=0.1，输出为高斯分布 N(μ, σ²I)
"""
class ActionTokenActor(nn.Module):
    """
    ActionToken actor from paper (Eq. 4-5).

    π_θ(a_{1:C} | x, ã_{1:C}) = N(μ_θ(x, ã_{1:C}), σ²I)

    The network takes (rl_token, vla_reference_action) as input and
    DIRECTLY outputs the full action chunk. The VLA reference is just
    a conditioning signal — the BC regularization in the loss (not in
    the architecture) keeps the output close to VLA.
    """

    def __init__(
        self,
        bottleneck_dim: int = 256,
        action_dim: int = 7,
        chunk_len: int = 8,
        hidden_dim: int = 256,    # paper: 256 for most tasks, 512 for hard
        ref_dropout: float = 0.5,  # paper: 50%
        fixed_std: float = 0.1,    # paper: small fixed std
        prop_dim: int = 0,         # proprioceptive state dim (paper: eef_pos+axisangle+gripper=8)
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.ref_dropout = ref_dropout
        self.prop_dim = prop_dim

        flat_action_dim = action_dim * chunk_len
        input_dim = bottleneck_dim + prop_dim + flat_action_dim

        # Paper Appendix B: two-layer MLP (256 hidden) for most tasks,
        # three-layer MLP (512 hidden) for screw task
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_action_dim),
        )

        # Kaiming init for hidden layers (default), small normal for output layer.
        # Paper: actor directly outputs actions; BC regularization in the loss
        # (not architecture) keeps output close to VLA reference.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

        # Paper: small fixed standard deviation
        self.register_buffer("fixed_std", torch.tensor(fixed_std))

    def _get_mean(
        self,
        rl_token: torch.Tensor,
        vla_action: torch.Tensor,
        prop_state: torch.Tensor = None,
        apply_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Compute μ_θ(x, ã) — the mean action output (paper Eq. 4).

        The actor DIRECTLY outputs the full action chunk, conditioned on
        (z_rl, s_p, ã). NO residual connection — BC regularization β‖a - ã‖²
        in the loss (Eq. 5) keeps actions close to VLA reference.
        Reference-action dropout (50%) prevents identity collapse.
        """
        B = rl_token.size(0)
        rl_feat = rl_token.squeeze(1) if rl_token.dim() == 3 else rl_token  # (B, D)
        vla_flat = vla_action.reshape(B, -1)  # (B, C*A)

        # Reference-action dropout (paper Sec. IV-B):
        # zero out VLA ref input for a fraction of the batch
        # 50% 的概率将 VLA 参考动作置零,防止 Actor 退化到恒等映射（即直接复制 VLA 输出而不学习）,迫使 Actor 即使在缺失 VLA 参考时也能独立决策
        if apply_dropout and self.ref_dropout > 0:
            mask = (torch.rand(B, 1, device=rl_feat.device) > self.ref_dropout).float()
            vla_flat_input = vla_flat * mask
        else:
            vla_flat_input = vla_flat

        if self.prop_dim > 0:
            if prop_state is None:
                prop_state = torch.zeros(B, self.prop_dim, device=rl_feat.device,
                                         dtype=rl_feat.dtype)
            x = torch.cat([rl_feat, prop_state, vla_flat_input], dim=-1)
        else:
            x = torch.cat([rl_feat, vla_flat_input], dim=-1)

        raw_output = self.net(x)  # (B, C*A)
        return raw_output.reshape(B, self.chunk_len, self.action_dim)

    def forward(
        self,
        rl_token: torch.Tensor,
        vla_action: torch.Tensor,
        prop_state: torch.Tensor = None,
        deterministic: bool = False,
    ):
        """
        Args:
            rl_token: (B, 1, D) or (B, D)
            vla_action: (B, chunk_len, action_dim) — VLA reference
            prop_state: (B, prop_dim) — proprioceptive state (eef_pos+axisangle+gripper)
        Returns:
            action: (B, chunk_len, action_dim)
            log_prob: (B,) or None if deterministic
        """
        mean = self._get_mean(rl_token, vla_action, prop_state,
                              apply_dropout=(self.training and not deterministic))

        if deterministic:
            return mean, None

        std = self.fixed_std.expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        # VAE 中常用的重参数化技巧。这样即使动作是随机采样的，梯度也能从 Q 网络流回 Actor 的均值网络 μ θ
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=(-2, -1))  # (B,)
        return action, log_prob

    def log_prob_of(
        self,
        rl_token: torch.Tensor,
        vla_action: torch.Tensor,
        taken_action: torch.Tensor,
        prop_state: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute log_prob of a previously taken action under current policy."""
        mean = self._get_mean(rl_token, vla_action, prop_state, apply_dropout=False)
        std = self.fixed_std.expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(taken_action).sum(dim=(-2, -1))  # (B,)


def soft_update_target(source: nn.Module, target: nn.Module, tau: float = 0.005):
    """Polyak averaging: target = (1 - tau) * target + tau * source."""
    with torch.no_grad():
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

"""
含义： 论文 Eq.3 的 Twin-Q Critic（TD3 风格）。它评估状态-动作对的 Q 值。
Q 网络以 (rl_token, action_chunk, prop_state) 为输入，输出标量 Q 值。包含两个独立
的 Q 网络（q1、q2），目标 Q 值取二者最小值以防止 Q 值高估（overestimation bias）。

    输入：
        rl_token: (B, 1, D) 或 (B, D) — RL 潜在表示
        action: (B, chunk_len, action_dim) — 动作序列
        prop_state: (B, prop_dim)（可选）— 本体感知状态
    输出：
        q1: (B,) — 第一个 Q 网络的输出
        q2: (B,) — 第二个 Q 网络的输出
    额外方法： q1_forward() 仅计算 Q1，用于 Actor 的更新以减少计算量。
"""
# 评估在当前状态下，执行这一组“动作序列”到底好不好（打分）
class ActionTokenQCritic(nn.Module):
    """
    Twin Q-critic from RL Token paper (Eq. 3, following TD3).

    Q_ψ(x, a_{1:C}) takes the RL token state AND the action chunk as input.
    Contains two independent Q-networks; use min(Q1, Q2) for target values.
    """

    def __init__(
        self,
        bottleneck_dim: int = 256,
        action_dim: int = 7,
        chunk_len: int = 8,
        hidden_dim: int = 256,
        prop_dim: int = 0,  # proprioceptive state dim
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.prop_dim = prop_dim

        flat_action_dim = action_dim * chunk_len
        input_dim = bottleneck_dim + prop_dim + flat_action_dim    # rl_token 隐空间维度 + 机器人本体感受状态维度 (prop_dim) + 展平后的动作维度

        # Twin Q-networks (TD3 style)
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        rl_token: torch.Tensor,
        action: torch.Tensor,
        prop_state: torch.Tensor = None,
    ) -> tuple:
        """
        Returns: q1: (B,), q2: (B,)
        """
        if rl_token.dim() == 3:
            rl_token = rl_token.squeeze(1)
        B = rl_token.size(0)
        action_flat = action.reshape(B, -1)
        if self.prop_dim > 0:
            if prop_state is None:
                prop_state = torch.zeros(B, self.prop_dim, device=rl_token.device,
                                         dtype=rl_token.dtype)
            x = torch.cat([rl_token, prop_state, action_flat], dim=-1)
        else:
            x = torch.cat([rl_token, action_flat], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_forward(
        self,
        rl_token: torch.Tensor,
        action: torch.Tensor,
        prop_state: torch.Tensor = None,
    ) -> torch.Tensor:
        """Single Q1 forward (used for actor loss to save compute)."""
        if rl_token.dim() == 3:
            rl_token = rl_token.squeeze(1)
        B = rl_token.size(0)
        action_flat = action.reshape(B, -1)
        if self.prop_dim > 0:
            if prop_state is None:
                prop_state = torch.zeros(B, self.prop_dim, device=rl_token.device,
                                         dtype=rl_token.dtype)
            x = torch.cat([rl_token, prop_state, action_flat], dim=-1)
        else:
            x = torch.cat([rl_token, action_flat], dim=-1)
        return self.q1(x).squeeze(-1)


# ── Keep old V(s) critic for backward compatibility with PPO path ──
"""
含义： 传统的状态值函数 V(s)（Legacy 实现，仅用于 PPO 路径）。它以 RL token 为输入，
直接估计当前状态的标量价值。与 QCritic 不同，它不依赖动作，只评估状态本身。
    输入： rl_token: (B, 1, D) 或 (B, D)
    输出： value: (B,) — 每个样本的状态值估计
"""
class ActionTokenCritic(nn.Module):
    """State value estimator V(s) from rl_token. (Legacy, for PPO path only.)"""

    def __init__(
        self,
        bottleneck_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rl_token: torch.Tensor) -> torch.Tensor:
        if rl_token.dim() == 3:
            rl_token = rl_token.squeeze(1)
        return self.net(rl_token).squeeze(-1)
