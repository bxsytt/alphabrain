"""
ActionToken Encoder-Decoder: Information bottleneck between frozen VLA and
small RL network.

Inspired by the "RL Token" paper (Physical Intelligence, 2026), but with
deviations from the paper's construction:
  - Encoder input is the VLA's action-query hidden states (M × H) gathered at
    the action-token positions, not the full image-token sequence (N × H) as
    in the paper's Fig. 2.
  - An extra `Linear(H → bottleneck_dim)` projection compresses per-token dim
    (e.g. 2048 → 256); the paper keeps the RL token at the VLA hidden dim.
  - The decoder is a self-attention transformer with a causal mask and a
    prefix token, not the encoder-decoder cross-attention structure of the
    paper's Eq. 2.
A faithful paper-accurate reimplementation is still under test.

Paper Eq. 1 — Encoder:
  z_rl = g_φ([z_{1:M}, e_rl])_{M+1}
  Append learnable embedding e_rl to VLA token sequence, run through
  self-attention encoder transformer, take the e_rl position output.

Paper Eq. 2 — Decoder (autoregressive reconstruction):
  L_ro = E[ Σ_{i=1}^{M} ‖h_φ(d_φ([z_rl, sg(z_{1:i-1})]))_i − sg(z_i)‖² ]
  Reconstruct VLA tokens autoregressively from z_rl to enforce information
  preservation in the bottleneck.
"""

"""
Actor-Critic 网络定义 — 包含 ActionTokenActor（π_θ网络，输入rl_token + VLA参考动作 + 本体感知，输出动作）、
ActionTokenCritic（V(s)值函数，PPO路径使用）、ActionTokenQCritic（Twin-Q网络，TD3路径使用）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
含义： 论文 Eq.1 的实现。在 action queries 序列末尾添加一个可学习的 e_rl 嵌入，通过多层
自注意力 Transformer 处理整个序列，取 e_rl 位置的输出，再通过线性投影压缩到瓶颈维度。

    输入： action_queries: (B, M, H)
    输出： rl_token: (B, 1, D_bottleneck)
"""
class ActionTokenEncoder(nn.Module):
    """
    Paper Eq. 1: Compress VLA action_queries (B, M, H) → rl_token (B, 1, D).

    Appends a learnable e_rl to the token sequence and processes with
    self-attention (TransformerEncoderLayer). The output at the e_rl
    position is projected to the bottleneck dimension.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Learnable RL embedding e_rl (appended to token sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)    # [1,1,2048],

        # Self-attention encoder layers (paper: g_φ processes [z_{1:M}, e_rl])
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=input_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.bottleneck_proj = nn.Linear(input_dim, bottleneck_dim)

    """
    输入: action_queries (B, M, H)
        1. 附加可学习 e_rl token: → (B, M+1, H)
        2. 多层 Transformer self-attention
        3. 取 e_rl 位置输出 → Linear(H → D_bottleneck)
    输出: rl_token (B, 1, D_bottleneck)

    附加一个可学习的 e_rl token 到 action_queries 序列末尾，经过多层 Transformer self-attention，
    取 e_rl 位置的输出，再通过线性投影压缩到瓶颈维度 → 得到 rl_token (B, 1, D_bottleneck)
    """
    def forward(self, action_queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_queries: (B, M, H) from frozen VLA
        Returns:
            rl_token: (B, 1, D_bottleneck)
        """
        action_queries = action_queries.float()
        B = action_queries.size(0)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, H)
        seq = torch.cat([action_queries, cls], dim=1)         # (B, M+1, H)
        for layer in self.self_attn_layers:
            seq = layer(seq)                                  # (B, M+1, H)
        rl_token = self.bottleneck_proj(seq[:, -1:, :])       # (B, 1, D)
        return rl_token

"""
含义： 论文 Eq.2 的实现。以 RL token 为前缀，自回归地逐步重构每个 VLA token。
训练时使用 Teacher Forcing（输入 sg(z_{1:i-1}) 预测 z_i），推理时使用位置嵌入。

    输入：
        rl_token: (B, 1, D_bottleneck)
        target_tokens: (B, M, H)（可选，训练时提供用于 Teacher Forcing）
    输出： reconstructed: (B, M, H) — 每个位置对应原始 VLA token 的重构
"""
class ActionTokenDecoder(nn.Module):
    """
    Paper Eq. 2: Autoregressive reconstruction of VLA tokens from z_rl.

    L_ro = E[ Σ_i ‖h_φ(d_φ([z_rl, sg(z_{1:i-1})]))_i − sg(z_i)‖² ]

    The decoder takes z_rl as prefix, and autoregressively reconstructs
    each VLA token conditioned on z_rl and previously reconstructed tokens.
    A causal mask ensures position i can only attend to positions < i
    (plus the z_rl prefix which is always visible).
    """

    def __init__(
        self,
        bottleneck_dim: int = 256,
        output_dim: int = 2048,
        chunk_len: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.output_dim = output_dim
        self.expand_proj = nn.Linear(bottleneck_dim, output_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, chunk_len, output_dim) * 0.02)

        # Self-attention decoder layers with causal masking
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=num_heads,
                dim_feedforward=output_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

    """
    输入: rl_token (B, 1, D_bottleneck) + target_tokens (B, M, H)
        1. 投影 rl_token 回到 H 维: → (B, 1, H)
        2. Teacher Forcing: 输入序列 = [z_rl, sg(z_1), sg(z_2), ..., sg(z_{M-1})]
        - sg = stop-gradient，即 .detach()
        3. 加位置编码
        4. Causal mask 确保自回归结构
        5. 多层 Transformer self-attention
    输出: reconstructed (B, M, H) — 每个位置对应原始 z_i 的重构

    以 rl_token 为前缀，采用 Teacher Forcing 模式自回归重构：输入是 [z_rl, sg(z_1), sg(z_2), ..., sg(z_{M-1})]，
    输出预测 [z_1, z_2, ..., z_M]，所有目标都经过 stop-gradient（.detach()）
    """
    def forward(
        self,
        rl_token: torch.Tensor,
        target_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            rl_token: (B, 1, D_bottleneck)
            target_tokens: (B, M, H) stop-gradient VLA tokens for teacher forcing.
                           If None, uses learned positional embeddings (inference mode).
        Returns:
            reconstructed: (B, M, H)
        """
        B = rl_token.size(0)
        prefix = self.expand_proj(rl_token)                   # (B, 1, H)

        if target_tokens is not None:
            # Training: teacher forcing with stop-gradient targets
            # Sequence: [z_rl, sg(z_1), sg(z_2), ..., sg(z_{M-1})]
            # Target:   [z_1,  z_2,     z_3,     ...,  z_M        ]
            # Shifted input: z_rl is position 0, z_1 is position 1, etc.
            shifted_input = target_tokens[:, :-1, :].detach()  # (B, M-1, H)
            seq = torch.cat([prefix, shifted_input], dim=1)    # (B, M, H)
            seq = seq + self.pos_embed                         # add positional info
        else:
            # Inference: use positional embeddings (no teacher forcing)
            # 没有真实动作参考，直接把 prefix 复制 M 份，然后加上 位置编码 (pos_embed)，让每一位根据自己的“位置感”来生成动作。
            seq = prefix.expand(-1, self.chunk_len, -1) + self.pos_embed  # (B, M, H)

        # Causal mask: position i can only attend to positions <= i
        # This ensures autoregressive structure
        M = seq.size(1)
        # 因果掩码：确保在计算第 i 个位置的动作时，模型只能看到之前的 Token，不能“偷看”后面的 Token。
        causal_mask = torch.triu(
            torch.ones(M, M, device=seq.device, dtype=torch.bool), diagonal=1
        )  # True = masked out

        # 将序列送入多层 Transformer Block。每一层都会利用 causal_mask 进行自注意力计算，让序列中的每个位置通过上下文信息进行演化
        for layer in self.self_attn_layers:
            seq = layer(seq, src_mask=causal_mask, is_causal=True)

        # 在序列的每一个位置 i，输出的向量就是对目标动作 Token z_i 的预测
        return seq  # (B, M, H) — each position predicts the corresponding target


"""
编码器:将冻结的 VLA 模型输出的高维 action queries（B×M×H，如 2048 维）压缩 为一个紧凑的
 RL token（B×1×D_bottleneck，如 256 维）

解码器:在预训练阶段通过自回归重构（autoregressive reconstruction）从 RL token 恢复
出原始 VLA tokens，迫使瓶颈保留足够信息

输入： action_queries，形状 (B, M, H) — 冻结 VLA 模型在 action token 位置处的隐藏状态序列
输出（forward）：
    rl_token: (B, 1, D) — 压缩后的 RL 潜在表示
    recon_loss: 标量 — 自回归重构的 MSE 损失
"""
class ActionTokenEncoderDecoder(nn.Module):
    """
    Combined Encoder-Decoder for ActionToken pretraining.

    Training: autoregressive reconstruction with teacher forcing.
    Inference: encoder only (decoder not used during RL).
    """

    def __init__(
        self,
        input_dim: int = 2048,
        bottleneck_dim: int = 256,
        chunk_len: int = 8,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = ActionTokenEncoder(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            num_layers=encoder_layers,
            dropout=dropout,
        )
        self.decoder = ActionTokenDecoder(
            bottleneck_dim=bottleneck_dim,
            output_dim=input_dim,
            chunk_len=chunk_len,
            num_heads=num_heads,
            num_layers=decoder_layers,
            dropout=dropout,
        )

    def encode(self, action_queries: torch.Tensor) -> torch.Tensor:
        return self.encoder(action_queries)

    def decode(self, rl_token: torch.Tensor, target_tokens=None) -> torch.Tensor:
        return self.decoder(rl_token, target_tokens)

    def forward(self, action_queries: torch.Tensor):
        """
        Full encode-decode pass with autoregressive reconstruction loss.

        Paper Eq. 2:
          L_ro = E[ Σ_i ‖reconstructed_i − sg(z_i)‖² ]

        Returns:
            rl_token: (B, 1, D)
            recon_loss: scalar MSE reconstruction loss
        """
        action_queries = action_queries.float()
        rl_token = self.encoder(action_queries)
        # Autoregressive decode with teacher forcing
        reconstructed = self.decoder(rl_token, target_tokens=action_queries.detach())
        # 自回归重构的 MSE 损失，迫使 rl_token 瓶颈保留足够多的原始 VLA 信息
        recon_loss = F.mse_loss(reconstructed, action_queries.detach())
        return rl_token, recon_loss
