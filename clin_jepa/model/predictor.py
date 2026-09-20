"""Latent trajectory predictor: a block-causal Transformer over (demographics, state, action) embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_block_causal_mask(num_timesteps: int, device: torch.device) -> torch.Tensor:
    """Additive mask over ``[d, s_1, a_1, ...]``: a token sees the demographics token and every hour up to its own."""
    hour = (torch.arange(2 * num_timesteps + 1, device=device) + 1) // 2
    mask = torch.zeros(len(hour), len(hour), device=device)
    return mask.masked_fill(hour.unsqueeze(0) > hour.unsqueeze(1), float("-inf"))


class ACTransformerPredictor(nn.Module):
    """Action-conditioned Transformer predictor (~92.5M params)."""

    def __init__(
        self,
        state_dim: int = 4096,
        action_dim: int = 4096,
        static_dim: int = 4096,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 4096,
        max_timesteps: int = 72,
        ffn_dropout: float = 0.15,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.max_timesteps = max_timesteps

        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.static_proj = nn.Linear(static_dim, hidden_dim)

        max_seq_len = 2 * (max_timesteps + 2) + 1
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        encoder_layer.self_attn.dropout = attn_dropout
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, state_dim)

        self._cached_masks: dict[int, torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def _get_block_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if T not in self._cached_masks or self._cached_masks[T].device != device:
            self._cached_masks[T] = build_block_causal_mask(T, device)
        return self._cached_masks[T]

    def _build_padding_mask(self, lengths: torch.Tensor, T_max: int, device: torch.device) -> torch.Tensor:
        """True at the tokens past each trajectory's own length."""
        positions = torch.arange(2 * T_max + 1, device=device)
        return positions.unsqueeze(0) > 2 * lengths.to(device).unsqueeze(1)

    def forward(
        self,
        z_states: torch.Tensor,
        z_static: torch.Tensor,
        z_actions: torch.Tensor,
        lengths: torch.Tensor,
        noise_std: float = 0.0,
        return_last_only: bool = False,
    ) -> torch.Tensor:
        """Predict next-state embeddings."""
        B, T_max, _ = z_states.shape
        device = z_states.device

        z_states_f = z_states.float()
        z_actions_f = z_actions.float()
        z_static_f = z_static.float()

        if noise_std > 0.0 and self.training:
            z_states_f = z_states_f + torch.randn_like(z_states_f) * noise_std
            z_actions_f = z_actions_f + torch.randn_like(z_actions_f) * noise_std

        s = self.state_proj(z_states_f)
        a = self.action_proj(z_actions_f)
        d = self.static_proj(z_static_f)

        seq_len = 2 * T_max + 1
        tokens = torch.zeros(B, seq_len, self.hidden_dim, device=device, dtype=s.dtype)
        tokens[:, 0, :] = d
        tokens[:, 1::2, :] = s
        tokens[:, 2::2, :] = a

        pos_ids = torch.arange(seq_len, device=device)
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0)

        block_mask = self._get_block_causal_mask(T_max, device)
        padding_mask = self._build_padding_mask(lengths, T_max, device)

        out = self.transformer(
            tokens,
            mask=block_mask,
            src_key_padding_mask=padding_mask,
        )

        state_out = out[:, 1::2, :]

        preds = self.output_proj(state_out)
        return preds[:, -1, :] if return_last_only else preds[:, :-1, :]

    def forward_rollout_step(
        self,
        z_history: torch.Tensor,
        z_static: torch.Tensor,
        za_history: torch.Tensor,
        za_next: torch.Tensor,
        z_last: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single autoregressive rollout step."""
        z_seq = torch.cat([z_history, z_last.unsqueeze(1)], dim=1)
        za_seq = torch.cat([za_history, za_next.unsqueeze(1)], dim=1)
        lengths = torch.full(
            (z_seq.shape[0],), z_seq.shape[1],
            device=z_seq.device, dtype=torch.long,
        )

        return self.forward(
            z_states=z_seq,
            z_static=z_static,
            z_actions=za_seq,
            lengths=lengths,
            return_last_only=True,
        )
