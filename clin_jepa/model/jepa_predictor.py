"""Cross-attention head that predicts the embeddings of masked hours from the visible ones."""

from __future__ import annotations

import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """Cross-attention decoder over the visible hours, queried at the positions of the masked ones."""

    def __init__(
        self,
        embed_dim: int = 4096,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_position_id: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_position_id = max_position_id

        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        self.pos_embed = nn.Embedding(max_position_id, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, embed_dim)

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

    def forward(
        self,
        memory_embeddings: torch.Tensor,
        memory_positions: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention forward pass."""
        memory = self.input_proj(memory_embeddings.float())
        memory = memory + self.pos_embed(memory_positions)

        queries = self.pos_embed(query_positions)

        out = self.decoder(tgt=queries, memory=memory)

        return self.output_proj(out)
