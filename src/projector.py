"""
projector.py - Kiến trúc mạng AnyProjector (Q-Former).

Q-Former (BLIP-2 style) bridge giữa Audio Encoder và LLM.
Dùng learnable query tokens + cross-attention để nén encoder output
thành số lượng tokens cố định, bất kể audio length.

Kiến trúc:
    ┌──────────────────────────────────┐
    │  Encoder Output (1500, enc_dim)  │  ← Whisper (pad 30s)
    │  + Attention Mask                │  ← Chỉ real tokens, bỏ pad
    └──────────────┬───────────────────┘
                   │
    ┌──────────────▼───────────────────┐
    │  Pre-Projection                  │
    │  Linear(enc_dim → enc_dim)       │
    │  + GELU + LayerNorm              │
    │  (transform features trước khi   │
    │   Q-Former compress)             │
    └──────────────┬───────────────────┘
                   │  Key, Value
    ┌──────────────▼───────────────────┐
    │  Learnable Queries (64, qf_dim)  │
    │  ┌─────────────────────────────┐ │
    │  │ Self-Attention              │ │
    │  │ Cross-Attention (to encoder)│ │
    │  │ Feed-Forward Network        │ │
    │  └─────────────────────────────┘ │
    │         × num_layers             │
    └──────────────┬───────────────────┘
                   │
    ┌──────────────▼───────────────────┐
    │  Output Projection               │
    │  Linear(qf_dim → llm_dim)        │
    └──────────────┬───────────────────┘
                   │
                   ▼
    (batch, 64, llm_dim) → LLM inputs_embeds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QFormerLayer(nn.Module):
    """Single Q-Former layer: Self-Attn → Cross-Attn → FFN.

    Supports optional dropout (v0.9.7+) for regularization.
    During inference (.eval()), dropout is automatically disabled.
    """

    def __init__(self, qformer_dim: int, encoder_dim: int, num_heads: int = 8,
                 ffn_ratio: int = 4, dropout: float = 0.0):
        super().__init__()

        # Self-Attention (queries attend to each other)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=qformer_dim, num_heads=num_heads, batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(qformer_dim)
        self.self_attn_drop = nn.Dropout(dropout)

        # Cross-Attention (queries attend to encoder output)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=qformer_dim, num_heads=num_heads,
            kdim=encoder_dim, vdim=encoder_dim, batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(qformer_dim)
        self.cross_attn_drop = nn.Dropout(dropout)

        # Feed-Forward Network (with dropout between GELU and second Linear)
        ffn_hidden = qformer_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(qformer_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, qformer_dim),
        )
        self.ffn_norm = nn.LayerNorm(qformer_dim)

    def forward(self, queries: torch.Tensor, encoder_out: torch.Tensor,
                encoder_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            queries: (batch, num_queries, qformer_dim)
            encoder_out: (batch, enc_seq_len, encoder_dim)
            encoder_mask: (batch, enc_seq_len) — True = pad (ignored),
                          False = real token. Used as key_padding_mask.
        Returns:
            queries: (batch, num_queries, qformer_dim)
        """
        # Self-Attention + residual
        q = self.self_attn_norm(queries)
        q, _ = self.self_attn(q, q, q)
        queries = queries + self.self_attn_drop(q)

        # Cross-Attention + residual
        q = self.cross_attn_norm(queries)
        q, _ = self.cross_attn(
            query=q, key=encoder_out, value=encoder_out,
            key_padding_mask=encoder_mask,
        )
        queries = queries + self.cross_attn_drop(q)

        # FFN + residual
        queries = queries + self.ffn(self.ffn_norm(queries))

        return queries


class AnyProjector(nn.Module):
    """Q-Former Projector — bridge giữa Audio Encoder và LLM.

    Dùng learnable query tokens + cross-attention để nén encoder output
    thành số lượng tokens cố định. Hỗ trợ attention mask để bỏ qua
    padding tokens từ encoder (Whisper pad 30s).

    Args:
        encoder_dim: Hidden size của Audio Encoder (e.g. 768, 1024).
        llm_dim: Hidden size của LLM (e.g. 1536, 3072).
        num_queries: Số learnable query tokens (output length).
        qformer_dim: Hidden dim bên trong Q-Former.
        num_layers: Số Q-Former layers (self-attn + cross-attn + FFN).
        num_heads: Số attention heads.
    """

    def __init__(self, encoder_dim: int, llm_dim: int,
                 num_queries: int = 64, qformer_dim: int = 768,
                 num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.num_queries = num_queries
        self.qformer_dim = qformer_dim

        # Pre-projection: transform encoder features before Q-Former
        # Decouples "feature transformation" from "information compression"
        self.pre_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.LayerNorm(encoder_dim),
        )

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, qformer_dim) * 0.02
        )

        # Q-Former transformer layers
        self.layers = nn.ModuleList([
            QFormerLayer(qformer_dim, encoder_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm before projection
        self.output_norm = nn.LayerNorm(qformer_dim)

        # Project Q-Former dim → LLM dim
        self.output_proj = nn.Sequential(
            nn.Linear(qformer_dim, llm_dim),
        )

    def forward(self, encoder_output: torch.Tensor,
                encoder_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass qua Q-Former.

        Args:
            encoder_output: Output từ Audio Encoder.
                Shape: (batch, enc_seq_len, encoder_dim)
            encoder_mask: Padding mask cho encoder tokens.
                Shape: (batch, enc_seq_len)
                True = padding (bỏ qua), False = real token.
                None = không mask (attend tất cả).

        Returns:
            projected_queries: Output embeddings cho LLM.
                Shape: (batch, num_queries, llm_dim)
        """
        batch_size = encoder_output.shape[0]

        # Pre-project encoder features (transform before compress)
        encoder_output = self.pre_proj(encoder_output)

        # Expand queries for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)

        # Pass through Q-Former layers
        for layer in self.layers:
            queries = layer(queries, encoder_output, encoder_mask)

        # Normalize + project to LLM space
        queries = self.output_norm(queries)
        projected = self.output_proj(queries)

        return projected

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        n_layers = len(self.layers)
        return (
            f"AnyProjector(Q-Former)\n"
            f"  encoder_dim={self.encoder_dim}, llm_dim={self.llm_dim}\n"
            f"  pre_proj=Linear({self.encoder_dim}->{self.encoder_dim})+GELU+LN\n"
            f"  queries={self.num_queries}, qformer_dim={self.qformer_dim}\n"
            f"  layers={n_layers}, output=Linear({self.qformer_dim}->{self.llm_dim})\n"
            f"  trainable_params={params:,}"
        )
