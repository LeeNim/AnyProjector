"""
projector.py - Kiến trúc mạng AnyProjector (Phase 1).

Mạng cầu nối (Projector) tự động co giãn theo encoder_dim và llm_dim.

Kiến trúc:
    ┌──────────────────────┐
    │  Encoder Output      │  (batch, seq_len, encoder_dim)
    └──────────┬───────────┘
               │
    ┌──────────▼───────────┐
    │  Temporal Compression│  Conv1d(stride=2) → giảm 50% token
    │  (encoder_dim)       │  (batch, seq_len//2, encoder_dim)
    └──────────┬───────────┘
               │
        ┌──────┴──────┐
        │             │
   ┌────▼────┐   ┌────▼────┐
   │Semantic │   │  VAD    │
   │ Route   │   │ Route   │
   │Linear   │   │Linear   │
   │(→llm_d) │   │(→1)     │
   └────┬────┘   └────┬────┘
        │             │
   (batch,seq//2,  (batch, 1)
    llm_dim)       σ → [0,1]
"""

import torch
import torch.nn as nn


class AnyProjector(nn.Module):
    """Projector mạng cầu nối giữa Audio Encoder và LLM.

    Tự động co giãn theo kích thước hidden của encoder và LLM.
    Chỉ module này có requires_grad=True trong quá trình huấn luyện.

    Args:
        encoder_dim: Kích thước hidden output của Audio Encoder.
        llm_dim: Kích thước hidden input của LLM.
        conv_kernel_size: Kernel size cho Conv1d temporal compression.
    """

    def __init__(self, encoder_dim: int, llm_dim: int, conv_kernel_size: int = 3):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim

        # --- Nén thời gian (Temporal Compression) ---
        # Conv1d stride=2 → giảm 50% số lượng token truyền tải
        # padding = kernel_size // 2 để giữ output gần đúng seq_len // 2
        self.temporal_conv = nn.Conv1d(
            in_channels=encoder_dim,
            out_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            stride=2,
            padding=conv_kernel_size // 2,
        )
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(encoder_dim)

        # --- Nhánh LLM (Semantic Route) ---
        # Chiếu không gian âm thanh → không gian ngữ nghĩa LLM
        self.semantic_proj = nn.Linear(encoder_dim, llm_dim)

        # --- Nhánh VAD (Control Route) ---
        # Nhả ra xác suất nhị phân: 0 = Đang nói, 1 = Đã nói xong / Ngắt
        self.vad_head = nn.Linear(encoder_dim, 1)

    def forward(
        self, encoder_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass qua Projector.

        Args:
            encoder_output: Output từ Audio Encoder.
                Shape: (batch, seq_len, encoder_dim)

        Returns:
            semantic_embeds: Vector ngữ nghĩa cho LLM inputs_embeds.
                Shape: (batch, seq_len // 2, llm_dim)
            vad_prob: Xác suất VAD (kết thúc câu nói).
                Shape: (batch, 1), giá trị trong [0, 1]
        """
        # --- Temporal Compression ---
        # Conv1d cần input shape: (batch, channels, seq_len)
        x = encoder_output.transpose(1, 2)  # (batch, encoder_dim, seq_len)
        x = self.temporal_conv(x)  # (batch, encoder_dim, seq_len // 2)
        x = x.transpose(1, 2)  # (batch, seq_len // 2, encoder_dim)

        # Activation + LayerNorm
        x = self.activation(x)
        x = self.layer_norm(x)

        # --- Semantic Route ---
        semantic_embeds = self.semantic_proj(x)  # (batch, seq_len // 2, llm_dim)

        # --- VAD Route ---
        # Global average pooling over time → xác suất duy nhất cho cả utterance
        vad_logit = self.vad_head(x.mean(dim=1))  # (batch, 1)
        vad_prob = torch.sigmoid(vad_logit)

        return semantic_embeds, vad_prob

    def count_parameters(self) -> int:
        """Đếm tổng số tham số trainable."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"AnyProjector(\n"
            f"  encoder_dim={self.encoder_dim}, llm_dim={self.llm_dim}\n"
            f"  temporal_conv=Conv1d({self.encoder_dim}→{self.encoder_dim}, stride=2)\n"
            f"  semantic_proj=Linear({self.encoder_dim}→{self.llm_dim})\n"
            f"  vad_head=Linear({self.encoder_dim}→1)\n"
            f"  trainable_params={params:,}\n"
            f")"
        )
