"""
system.py - Khởi tạo và quản lý toàn bộ hệ thống AnyProjector (Phase 1).

Orchestration:
1. Tải Audio Encoder → đóng băng 100%
2. Tải LLM Decoder → 4-bit quantization (bitsandbytes) → đóng băng 100%
3. Tạo AnyProjector (Projector) → requires_grad=True
4. Nối output Projector vào inputs_embeds của LLM
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.model_loader import ModelInfo, read_model_config, extract_hidden_size
from src.projector import AnyProjector

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """Thông tin tổng quan hệ thống sau khi khởi tạo."""
    encoder_dim: int
    llm_dim: int
    projector_params: int
    device: str
    encoder_type: str
    llm_type: str
    llm_quantization: str


def _get_device() -> torch.device:
    """Xác định device tối ưu (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info(f"GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU detected. Running on CPU (training will be very slow).")
    return device


def _freeze_model(model: nn.Module, name: str) -> None:
    """Đóng băng 100% tham số của model.

    Args:
        model: Model cần freeze.
        name: Tên model (dùng cho logging).
    """
    total_params = 0
    for param in model.parameters():
        param.requires_grad = False
        total_params += param.numel()
    logger.info(f"Frozen {name}: {total_params:,} parameters (requires_grad=False)")


class AnyProjectorSystem:
    """Hệ thống AnyProjector hoàn chỉnh.

    Quản lý vòng đời của:
    - Audio Encoder (frozen)
    - LLM Decoder (4-bit, frozen)
    - AnyProjector module (trainable)

    Usage:
        system = AnyProjectorSystem(encoder_info, llm_info)
        system.build()
        # system.encoder, system.llm, system.projector sẵn sàng
    """

    def __init__(self, encoder_info: ModelInfo, llm_info: ModelInfo):
        """
        Args:
            encoder_info: ModelInfo của Audio Encoder (từ Phase 0).
            llm_info: ModelInfo của LLM Decoder (từ Phase 0).
        """
        self.encoder_info = encoder_info
        self.llm_info = llm_info

        # Sẽ được gán sau khi build()
        self.device: torch.device | None = None
        self.encoder: nn.Module | None = None
        self.processor: AutoProcessor | None = None
        self.llm: nn.Module | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.projector: AnyProjector | None = None
        self._built = False

    def build(self, progress_callback=None) -> SystemInfo:
        """Khởi tạo toàn bộ hệ thống.

        Workflow:
        1. Detect device (GPU/CPU)
        2. Load & freeze Audio Encoder
        3. Load & freeze LLM (4-bit)
        4. Create AnyProjector (trainable)

        Args:
            progress_callback: Hàm callback báo tiến trình.

        Returns:
            SystemInfo chứa thông tin tổng quan.
        """
        if self._built:
            logger.warning("System already built. Skipping.")
            return self._get_system_info()

        # --- 1. Device ---
        self.device = _get_device()
        if progress_callback:
            progress_callback(f"🖥️ Device: {self.device}")

        # --- 2. Load Audio Encoder (frozen) ---
        if progress_callback:
            progress_callback("🔊 Đang tải Audio Encoder...")
        self._load_encoder()

        # --- 3. Load LLM Decoder (4-bit, frozen) ---
        if progress_callback:
            progress_callback("🧠 Đang tải LLM Decoder (4-bit quantization)...")
        self._load_llm()

        # --- 4. Create Projector (trainable) ---
        if progress_callback:
            progress_callback("🔗 Đang tạo AnyProjector...")
        self._create_projector()

        self._built = True

        info = self._get_system_info()
        if progress_callback:
            progress_callback(
                f"✅ Hệ thống sẵn sàng!\n"
                f"   Projector: {info.projector_params:,} trainable params"
            )

        return info

    def _load_encoder(self) -> None:
        """Tải Audio Encoder và đóng băng 100%.

        Hỗ trợ Whisper và các model audio khác trên HuggingFace.
        """
        model_path = str(self.encoder_info.local_path)
        model_type = self.encoder_info.model_type

        logger.info(f"Loading encoder: {self.encoder_info.hf_id} (type: {model_type})")

        # Load processor (feature extractor) cho audio input
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
        except Exception:
            logger.warning("AutoProcessor not available, trying AutoFeatureExtractor")
            from transformers import AutoFeatureExtractor
            self.processor = AutoFeatureExtractor.from_pretrained(model_path)

        # Whisper dùng encoder-decoder, ta chỉ cần encoder
        if "whisper" in model_type.lower():
            from transformers import WhisperModel
            full_model = WhisperModel.from_pretrained(model_path)
            self.encoder = full_model.encoder
            del full_model
        else:
            # Model audio encoder khác (Wav2Vec2, HuBERT, etc.)
            self.encoder = AutoModel.from_pretrained(model_path)

        # Đóng băng 100%
        _freeze_model(self.encoder, "Audio Encoder")

        # Chuyển lên device
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        logger.info("Audio Encoder loaded and frozen.")

    def _load_llm(self) -> None:
        """Tải LLM Decoder ở 4-bit quantization và đóng băng 100%.

        Sử dụng bitsandbytes cho 4-bit quantization (QLoRA-ready).
        """
        model_path = str(self.llm_info.local_path)

        logger.info(f"Loading LLM: {self.llm_info.hf_id}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit quantization config (bitsandbytes)
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self._llm_quantization = "4-bit (NF4, double quant)"
        else:
            # CPU fallback — no quantization
            logger.warning("No GPU: loading LLM in float32 (no quantization)")
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            )
            self.llm = self.llm.to(self.device)
            self._llm_quantization = "none (CPU float32)"

        # Đóng băng 100%
        _freeze_model(self.llm, "LLM Decoder")
        self.llm.eval()
        logger.info("LLM Decoder loaded, quantized, and frozen.")

    def _create_projector(self) -> None:
        """Tạo AnyProjector với kích thước tự động từ encoder/LLM config.

        Chỉ module này có requires_grad=True.
        """
        encoder_dim = self.encoder_info.hidden_size
        llm_dim = self.llm_info.hidden_size

        logger.info(
            f"Creating AnyProjector: encoder_dim={encoder_dim}, llm_dim={llm_dim}"
        )

        self.projector = AnyProjector(encoder_dim=encoder_dim, llm_dim=llm_dim)
        self.projector = self.projector.to(self.device)
        self.projector.train()

        logger.info(f"Projector created: {self.projector}")

    def forward_projector(
        self, encoder_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chạy encoder output qua Projector → sẵn sàng nối vào LLM.

        Args:
            encoder_output: Output từ encoder. (batch, seq_len, encoder_dim)

        Returns:
            semantic_embeds: Cho LLM inputs_embeds. (batch, seq_len//2, llm_dim)
            vad_prob: Xác suất VAD. (batch, 1)
        """
        if not self._built:
            raise RuntimeError("System not built. Call build() first.")
        return self.projector(encoder_output)

    def get_llm_embedding_layer(self) -> nn.Embedding:
        """Lấy embedding layer của LLM (dùng để nối inputs_embeds).

        Returns:
            nn.Embedding layer của LLM.
        """
        if not self._built:
            raise RuntimeError("System not built. Call build() first.")
        return self.llm.get_input_embeddings()

    def _get_system_info(self) -> SystemInfo:
        """Trả về thông tin tổng quan hệ thống."""
        return SystemInfo(
            encoder_dim=self.encoder_info.hidden_size,
            llm_dim=self.llm_info.hidden_size,
            projector_params=self.projector.count_parameters() if self.projector else 0,
            device=str(self.device),
            encoder_type=self.encoder_info.model_type or "unknown",
            llm_type=self.llm_info.model_type or "unknown",
            llm_quantization=getattr(self, "_llm_quantization", "unknown"),
        )
