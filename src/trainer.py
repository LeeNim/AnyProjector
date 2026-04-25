"""
trainer.py - Huấn luyện đa nhiệm cho AnyProjector (Phase 2).

Multi-task Loss:
    L_total = L_LM + λ · L_VAD

    L_LM  = CrossEntropyLoss trên output LLM (căn chỉnh vector audio → text)
    L_VAD = BCELoss trên VAD prediction (phân loại kết thúc câu nói)

Chỉ cập nhật trọng số của Projector. Encoder và LLM đóng băng 100%.

Forward Pipeline (input chỉ có audio, text chỉ là target label):
    Audio → Encoder(frozen) → encoder_vectors (B, T, encoder_dim)
         → Projector(trainable) → semantic_embeds (B, T//2, llm_dim) + vad_prob (B, 1)
         → semantic_embeds vào LLM(frozen) qua inputs_embeds
         → LLM sinh ra câu trả lời (text)
         → Loss = CE(logits, target_text) + λ · BCE(vad_prob, vad_label)

    Khi huấn luyện: dùng teacher forcing (kỹ thuật nội bộ) để LLM học
    dự đoán text từ audio context. Text target được nối vào SAU audio embeds
    chỉ để tính loss — KHÔNG phải input của hệ thống.

    Khi inference: chỉ cần audio → encoder → projector → LLM.generate()
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.system import AnyProjectorSystem

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Cấu hình huấn luyện Phase 2."""

    # --- Optimizer ---
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)

    # --- Scheduler ---
    warmup_steps: int = 100
    num_epochs: int = 10

    # --- Loss ---
    vad_loss_weight: float = 0.5  # λ cho L_VAD

    # --- Training ---
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    log_interval: int = 10  # Log mỗi N steps

    # --- Checkpoint ---
    save_dir: str = "./checkpoints"
    save_every_epoch: bool = True


@dataclass
class TrainingMetrics:
    """Metrics cho 1 epoch."""

    epoch: int = 0
    total_loss: float = 0.0
    lm_loss: float = 0.0
    vad_loss: float = 0.0
    num_steps: int = 0
    elapsed_seconds: float = 0.0

    @property
    def avg_total_loss(self) -> float:
        return self.total_loss / max(self.num_steps, 1)

    @property
    def avg_lm_loss(self) -> float:
        return self.lm_loss / max(self.num_steps, 1)

    @property
    def avg_vad_loss(self) -> float:
        return self.vad_loss / max(self.num_steps, 1)

    def summary(self) -> str:
        return (
            f"Epoch {self.epoch} | "
            f"Loss: {self.avg_total_loss:.4f} "
            f"(LM: {self.avg_lm_loss:.4f}, VAD: {self.avg_vad_loss:.4f}) | "
            f"Steps: {self.num_steps} | "
            f"Time: {self.elapsed_seconds:.1f}s"
        )


class AlignmentTrainer:
    """Trainer cho Phase 2: Multi-task Alignment Training.

    Chỉ cập nhật tham số của Projector.
    Encoder và LLM hoàn toàn đóng băng.

    Usage:
        trainer = AlignmentTrainer(system, config)
        for epoch in range(config.num_epochs):
            metrics = trainer.train_epoch(dataloader, epoch)
            trainer.save_checkpoint(epoch)
    """

    def __init__(self, system: AnyProjectorSystem, config: TrainingConfig):
        """
        Args:
            system: AnyProjectorSystem đã build() xong.
            config: TrainingConfig.
        """
        if not system._built:
            raise RuntimeError("System chưa build(). Gọi system.build() trước.")

        self.system = system
        self.config = config
        self.device = system.device

        # Loss functions
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.vad_criterion = nn.BCELoss()

        # Optimizer — CHỈ cập nhật Projector
        self.optimizer = torch.optim.AdamW(
            system.projector.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        # Scheduler (linear warmup + cosine decay)
        self.scheduler = None  # Sẽ tạo khi biết total_steps

        # Tracking
        self.global_step = 0

        logger.info(
            f"AlignmentTrainer initialized:\n"
            f"  LR: {config.learning_rate}\n"
            f"  VAD weight (λ): {config.vad_loss_weight}\n"
            f"  Grad accumulation: {config.gradient_accumulation_steps}\n"
            f"  Trainable params: {system.projector.count_parameters():,}"
        )

    def _setup_scheduler(self, total_steps: int) -> None:
        """Tạo learning rate scheduler với warmup.

        Warmup linear → Cosine decay.
        """
        from torch.optim.lr_scheduler import LambdaLR
        import math

        warmup_steps = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        logger.info(
            f"Scheduler created: warmup={warmup_steps}, total={total_steps} steps"
        )

    def _process_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Xử lý 1 batch qua toàn bộ pipeline.

        Pipeline (input CHỈ CÓ audio, text CHỈ LÀ target label):
            1. Audio waveform → Processor → Encoder(frozen) → encoder_vectors
            2. encoder_vectors → Projector(trainable) → semantic_embeds (llm_dim) + vad_prob
            3. semantic_embeds vào LLM(frozen) qua inputs_embeds → LLM sinh text
            4. Loss = CE(logits vs target_text) + λ·BCE(vad_prob vs vad_label)

        Teacher forcing: text target được nối SAU audio embeds để LLM học
        dự đoán next token. Loss chỉ tính trên vị trí text, không tính trên
        vị trí audio. Tại inference, chỉ cần audio → LLM.generate().

        Args:
            batch: Dict từ collate_alignment.

        Returns:
            (total_loss, lm_loss_value, vad_loss_value) — scalar tensors.
        """
        waveforms = batch["waveforms"].to(self.device)       # (B, max_samples)
        vad_labels = batch["is_end_of_speech"].to(self.device)  # (B,)

        # Text chỉ dùng làm TARGET LABEL, không phải input hệ thống
        target_texts = batch["texts"]                          # list[str]

        # ================================================================
        # BƯỚC 1: Audio → Encoder (frozen) → encoder vectors
        # ================================================================
        with torch.no_grad():
            audio_inputs = self.system.processor(
                waveforms.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            audio_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in audio_inputs.items()
            }
            encoder_outputs = self.system.encoder(**audio_inputs)

            if hasattr(encoder_outputs, "last_hidden_state"):
                encoder_vectors = encoder_outputs.last_hidden_state  # (B, T, encoder_dim)
            else:
                encoder_vectors = encoder_outputs[0]

        # ================================================================
        # BƯỚC 2: Encoder vectors → Projector (trainable) → llm_dim vectors
        # Gradient CHỈ chảy qua Projector (encoder đã frozen)
        # ================================================================
        semantic_embeds, vad_prob = self.system.projector(encoder_vectors)
        # semantic_embeds: (B, T//2, llm_dim) — vector cùng dim với LLM
        # vad_prob: (B, 1) — xác suất kết thúc câu nói

        # ================================================================
        # BƯỚC 3: semantic_embeds → LLM (frozen) → LLM sinh text
        #
        # Teacher forcing (kỹ thuật training):
        #   - Audio embeds = "context" (đầu vào thực sự của hệ thống)
        #   - Target text embeds = nối SAU audio để LLM học dự đoán next token
        #   - Labels: [-100 cho vị trí audio, text_ids cho vị trí text]
        #   → Loss chỉ tính trên phần text mà LLM cần sinh ra
        #
        # Tại inference: chỉ dùng semantic_embeds → LLM.generate()
        # ================================================================
        with torch.no_grad():
            target_tokens = self.system.tokenizer(
                target_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Embed target text qua LLM embedding layer (chỉ để teacher forcing)
            target_embeds = self.system.get_llm_embedding_layer()(
                target_tokens["input_ids"]
            )  # (B, text_len, llm_dim)

        audio_seq_len = semantic_embeds.shape[1]
        batch_size = semantic_embeds.shape[0]

        # Nối: [audio_context | target_text] — teacher forcing sequence
        llm_input_embeds = torch.cat(
            [semantic_embeds, target_embeds], dim=1
        )  # (B, audio_len + text_len, llm_dim)

        # Labels: -100 = ignore (vị trí audio), text_ids = target (vị trí text)
        audio_ignore = torch.full(
            (batch_size, audio_seq_len),
            fill_value=-100, dtype=torch.long, device=self.device,
        )
        labels = torch.cat(
            [audio_ignore, target_tokens["input_ids"]], dim=1
        )  # (B, audio_len + text_len)

        # Attention mask
        audio_attn = torch.ones(
            (batch_size, audio_seq_len),
            dtype=torch.long, device=self.device,
        )
        attn_mask = torch.cat(
            [audio_attn, target_tokens["attention_mask"]], dim=1
        )

        # Forward qua LLM — gradient chảy ngược qua inputs_embeds → projector
        llm_outputs = self.system.llm(
            inputs_embeds=llm_input_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )

        # ================================================================
        # BƯỚC 4: Compute Losses
        # ================================================================
        # L_LM: CE loss — LLM học sinh text từ audio context
        lm_loss = llm_outputs.loss

        # L_VAD: BCE loss — Projector học phân loại kết thúc câu
        vad_loss = self.vad_criterion(vad_prob.squeeze(-1), vad_labels)

        # L_total = L_LM + λ · L_VAD
        total_loss = lm_loss + self.config.vad_loss_weight * vad_loss

        return total_loss, lm_loss.detach(), vad_loss.detach()

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        progress_callback=None,
    ) -> TrainingMetrics:
        """Huấn luyện 1 epoch.

        Args:
            dataloader: DataLoader từ create_alignment_dataloader().
            epoch: Số thứ tự epoch (0-indexed).
            progress_callback: Hàm callback báo tiến trình.

        Returns:
            TrainingMetrics cho epoch này.
        """
        self.system.projector.train()
        metrics = TrainingMetrics(epoch=epoch)
        start_time = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            # Forward + Loss
            total_loss, lm_loss_val, vad_loss_val = self._process_batch(batch)

            # Scale loss cho gradient accumulation
            scaled_loss = total_loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()

            # Update metrics
            metrics.total_loss += total_loss.item()
            metrics.lm_loss += lm_loss_val.item()
            metrics.vad_loss += vad_loss_val.item()
            metrics.num_steps += 1

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.system.projector.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = metrics.total_loss / metrics.num_steps
                lr = self.optimizer.param_groups[0]["lr"]
                log_msg = (
                    f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )
                logger.info(log_msg)
                if progress_callback:
                    progress_callback(log_msg)

        # Flush remaining gradients
        if (batch_idx + 1) % self.config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.system.projector.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        metrics.elapsed_seconds = time.time() - start_time
        logger.info(metrics.summary())

        return metrics

    def train(
        self,
        dataloader: DataLoader,
        progress_callback=None,
    ) -> list[TrainingMetrics]:
        """Chạy toàn bộ quá trình huấn luyện.

        Args:
            dataloader: DataLoader cho alignment data.
            progress_callback: Hàm callback báo tiến trình.

        Returns:
            List TrainingMetrics cho mỗi epoch.
        """
        total_steps = (
            len(dataloader)
            // self.config.gradient_accumulation_steps
            * self.config.num_epochs
        )
        self._setup_scheduler(total_steps)

        all_metrics = []

        for epoch in range(self.config.num_epochs):
            if progress_callback:
                progress_callback(f"=== Epoch {epoch + 1}/{self.config.num_epochs} ===")

            metrics = self.train_epoch(dataloader, epoch, progress_callback)
            all_metrics.append(metrics)

            # Save checkpoint
            if self.config.save_every_epoch:
                self.save_checkpoint(epoch)

            if progress_callback:
                progress_callback(metrics.summary())

        return all_metrics

    def save_checkpoint(self, epoch: int) -> Path:
        """Lưu checkpoint (chỉ Projector weights — nhẹ).

        Args:
            epoch: Epoch number.

        Returns:
            Đường dẫn file checkpoint.
        """
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_dir / f"projector_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "projector_state_dict": self.system.projector.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "config": {
                    "encoder_dim": self.system.projector.encoder_dim,
                    "llm_dim": self.system.projector.llm_dim,
                    "encoder_hf_id": self.system.encoder_info.hf_id,
                    "llm_hf_id": self.system.llm_info.hf_id,
                },
            },
            checkpoint_path,
        )

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Tải checkpoint để resume training.

        Args:
            checkpoint_path: Đường dẫn file checkpoint.

        Returns:
            Epoch number từ checkpoint.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.system.projector.load_state_dict(ckpt["projector_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]

        logger.info(
            f"Checkpoint loaded: epoch={ckpt['epoch']}, step={self.global_step}"
        )
        return ckpt["epoch"]
