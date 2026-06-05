"""
train_phase2.py — Production Training Script cho Phase 2 Alignment.

Huấn luyện Projector để căn chỉnh audio embeddings vào không gian LLM.
Chỉ Projector có gradient. Encoder + LLM hoàn toàn đóng băng.

Pipeline:
    Audio → Whisper Encoder(frozen) → AnyProjector(trainable) → LLM(frozen) → Loss

Usage:
    python -m src.train_phase2 --dataset_dir dataset/phase2_alignment
    python -m src.train_phase2 --dataset_dir dataset/phase2_alignment --epochs 50 --lr 5e-4
    python -m src.train_phase2 --resume checkpoints/phase2/latest.pt
"""

import argparse
import gc
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset as hf_load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader, random_split

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.projector import AnyProjector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # Override transformers' logging config
)
logger = logging.getLogger("phase2")
logger.setLevel(logging.INFO)


# ============================================================================
# Config
# ============================================================================
@dataclass
class Phase2Config:
    """Phase 2 training configuration."""

    # --- Model IDs ---
    encoder_id: str = "openai/whisper-medium"
    llm_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # --- Dataset (HuggingFace) ---
    # List of (name, transcript_field, mode) tuples
    # mode: "all_train"   = merge all splits into train (small datasets)
    #       "auto_split"  = single split, auto 90/10 train/val
    #       "split"       = use original HF splits (train->train, test/val->val)
    datasets: tuple = (
        ("NhutP/VietSpeech", "transcription", "auto_split"),
    )
    max_samples_per_dataset: int = 200000  # 200K subset (0 = all)
    auto_val_ratio: float = 0.1  # For auto_split mode
    sample_rate: int = 16000
    max_audio_seconds: float = 30.0
    num_workers: int = 4  # DataLoader workers (0 = main thread only)
    preload_ram: bool = True  # Preload all audio to RAM at startup

    # --- Q-Former Projector ---
    num_queries: int = 64  # Output token count (input to LLM)
    qformer_dim: int = 768  # Hidden dim inside Q-Former
    qformer_layers: int = 4  # Number of transformer layers
    qformer_heads: int = 16  # Attention heads

    # --- LoRA (LLM fine-tuning) ---
    lora_enabled: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: tuple = ("q_proj", "v_proj")

    # --- Encoder Unfreezing ---
    unfreeze_encoder_layers: int = 0  # Fully frozen — focus on projector

    # --- Training ---
    num_epochs: int = 30
    batch_size: int = 32  # H100: 32-64, A100: 16-32, T4: 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05  # 5% steps warmup
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 2  # Effective batch = 32*2 = 64

    # --- Prompt ---
    prompt_text: str = "Transcribe the following audio in Vietnamese:"

    # --- Checkpoint ---
    save_dir: str = "checkpoints/phase2"
    save_every: int = 5  # Save every N epochs

    # --- Early Stopping ---
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 0.01

    # --- Resume ---
    resume_from: str | None = None

    # --- Local Cache (for Colab Drive workflow) ---
    local_cache_dir: str | None = None  # Path to pre-cached dataset on local disk

    # --- Streaming Mode (for 1M+ samples, 0 disk/RAM) ---
    streaming: bool = False  # Stream from HF, no download
    streaming_val_samples: int = 5000  # Cache this many val samples
    steps_per_epoch: int = 5000  # Steps per "epoch" in streaming mode
    streaming_shuffle_buffer: int = 10000  # Approximate shuffle buffer size


# ============================================================================
# Dataset
# ============================================================================
class Phase2Dataset(Dataset):
    """Dataset cho Phase 2 Alignment.

    Nhận list các entries đã chuẩn hóa: [{"audio": {...}, "transcript": str}, ...]
    Hỗ trợ preload toàn bộ audio vào RAM để loại bỏ decode overhead.
    """

    def __init__(self, entries: list, sample_rate: int = 16000,
                 max_audio_seconds: float = 30.0, preload_ram: bool = False):
        self.entries = entries
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_seconds * sample_rate)

        logger.info(f"Dataset: {len(entries)} samples")

        # Preload all audio into RAM as tensors
        self.cache = None
        if preload_ram:
            logger.info("Preloading audio to RAM...")
            self.cache = []
            for i, entry in enumerate(entries):
                waveform = self._process_audio(entry["audio"])
                self.cache.append(waveform)
                if (i + 1) % 1000 == 0:
                    logger.info(f"  preloaded {i+1}/{len(entries)}")
            ram_mb = sum(w.nbytes for w in self.cache) / 1024**2
            logger.info(f"  Done! {len(self.cache)} samples cached ({ram_mb:.0f} MB RAM)")

    def __len__(self):
        return len(self.entries)

    def _process_audio(self, audio_feature: dict) -> torch.Tensor:
        """Convert HF audio feature to processed waveform tensor."""
        waveform = torch.from_numpy(audio_feature["array"].astype(np.float32))
        sr = audio_feature["sampling_rate"]

        # Mono
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)

        # Resample if needed
        if sr != self.sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        # Truncate
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]

        return waveform

    def __getitem__(self, idx):
        if self.cache is not None:
            waveform = self.cache[idx]
        else:
            waveform = self._process_audio(self.entries[idx]["audio"])

        return {
            "waveform": waveform,
            "transcript": self.entries[idx]["transcript"],
        }


def collate_fn(batch):
    """Pad waveforms to same length in batch."""
    waveforms = [s["waveform"] for s in batch]
    transcripts = [s["transcript"] for s in batch]
    lengths = torch.tensor([w.shape[0] for w in waveforms])

    waveforms_padded = nn.utils.rnn.pad_sequence(
        waveforms, batch_first=True, padding_value=0.0
    )
    return {
        "waveforms": waveforms_padded,
        "lengths": lengths,
        "transcripts": transcripts,
    }


class StreamingBatchIterator:
    """Iterator that yields collated batches from an HF IterableDataset.

    Streams data from HF on-the-fly — 0 disk, minimal RAM.
    Each 'epoch' yields `steps_per_epoch` batches then stops.
    """

    def __init__(self, hf_iterable_dataset, transcript_field: str,
                 batch_size: int, steps_per_epoch: int,
                 sample_rate: int = 16000, max_audio_seconds: float = 30.0):
        self.ds = hf_iterable_dataset
        self.transcript_field = transcript_field
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_seconds * sample_rate)
        self._iter = None

    def _process_audio(self, audio_feature: dict) -> torch.Tensor:
        waveform = torch.from_numpy(audio_feature["array"].astype(np.float32))
        sr = audio_feature["sampling_rate"]
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)
        if sr != self.sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]
        return waveform

    def _get_iter(self):
        """Get or restart the stream iterator."""
        if self._iter is None:
            self._iter = iter(self.ds)
        return self._iter

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        """Yield collated batches for one epoch."""
        it = self._get_iter()
        for _ in range(self.steps_per_epoch):
            batch_items = []
            for _ in range(self.batch_size):
                try:
                    sample = next(it)
                except StopIteration:
                    # Stream exhausted → restart
                    self._iter = iter(self.ds)
                    it = self._iter
                    sample = next(it)
                waveform = self._process_audio(sample["audio"])
                batch_items.append({
                    "waveform": waveform,
                    "transcript": sample[self.transcript_field],
                })
            yield collate_fn(batch_items)


# ============================================================================
# Trainer
# ============================================================================
class Phase2Trainer:
    """Trainer cho Phase 2 Alignment.

    Quản lý toàn bộ lifecycle: load models, train loop, checkpoint.
    """

    def __init__(self, config: Phase2Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # Will be set during setup
        self.encoder = None
        self.projector = None
        self.llm = None
        self.tokenizer = None
        self.processor = None
        self.embed_layer = None
        self.llm_dtype = None
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.start_epoch = 0

    def setup_models(self):
        """Load tất cả models, apply LoRA / unfreeze as configured."""
        from transformers import (
            WhisperModel, WhisperProcessor,
            AutoModelForCausalLM, AutoTokenizer, AutoConfig,
        )

        # --- 1. Load Whisper Encoder ---
        logger.info(f"Loading encoder: {self.config.encoder_id}")
        self.processor = WhisperProcessor.from_pretrained(self.config.encoder_id)
        whisper_full = WhisperModel.from_pretrained(self.config.encoder_id)
        self.encoder = whisper_full.encoder.eval()
        del whisper_full
        gc.collect()

        # Freeze all encoder params first
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Optionally unfreeze last N encoder layers
        n_unfreeze = self.config.unfreeze_encoder_layers
        if n_unfreeze > 0:
            encoder_layers = self.encoder.layers
            total_layers = len(encoder_layers)
            for layer in encoder_layers[-n_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True
                layer.train()
            unfrozen_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            logger.info(f"  🔓 Unfroze last {n_unfreeze}/{total_layers} encoder layers ({unfrozen_params:,} params)")

        encoder_dim = self.encoder.config.d_model
        logger.info(f"  encoder_dim={encoder_dim}, params={sum(p.numel() for p in self.encoder.parameters()):,}")

        # --- 2. Auto-detect LLM dim, create Projector ---
        llm_config = AutoConfig.from_pretrained(self.config.llm_id)
        llm_dim = llm_config.hidden_size if not hasattr(llm_config, 'text_config') else llm_config.text_config.hidden_size
        logger.info(f"LLM dim: {llm_dim} (from {self.config.llm_id})")

        self.projector = AnyProjector(
            encoder_dim=encoder_dim, llm_dim=llm_dim,
            num_queries=self.config.num_queries,
            qformer_dim=self.config.qformer_dim,
            num_layers=self.config.qformer_layers,
            num_heads=self.config.qformer_heads,
        )
        self.projector.to(self.device).train()
        logger.info(f"Projector: {self.projector.count_parameters():,} trainable params")

        # --- 3. Load LLM (bf16, encoder to CPU to free VRAM) ---
        logger.info(f"Loading LLM: {self.config.llm_id}")
        self.encoder.cpu()  # Free VRAM
        torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()

        # --- 4. Apply LoRA if enabled ---
        if self.config.lora_enabled:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=list(self.config.lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            lora_params = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            logger.info(f"  🔗 LoRA applied: rank={self.config.lora_rank}, {lora_params:,} trainable LLM params")

        self.embed_layer = self.llm.get_input_embeddings()
        self.llm_dtype = next(self.llm.parameters()).dtype

        # Move encoder back to GPU
        self.encoder.to(self.device)
        llm_total = sum(p.numel() for p in self.llm.parameters())
        llm_train = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        logger.info(f"  LLM: {llm_total:,} total, {llm_train:,} trainable")

        # --- 5. VRAM report ---
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"  VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    def setup_optimizer(self, total_steps: int):
        """Create optimizer + warmup cosine scheduler."""
        # Collect all trainable parameters
        self._trainable_params = list(self.projector.parameters())
        trainable_params = self._trainable_params

        # Add LoRA params if enabled
        if self.config.lora_enabled:
            trainable_params += [p for p in self.llm.parameters() if p.requires_grad]

        # Add unfrozen encoder params
        if self.config.unfreeze_encoder_layers > 0:
            trainable_params += [p for p in self.encoder.parameters() if p.requires_grad]

        total_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Total trainable params: {total_trainable:,}")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        warmup_steps = int(total_steps * self.config.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        logger.info(f"Optimizer: AdamW lr={self.config.learning_rate}, warmup={warmup_steps}/{total_steps}")

    def process_batch(self, batch: dict) -> torch.Tensor:
        """Process 1 batch → return loss (scalar, has grad to projector)."""
        waveforms = batch["waveforms"]  # (B, max_samples)
        transcripts = batch["transcripts"]

        # --- Compute encoder padding mask ---
        # Whisper encoder: 30s audio → 1500 tokens (0.02s/token)
        # Real audio length → real encoder tokens, rest is padding
        encoder_seq_len = 1500  # Whisper fixed output
        samples_per_token = (self.config.max_audio_seconds * self.config.sample_rate) / encoder_seq_len
        encoder_mask = torch.zeros(
            len(waveforms), encoder_seq_len, dtype=torch.bool, device=self.device
        )
        for i, w in enumerate(waveforms):
            real_tokens = min(encoder_seq_len, int(w.shape[0] / samples_per_token))
            encoder_mask[i, real_tokens:] = True  # True = padding → ignore

        # --- Audio → Encoder (frozen, on GPU) ---
        with torch.no_grad():
            audio_inputs = self.processor(
                [w.numpy() for w in waveforms],
                sampling_rate=self.config.sample_rate,
                return_tensors="pt",
                padding="max_length",
            )
            input_features = audio_inputs.input_features.to(self.device)
            encoder_output = self.encoder(input_features).last_hidden_state

        # --- Encoder → Q-Former Projector (trainable) ---
        audio_embeds = self.projector(encoder_output, encoder_mask)  # (B, num_queries, llm_dim)

        # --- Prepare prompt using chat template (frozen) ---
        # Qwen2.5-Instruct expects: <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
        # We split into: prefix [before audio] + audio_embeds + suffix [after audio, before response]
        with torch.no_grad():
            prompt_prefix = "<|im_start|>user\n" + self.config.prompt_text + "\n"
            prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"

            prefix_tokens = self.tokenizer(
                prompt_prefix,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(self.device)
            prefix_embeds = self.embed_layer(prefix_tokens)
            prefix_embeds = prefix_embeds.expand(len(transcripts), -1, -1)

            suffix_tokens = self.tokenizer(
                prompt_suffix,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(self.device)
            suffix_embeds = self.embed_layer(suffix_tokens)
            suffix_embeds = suffix_embeds.expand(len(transcripts), -1, -1)

            # Append <|im_end|> so model learns to stop generating
            transcripts_with_eos = [t + "<|im_end|>" for t in transcripts]
            target_tokens = self.tokenizer(
                transcripts_with_eos,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                truncation=True,
                max_length=128,
            ).to(self.device)
            target_embeds = self.embed_layer(target_tokens.input_ids)  # (B, text_len, llm_dim)

        # --- Combine: [prefix | audio | suffix | target] → LLM ---
        full_input = torch.cat(
            [prefix_embeds, audio_embeds, suffix_embeds, target_embeds], dim=1
        ).to(self.llm_dtype)

        # --- Labels: [-100 for prefix+audio+suffix, target_ids for text] ---
        batch_size = len(transcripts)
        ignore_len = prefix_embeds.shape[1] + audio_embeds.shape[1] + suffix_embeds.shape[1]
        ignore_labels = torch.full(
            (batch_size, ignore_len), -100,
            dtype=torch.long, device=self.device,
        )
        labels = torch.cat([ignore_labels, target_tokens.input_ids], dim=1)

        # --- Attention mask ---
        audio_attn = torch.ones(
            (batch_size, ignore_len),
            dtype=torch.long, device=self.device,
        )
        attn_mask = torch.cat([audio_attn, target_tokens.attention_mask], dim=1)

        # --- Forward LLM ---
        outputs = self.llm(
            inputs_embeds=full_input,
            attention_mask=attn_mask,
            labels=labels,
        )

        return outputs.loss

    def log_gradient_diagnostics(self):
        """Log gradient norms per component to diagnose gradient flow."""
        diagnostics = {}

        # Projector gradient norms
        proj_grad_norm = 0.0
        proj_param_count = 0
        proj_zero_grads = 0
        for name, param in self.projector.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                proj_grad_norm += grad_norm ** 2
                proj_param_count += 1
                if grad_norm < 1e-8:
                    proj_zero_grads += 1
            elif param.requires_grad:
                proj_zero_grads += 1
                proj_param_count += 1
        proj_grad_norm = proj_grad_norm ** 0.5
        diagnostics["projector_grad_norm"] = proj_grad_norm
        diagnostics["projector_zero_grads"] = f"{proj_zero_grads}/{proj_param_count}"

        # Query tokens gradient
        query_grad = 0.0
        if self.projector.query_tokens.grad is not None:
            query_grad = self.projector.query_tokens.grad.data.norm(2).item()
        diagnostics["query_grad_norm"] = query_grad

        # Encoder gradient (if unfreezing)
        enc_grad_norm = 0.0
        enc_count = 0
        for name, param in self.encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                enc_grad_norm += param.grad.data.norm(2).item() ** 2
                enc_count += 1
        if enc_count > 0:
            diagnostics["encoder_grad_norm"] = enc_grad_norm ** 0.5

        return diagnostics

    def log_embedding_diagnostics(self, batch):
        """Check if projector outputs collapse (all same regardless of input)."""
        self.projector.eval()
        with torch.no_grad():
            waveforms = batch["waveforms"]
            if waveforms.shape[0] < 2:
                return {}

            # Process 2 samples
            embeds = []
            for i in range(min(2, waveforms.shape[0])):
                wf = waveforms[i]
                audio_inputs = self.processor(
                    wf.numpy(), sampling_rate=self.config.sample_rate,
                    return_tensors="pt", padding="max_length",
                )
                input_features = audio_inputs.input_features.to(self.device)
                enc_out = self.encoder(input_features).last_hidden_state
                proj_out = self.projector(enc_out)
                embeds.append(proj_out.flatten())

            # Cosine similarity between outputs of different audio
            cos_sim = torch.nn.functional.cosine_similarity(
                embeds[0].unsqueeze(0), embeds[1].unsqueeze(0)
            ).item()

            # Stats
            mean_norm = sum(e.norm().item() for e in embeds) / 2
            std_val = embeds[0].std().item()

        self.projector.train()
        return {
            "embed_cosine_sim": cos_sim,  # 1.0 = collapsed, <0.9 = differentiating
            "embed_mean_norm": mean_norm,
            "embed_std": std_val,
        }

    def _vram_info(self) -> str:
        """Get current VRAM usage string."""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"{alloc:.1f}/{total:.1f}GB"
        return "N/A"

    def train(self, preloaded_dataset=None):
        """Full training loop with verbose logging.

        Args:
            preloaded_dataset: Optional HF Dataset already loaded in memory.
                If provided, skips all downloading/caching and uses this directly.
        """
        config = self.config

        # --- Setup ---
        self.setup_models()

        # --- Dataset ---
        if config.streaming:
            # ===== STREAMING MODE: 0 disk, minimal RAM =====
            ds_name, transcript_field = config.datasets[0][0], config.datasets[0][1]
            logger.info(f"🌊 STREAMING MODE: {ds_name}")
            logger.info(f"  Steps/epoch: {config.steps_per_epoch}, val samples: {config.streaming_val_samples}")

            # Stream train data (shuffled with buffer)
            ds_stream = hf_load_dataset(ds_name, split='train', streaming=True, token=True)
            ds_stream = ds_stream.shuffle(buffer_size=config.streaming_shuffle_buffer, seed=42)

            train_loader = StreamingBatchIterator(
                ds_stream,
                transcript_field=transcript_field,
                batch_size=config.batch_size,
                steps_per_epoch=config.steps_per_epoch,
                sample_rate=config.sample_rate,
                max_audio_seconds=config.max_audio_seconds,
            )

            # Cache small val set (download only val_samples)
            logger.info(f"  Caching {config.streaming_val_samples} val samples...")
            ds_val_stream = hf_load_dataset(ds_name, split='train', streaming=True, token=True)
            val_entries = []
            # Skip past train data to get different val samples
            skip_count = config.steps_per_epoch * config.batch_size
            for i, sample in enumerate(ds_val_stream):
                if i < skip_count:
                    continue
                if len(val_entries) >= config.streaming_val_samples:
                    break
                val_entries.append({
                    "audio": sample["audio"],
                    "transcript": sample[transcript_field],
                })
                if (len(val_entries)) % 1000 == 0:
                    logger.info(f"    cached {len(val_entries)}/{config.streaming_val_samples}")

            val_dataset = Phase2Dataset(
                val_entries,
                sample_rate=config.sample_rate,
                max_audio_seconds=config.max_audio_seconds,
                preload_ram=True,  # Val set is small, always preload
            )
            nw = config.num_workers
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size,
                shuffle=False, collate_fn=collate_fn,
                num_workers=nw, pin_memory=True,
                persistent_workers=nw > 0,
            )

            logger.info(f"  Streaming train: {config.steps_per_epoch} batches/epoch")
            logger.info(f"  Cached val: {len(val_dataset)} samples")
            streaming_mode = True

        else:
            # ===== STANDARD MODE: download + cache =====
            from datasets import concatenate_datasets
            train_entries = []
            val_entries = []

            if preloaded_dataset is not None:
                # ===== PRELOADED MODE: dataset already in memory =====
                logger.info(f"📦 Using preloaded dataset: {len(preloaded_dataset)} samples")
                transcript_field = config.datasets[0][1]
                limit = config.max_samples_per_dataset
                total = len(preloaded_dataset)
                if limit > 0 and total > limit:
                    preloaded_dataset = preloaded_dataset.select(range(limit))
                    total = limit
                    logger.info(f"  Selected {total} samples (max_samples={limit})")

                # Batch extract — much faster than row-by-row access
                logger.info(f"  Extracting {total} samples...")
                t_extract = time.time()
                audios = preloaded_dataset["audio"]
                transcripts = preloaded_dataset[transcript_field]
                entries = [
                    {"audio": audios[i], "transcript": transcripts[i]}
                    for i in range(total)
                ]
                del audios, transcripts
                logger.info(f"  ✅ Extracted {len(entries)} samples in {time.time() - t_extract:.1f}s")

                # Auto-split
                import random
                rng = random.Random(42)
                val_count = max(1, int(len(entries) * config.auto_val_ratio))
                train_count = len(entries) - val_count
                indices = list(range(len(entries)))
                rng.shuffle(indices)
                for idx in indices[:train_count]:
                    train_entries.append(entries[idx])
                for idx in indices[train_count:]:
                    val_entries.append(entries[idx])
                logger.info(f"  -> {train_count} TRAIN + {val_count} VAL (from preloaded)")
                del preloaded_dataset

            # Check for pre-cached local dataset (Drive → local workflow)
            elif config.local_cache_dir and Path(config.local_cache_dir).exists():
                logger.info(f"Loading dataset from local cache: {config.local_cache_dir}")
                cached_ds = load_from_disk(config.local_cache_dir)
                logger.info(f"  Loaded {len(cached_ds)} samples from cache")

                # Extract entries from cached dataset
                transcript_field = config.datasets[0][1]  # Use first dataset's transcript field
                limit = config.max_samples_per_dataset
                total = len(cached_ds)
                if limit > 0 and total > limit:
                    cached_ds = cached_ds.select(range(limit))
                    total = limit
                    logger.info(f"  Selected {total} samples (max_samples={limit})")

                entries = []
                for i in range(total):
                    entries.append({
                        "audio": cached_ds[i]["audio"],
                        "transcript": cached_ds[i][transcript_field],
                    })

                # Auto-split
                import random
                rng = random.Random(42)
                val_count = max(1, int(len(entries) * config.auto_val_ratio))
                train_count = len(entries) - val_count
                indices = list(range(len(entries)))
                rng.shuffle(indices)
                for idx in indices[:train_count]:
                    train_entries.append(entries[idx])
                for idx in indices[train_count:]:
                    val_entries.append(entries[idx])
                logger.info(f"  -> {train_count} TRAIN + {val_count} VAL (auto_split from cache)")
                del cached_ds
            else:
                # Standard HF download path
                for ds_config in config.datasets:
                    ds_name, transcript_field, mode = ds_config[0], ds_config[1], ds_config[2]
                    logger.info(f"Loading: {ds_name} (field='{transcript_field}', mode='{mode}')")

                    # Load all available splits
                    available = {}
                    for split_name in ["train", "train_115", "validation", "dev", "test"]:
                        try:
                            # Try without trust_remote_code first (Parquet datasets)
                            ds = hf_load_dataset(ds_name, split=split_name, token=True)
                            available[split_name] = ds
                            logger.info(f"  split '{split_name}': {len(ds)} samples")
                        except ValueError:
                            # Fallback: some older datasets need trust_remote_code
                            try:
                                ds = hf_load_dataset(ds_name, split=split_name, trust_remote_code=True, token=True)
                                available[split_name] = ds
                                logger.info(f"  split '{split_name}': {len(ds)} samples (trust_remote_code)")
                            except Exception:
                                pass
                        except Exception:
                            pass

                    if not available:
                        logger.warning(f"  No splits found for {ds_name}, skipping.")
                        continue

                    def extract_entries(dataset):
                        """Extract unified entries from HF dataset."""
                        result = []
                        limit = config.max_samples_per_dataset
                        total = len(dataset)
                        if limit > 0 and total > limit:
                            dataset = dataset.select(range(limit))
                            total = limit
                        for i in range(total):
                            result.append({
                                "audio": dataset[i]["audio"],
                                "transcript": dataset[i][transcript_field],
                            })
                        return result

                    if mode == "all_train":
                        # Merge ALL splits into training (for small datasets)
                        merged = concatenate_datasets(list(available.values()))
                        entries = extract_entries(merged)
                        train_entries.extend(entries)
                        logger.info(f"  -> {len(entries)} samples -> TRAIN (all_train)")

                    elif mode == "auto_split":
                        # Single split, auto-split into train/val
                        merged = concatenate_datasets(list(available.values()))
                        entries = extract_entries(merged)
                        val_count = max(1, int(len(entries) * config.auto_val_ratio))
                        train_count = len(entries) - val_count
                        # Deterministic split
                        import random
                        rng = random.Random(42)
                        indices = list(range(len(entries)))
                        rng.shuffle(indices)
                        for idx in indices[:train_count]:
                            train_entries.append(entries[idx])
                        for idx in indices[train_count:]:
                            val_entries.append(entries[idx])
                        logger.info(f"  -> {train_count} TRAIN + {val_count} VAL (auto_split)")

                    elif mode == "split":
                        # Use original HF splits
                        train_splits = [s for s in ["train", "train_115"] if s in available]
                        val_splits = [s for s in ["validation", "dev", "test"] if s in available]
                        # Remove train splits from val
                        val_splits = [s for s in val_splits if s not in train_splits]

                        if train_splits:
                            merged_train = concatenate_datasets([available[s] for s in train_splits])
                            t_entries = extract_entries(merged_train)
                            train_entries.extend(t_entries)
                            logger.info(f"  -> {len(t_entries)} samples -> TRAIN ({', '.join(train_splits)})")

                        if val_splits:
                            merged_val = concatenate_datasets([available[s] for s in val_splits])
                            v_entries = extract_entries(merged_val)
                            val_entries.extend(v_entries)
                            logger.info(f"  -> {len(v_entries)} samples -> VAL ({', '.join(val_splits)})")

            logger.info(f"Dataset: {len(train_entries)} train + {len(val_entries)} val = {len(train_entries) + len(val_entries)} total")

            train_dataset = Phase2Dataset(
                train_entries,
                sample_rate=config.sample_rate,
                max_audio_seconds=config.max_audio_seconds,
                preload_ram=config.preload_ram,
            )
            val_dataset = Phase2Dataset(
                val_entries,
                sample_rate=config.sample_rate,
                max_audio_seconds=config.max_audio_seconds,
                preload_ram=config.preload_ram,
            )

            nw = config.num_workers
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size,
                shuffle=True, collate_fn=collate_fn,
                num_workers=nw, pin_memory=True,
                persistent_workers=nw > 0,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size,
                shuffle=False, collate_fn=collate_fn,
                num_workers=nw, pin_memory=True,
                persistent_workers=nw > 0,
            )

            logger.info(f"Dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}")
            logger.info(f"Batches/epoch: {len(train_loader)}")
            streaming_mode = False

        # --- Optimizer ---
        if streaming_mode:
            steps_per_epoch = math.ceil(config.steps_per_epoch / config.gradient_accumulation_steps)
        else:
            steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * config.num_epochs
        self.setup_optimizer(total_steps)

        # --- Resume ---
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

        # --- Training Loop ---
        logger.info("")
        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║         🚀 PHASE 2 ALIGNMENT TRAINING                    ║")
        logger.info("╠" + "═" * 58 + "╣")
        logger.info(f"║  Encoder:    {config.encoder_id:<43}║")
        logger.info(f"║  LLM:        {config.llm_id:<43}║")
        qf_info = f"{self.projector.count_parameters():>10,} params (Q-Former {config.num_queries}q)"
        logger.info(f"║  Projector:  {qf_info:<43}║")
        logger.info(f"║  Epochs:     {config.num_epochs:<43}║")
        logger.info(f"║  Batch:      {config.batch_size} × {config.gradient_accumulation_steps} accum = {config.batch_size * config.gradient_accumulation_steps} effective{' ' * 24}║")
        logger.info(f"║  LR:         {config.learning_rate:<43}║")
        logger.info(f"║  Steps:      {steps_per_epoch}/epoch, {total_steps} total{' ' * 26}║")
        logger.info(f"║  VRAM:       {self._vram_info():<43}║")
        logger.info("╚" + "═" * 58 + "╝")
        logger.info("")

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        history = []  # Track loss per epoch

        for epoch in range(self.start_epoch, config.num_epochs):
            epoch_num = epoch + 1

            # ============ TRAIN ============
            logger.info(f"━━━ Epoch {epoch_num}/{config.num_epochs} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            self.projector.train()
            train_loss_sum = 0.0
            train_steps = 0
            batch_losses = []
            self.optimizer.zero_grad()
            epoch_start = time.time()
            epoch_grad_diag = None

            for batch_idx, batch in enumerate(train_loader):
                loss = self.process_batch(batch)

                # Cast loss to fp32 for stable backward
                loss_val = loss.float()
                scaled_loss = loss_val / config.gradient_accumulation_steps
                scaled_loss.backward()

                batch_loss = loss_val.item()
                train_loss_sum += batch_loss
                train_steps += 1
                batch_losses.append(batch_loss)

                # Progress bar (single line, overwrite)
                total = len(train_loader)
                done = batch_idx + 1
                pct = done / total
                bar_len = 25
                filled = int(bar_len * pct)
                bar = "█" * filled + "░" * (bar_len - filled)
                avg_loss = train_loss_sum / train_steps
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed_s = time.time() - epoch_start
                print(
                    f"\r  {bar} {done}/{total} | "
                    f"loss={avg_loss:.4f} | lr={lr:.2e} | "
                    f"{elapsed_s:.0f}s | VRAM {self._vram_info()}",
                    end="", flush=True,
                )

                # Optimizer step
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    # Collect gradient diagnostics BEFORE clipping (first step only)
                    if epoch_grad_diag is None:
                        epoch_grad_diag = self.log_gradient_diagnostics()
                    torch.nn.utils.clip_grad_norm_(
                        self._trainable_params, config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

            # Flush remaining grads
            if train_steps % config.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_params, config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            train_avg = train_loss_sum / max(train_steps, 1)
            elapsed = time.time() - epoch_start
            print()  # Newline after progress bar

            # ============ VALIDATION ============
            logger.info(f"  📊 Validating...")
            self.projector.eval()
            val_loss_sum = 0.0
            val_steps = 0

            with torch.no_grad():
                for val_idx, batch in enumerate(val_loader):
                    loss = self.process_batch(batch)
                    val_loss = loss.float().item()
                    val_loss_sum += val_loss
                    val_steps += 1
                    logger.info(f"     val batch {val_idx+1}/{len(val_loader)} | loss={val_loss:.4f}")

            val_avg = val_loss_sum / max(val_steps, 1)

            # ============ EPOCH SUMMARY ============
            history.append({
                "epoch": epoch_num,
                "train": train_avg,
                "val": val_avg,
                "lr": self.optimizer.param_groups[0]['lr'],
                "overfit_ratio": val_avg / max(train_avg, 1e-8),
                "grad_norm": epoch_grad_diag.get('projector_grad_norm', 0) if epoch_grad_diag else 0,
                "query_grad_norm": epoch_grad_diag.get('query_grad_norm', 0) if epoch_grad_diag else 0,
                "elapsed_s": elapsed,
            })
            # Overfit ratio classification
            overfit_ratio = val_avg / max(train_avg, 1e-8)
            if overfit_ratio > 2.0:
                fit_status = "⚠️ OVERFIT"
            elif overfit_ratio > 1.5:
                fit_status = "😐 MILD"
            elif overfit_ratio > 1.2:
                fit_status = "👍 OK"
            else:
                fit_status = "✅ GOOD"

            # Early stopping + overfit-aware best saving
            val_improved = val_avg < best_val_loss - config.early_stopping_min_delta
            fit_healthy = overfit_ratio <= 1.5  # GOOD (≤1.2) or OK (1.2-1.5)

            if val_improved and fit_healthy:
                # Val improved AND fit is healthy → save as best
                best_val_loss = val_avg
                best_epoch = epoch_num
                patience_counter = 0
                improved = "★ BEST"
                self.save_checkpoint("best", val_avg)
            elif val_improved and not fit_healthy:
                # Val improved but overfit starting → don't save, count patience
                patience_counter += 1
                improved = f"↘ val ok but overfit {patience_counter}/{config.early_stopping_patience}"
                logger.info(f"  [!] Val improved but overfit ratio {overfit_ratio:.2f}x > 1.5 — skipping save")
            else:
                # Val didn't improve → count patience
                patience_counter += 1
                improved = f"wait {patience_counter}/{config.early_stopping_patience}"

            logger.info(f"")
            logger.info(f"  ┌─────────────────────────────────────────────┐")
            logger.info(f"  │ Epoch {epoch_num:>2}/{config.num_epochs} Summary{' ' * 27}│")
            logger.info(f"  ├─────────────────────────────────────────────┤")
            logger.info(f"  │ Train Loss:    {train_avg:>8.4f}                      │")
            logger.info(f"  │ Val Loss:      {val_avg:>8.4f}  {improved:<19}│")
            logger.info(f"  │ Overfit Ratio: {overfit_ratio:>8.2f}x {fit_status:<18}│")
            logger.info(f"  │ Best Val:      {best_val_loss:>8.4f}  (epoch {best_epoch:>2}){' ' * 11}│")
            logger.info(f"  │ LR:            {self.optimizer.param_groups[0]['lr']:>8.2e}                      │")
            logger.info(f"  │ Time:          {elapsed:>8.1f}s                     │")
            logger.info(f"  │ Global Step:   {self.global_step:>8}                      │")
            logger.info(f"  │ VRAM:          {self._vram_info():>12}                  │")
            logger.info(f"  └─────────────────────────────────────────────┘")

            # ============ DIAGNOSTICS ============
            logger.info(f"  -- Gradient Flow Diagnostics --")
            if epoch_grad_diag:
                logger.info(f"    Projector grad norm:  {epoch_grad_diag['projector_grad_norm']:.6f}")
                logger.info(f"    Query grad norm:      {epoch_grad_diag['query_grad_norm']:.6f}")
                logger.info(f"    Zero grads:           {epoch_grad_diag['projector_zero_grads']}")
                if "encoder_grad_norm" in epoch_grad_diag:
                    logger.info(f"    Encoder grad norm:    {epoch_grad_diag['encoder_grad_norm']:.6f}")
                # Interpret
                if epoch_grad_diag['projector_grad_norm'] < 1e-6:
                    logger.info(f"    [!] GRADIENT DEAD - projector receives no gradient!")
                elif epoch_grad_diag['projector_grad_norm'] < 1e-4:
                    logger.info(f"    [!] GRADIENT WEAK - projector learning very slowly")
                else:
                    logger.info(f"    [OK] Gradient flowing to projector")
            else:
                logger.info(f"    (no gradient data collected)")

            # Embedding collapse check (every 5 epochs to save time)
            if epoch_num % 5 == 1 or epoch_num <= 3:
                try:
                    sample_batch = next(iter(val_loader))
                    embed_diag = self.log_embedding_diagnostics(sample_batch)
                    if embed_diag:
                        logger.info(f"  -- Embedding Diagnostics --")
                        logger.info(f"    Cosine sim (2 samples): {embed_diag['embed_cosine_sim']:.4f}")
                        logger.info(f"    Mean norm:              {embed_diag['embed_mean_norm']:.4f}")
                        logger.info(f"    Std dev:                {embed_diag['embed_std']:.6f}")
                        if embed_diag['embed_cosine_sim'] > 0.99:
                            logger.info(f"    [!] COLLAPSE - all outputs identical regardless of input!")
                        elif embed_diag['embed_cosine_sim'] > 0.95:
                            logger.info(f"    [!] NEAR-COLLAPSE - outputs barely differentiate")
                        else:
                            logger.info(f"    [OK] Embeddings differentiate between audio samples")
                except Exception as e:
                    logger.info(f"    (embed diagnostics error: {e})")

            logger.info(f"")

            # Save periodic checkpoint
            if (epoch_num) % config.save_every == 0:
                self.save_checkpoint(epoch_num, val_avg)

            # Overfit catastrophic check — instant stop at 3.0x
            if overfit_ratio >= 3.0:
                logger.info(f"🚨 CATASTROPHIC OVERFIT! Ratio {overfit_ratio:.2f}x ≥ 3.0 — stopping immediately.")
                logger.info(f"   Best model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
                self.save_checkpoint(f"overfit_stop_e{epoch_num}", val_avg)
                break

            # Early stopping check (patience)
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"🛑 Early stopping! Val loss không cải thiện sau {config.early_stopping_patience} epochs.")
                logger.info(f"   Best model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
                self.save_checkpoint(f"early_stop_e{epoch_num}", val_avg)
                break

        # Save final
        self.save_checkpoint("final", val_avg)

        # ============ FINAL SUMMARY ============
        logger.info("")
        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║         🏁 TRAINING COMPLETE                             ║")
        logger.info("╠" + "═" * 58 + "╣")
        logger.info(f"║  Best Val Loss:  {best_val_loss:<39.4f}║")
        logger.info(f"║  Final Train:    {history[-1]['train']:<39.4f}║")
        logger.info(f"║  Total Steps:    {self.global_step:<39}║")
        logger.info(f"║  Checkpoints:    {config.save_dir:<39}║")
        logger.info("╠" + "═" * 58 + "╣")
        logger.info("║  Loss History (last 10 epochs):                          ║")
        for h in history[-10:]:
            bar_len = int(max(0, min(30, (h['train'] / max(history[0]['train'], 1)) * 30)))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            logger.info(f"║  E{h['epoch']:>2} T={h['train']:.3f} V={h['val']:.3f} {bar} ║")
        logger.info("╚" + "═" * 58 + "╝")

        # ============ EXPORT LOGS & PLOTS (before backup — backup disconnects runtime!) ============
        self._export_training_log(history)
        self._plot_training_curves(history)

        # ============ AUTO-BACKUP TO DRIVE (includes CSV + PNG, then disconnects) ============
        self._backup_to_drive()

    def _backup_to_drive(self):
        """Auto-backup checkpoints to Google Drive (must be mounted beforehand)."""
        import shutil

        src_dir = Path(self.config.save_dir)
        dst_dir = Path("/content/drive/MyDrive/AnyProjector") / self.config.save_dir

        if not src_dir.exists():
            logger.warning("No checkpoints to backup.")
            return

        # Check Drive mounted
        if not Path("/content/drive").exists():
            logger.info("Google Drive not mounted — skipping backup.")
            logger.info("  (Chạy local hoặc chưa mount Drive trong notebook)")
            return

        # Ensure backup folder exists
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files (checkpoints + training_log.csv + training_curves.png)
        count = 0
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_dir / f.name)
                logger.info(f"  📁 Backed up: {f.name} → Drive")
                count += 1

        logger.info(f"✅ {count} checkpoint(s) saved to: {dst_dir}")

        # Disconnect runtime to free GPU
        try:
            from google.colab import runtime
            logger.info("🔌 Disconnecting Colab runtime...")
            runtime.unassign()
        except Exception:
            pass  # Not on Colab or API not available

    def _export_training_log(self, history: list):
        """Export epoch-level training log as CSV."""
        import csv
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / "training_log.csv"

        fieldnames = ["epoch", "train", "val", "overfit_ratio", "lr", "grad_norm", "query_grad_norm", "elapsed_s"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for h in history:
                writer.writerow({k: h.get(k, "") for k in fieldnames})

        logger.info(f"📋 Training log saved: {csv_path}")

        # Also print formatted table
        logger.info("")
        logger.info("┌───────┬──────────┬──────────┬────────┬──────────┬──────────┬─────────┐")
        logger.info("│ Epoch │  Train   │   Val    │ Ratio  │ Grad Norm│    LR    │  Time   │")
        logger.info("├───────┼──────────┼──────────┼────────┼──────────┼──────────┼─────────┤")
        for h in history:
            logger.info(
                f"│  {h['epoch']:>3}  │ {h['train']:.4f}  │ {h['val']:.4f}  │ {h['overfit_ratio']:.2f}x  │ "
                f"{h['grad_norm']:.4f}  │ {h['lr']:.2e} │ {h['elapsed_s']:>5.0f}s  │"
            )
        logger.info("└───────┴──────────┴──────────┴────────┴──────────┴──────────┴─────────┘")

    def _plot_training_curves(self, history: list):
        """Plot training curves and save as PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping plot generation.")
            return

        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        epochs = [h["epoch"] for h in history]
        train_losses = [h["train"] for h in history]
        val_losses = [h["val"] for h in history]
        ratios = [h["overfit_ratio"] for h in history]
        grad_norms = [h["grad_norm"] for h in history]
        lrs = [h["lr"] for h in history]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("AnyProjector Phase 2 Training Curves", fontsize=14, fontweight="bold")

        # 1. Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss")
        ax.plot(epochs, val_losses, "r-s", markersize=3, label="Val Loss")
        best_idx = val_losses.index(min(val_losses))
        ax.axvline(x=epochs[best_idx], color="green", linestyle="--", alpha=0.5, label=f"Best (E{epochs[best_idx]})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train vs Val Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Overfit ratio
        ax = axes[0, 1]
        ax.plot(epochs, ratios, "m-^", markersize=3)
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect fit")
        ax.axhline(y=1.2, color="orange", linestyle="--", alpha=0.5, label="OK threshold")
        ax.axhline(y=1.5, color="red", linestyle="--", alpha=0.5, label="Overfit threshold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val/Train Ratio")
        ax.set_title("Overfit Ratio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Gradient norm
        ax = axes[1, 0]
        ax.plot(epochs, grad_norms, "g-d", markersize=3, label="Projector Grad Norm")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Flow")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Learning rate
        ax = axes[1, 1]
        ax.plot(epochs, lrs, "c-", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule")
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = save_dir / "training_curves.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"📊 Training curves saved: {plot_path}")

        # Also try to display inline (Colab/Jupyter)
        try:
            from IPython.display import display, Image
            display(Image(filename=str(plot_path)))
        except Exception:
            pass  # Not in notebook environment

    def save_checkpoint(self, tag, val_loss=None):
        """Save projector weights + training state + LoRA/encoder if applicable."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ckpt_data = {
            "projector_state_dict": self.projector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "epoch": tag if isinstance(tag, int) else -1,
            "val_loss": val_loss,
            "config": {
                "encoder_id": self.config.encoder_id,
                "llm_id": self.config.llm_id,
                "encoder_dim": self.projector.encoder_dim,
                "llm_dim": self.projector.llm_dim,
                "num_queries": self.projector.num_queries,
                "qformer_dim": self.config.qformer_dim,
                "qformer_layers": self.config.qformer_layers,
                "qformer_heads": self.config.qformer_heads,
                "lora_enabled": self.config.lora_enabled,
                "unfreeze_encoder_layers": self.config.unfreeze_encoder_layers,
            },
        }

        # Save LoRA adapter weights
        if self.config.lora_enabled:
            lora_state = {k: v for k, v in self.llm.state_dict().items() if "lora" in k}
            ckpt_data["lora_state_dict"] = lora_state

        # Save unfrozen encoder layer weights
        if self.config.unfreeze_encoder_layers > 0:
            encoder_state = {k: v for k, v in self.encoder.state_dict().items()
                             if any(k.startswith(f"layers.{len(self.encoder.layers) - i - 1}")
                                    for i in range(self.config.unfreeze_encoder_layers))}
            ckpt_data["encoder_state_dict"] = encoder_state

        path = save_dir / f"projector_{tag}.pt"
        torch.save(ckpt_data, path)
        logger.info(f"  Checkpoint saved: {path}")

        # Also save as 'latest'
        latest = save_dir / "latest.pt"
        torch.save(torch.load(path, weights_only=False), latest)

        # Immediately copy to Drive (prevent loss on disconnect)
        self._copy_to_drive(path)
        if tag == "best":
            self._copy_to_drive(latest)

    def _copy_to_drive(self, src_path: Path):
        """Copy a single file to Google Drive immediately."""
        import shutil
        drive_dir = Path("/content/drive/MyDrive/AnyProjector") / self.config.save_dir
        if not Path("/content/drive").exists():
            return  # Not on Colab or Drive not mounted
        try:
            drive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, drive_dir / src_path.name)
            logger.info(f"  📁 → Drive: {src_path.name}")
        except Exception as e:
            logger.warning(f"  ⚠️ Drive copy failed: {e}")

    def load_checkpoint(self, path: str):
        """Resume from checkpoint."""
        logger.info(f"Resuming from: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.projector.load_state_dict(ckpt["projector_state_dict"])
        if self.optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.start_epoch = ckpt.get("epoch", 0)
        if isinstance(self.start_epoch, int) and self.start_epoch > 0:
            logger.info(f"  Resumed at epoch {self.start_epoch}, step {self.global_step}")


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Alignment Training")
    parser.add_argument("--encoder_id", default="openai/whisper-medium")
    parser.add_argument("--llm_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--save_dir", default="checkpoints/phase2")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    parser.add_argument("--prompt", default="Transcribe the following audio in Vietnamese:")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples per dataset (0=all)")
    return parser.parse_args()


def main():
    args = parse_args()

    config = Phase2Config(
        encoder_id=args.encoder_id,
        llm_id=args.llm_id,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        save_dir=args.save_dir,
        resume_from=args.resume,
        prompt_text=args.prompt,
        early_stopping_patience=args.patience,
        max_samples_per_dataset=args.max_samples,
    )

    trainer = Phase2Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
