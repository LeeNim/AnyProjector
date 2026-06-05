"""
inference.py - AnyProjector Inference Script

Load trained Q-Former projector checkpoint + Whisper encoder + LLM,
run inference on audio file(s) and output transcript.

Usage:
    python scripts/inference.py --audio path/to/audio.wav
    python scripts/inference.py --audio path/to/folder/  (batch)
    python scripts/inference.py --audio path/to/audio.wav --checkpoint path/to/ckpt.pt
"""

import sys
import os

# Fix Windows cp1252 encoding crash with Vietnamese text
if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import argparse
import time
from pathlib import Path

import torch
import numpy as np

from src.projector import AnyProjector


# -- Defaults --
DEFAULT_CHECKPOINT = "projectorTrained/projector_best.pt"
DEFAULT_ENCODER = "openai/whisper-medium"
DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROMPT = "Transcribe the following audio in Vietnamese:"
SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 30.0
MAX_NEW_TOKENS = 128


class AnyProjectorInference:
    """End-to-end inference: Audio -> Whisper -> Q-Former -> LLM -> Text."""

    def __init__(self, checkpoint_path: str,
                 encoder_id: str = DEFAULT_ENCODER,
                 llm_id: str = DEFAULT_LLM,
                 prompt: str = DEFAULT_PROMPT,
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt = prompt

        # --- Load checkpoint metadata ---
        print(f"Loading checkpoint: {checkpoint_path}")
        self.ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ckpt_config = self.ckpt.get("config", {})

        # Auto-detect model IDs from checkpoint if available
        self.encoder_id = ckpt_config.get("encoder_id", encoder_id)
        self.llm_id = ckpt_config.get("llm_id", llm_id)
        self.num_queries = ckpt_config.get("num_queries", 128)

        print(f"  Encoder: {self.encoder_id}")
        print(f"  LLM: {self.llm_id}")
        print(f"  Queries: {self.num_queries}")

        self._load_models()

    def _load_models(self):
        """Load all models and restore checkpoint weights."""
        from transformers import (
            WhisperModel, WhisperProcessor,
            AutoModelForCausalLM, AutoTokenizer, AutoConfig,
        )
        import gc

        # --- 1. Whisper Encoder ---
        print(f"Loading encoder: {self.encoder_id}")
        self.processor = WhisperProcessor.from_pretrained(self.encoder_id)
        whisper_full = WhisperModel.from_pretrained(self.encoder_id)
        self.encoder = whisper_full.encoder.eval().to(self.device)
        del whisper_full
        gc.collect()

        # Restore unfrozen encoder layers if saved
        if "encoder_state_dict" in self.ckpt:
            self.encoder.load_state_dict(self.ckpt["encoder_state_dict"], strict=False)
            print("  Loaded fine-tuned encoder layers from checkpoint")

        encoder_dim = self.encoder.config.d_model

        # --- 2. LLM ---
        print(f"Loading LLM: {self.llm_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_config = AutoConfig.from_pretrained(self.llm_id)
        llm_dim = llm_config.hidden_size if not hasattr(llm_config, 'text_config') else llm_config.text_config.hidden_size

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.llm.eval()

        # Restore LoRA weights if saved
        if "lora_state_dict" in self.ckpt:
            from peft import LoraConfig, get_peft_model
            ckpt_config = self.ckpt.get("config", {})
            lora_config = LoraConfig(
                r=ckpt_config.get("lora_rank", 8),
                lora_alpha=ckpt_config.get("lora_alpha", 16),
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.load_state_dict(self.ckpt["lora_state_dict"], strict=False)
            self.llm.eval()
            print("  Loaded LoRA adapter from checkpoint")

        self.embed_layer = self.llm.get_input_embeddings()
        self.llm_dtype = next(self.llm.parameters()).dtype

        # --- 3. Q-Former Projector ---
        print("Loading Q-Former projector...")
        ckpt_config = self.ckpt.get("config", {})

        # Auto-detect num_layers from state_dict if not in config
        proj_sd = self.ckpt["projector_state_dict"]
        if "qformer_layers" not in ckpt_config:
            layer_indices = {int(k.split(".")[1]) for k in proj_sd if k.startswith("layers.")}
            detected_layers = max(layer_indices) + 1 if layer_indices else 2
            print(f"  Auto-detected {detected_layers} Q-Former layers from state_dict")
        else:
            detected_layers = ckpt_config["qformer_layers"]

        self.projector = AnyProjector(
            encoder_dim=encoder_dim, llm_dim=llm_dim,
            num_queries=self.num_queries,
            qformer_dim=ckpt_config.get("qformer_dim", 768),
            num_layers=detected_layers,
            num_heads=ckpt_config.get("qformer_heads", 16),
        )
        self.projector.load_state_dict(proj_sd)
        self.projector.to(self.device).eval()
        print(f"  Projector: {self.projector.count_parameters():,} params")

        # VRAM report
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  VRAM: {allocated:.1f}GB")

        print("Ready!\n")

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe a single audio file.

        Returns:
            dict with keys: transcript, latency_ms, audio_duration_s
        """
        import librosa

        # Load audio
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        audio_duration = len(waveform) / SAMPLE_RATE

        t0 = time.time()

        with torch.no_grad():
            # --- Encoder mask ---
            encoder_seq_len = 1500
            samples_per_token = (MAX_AUDIO_SEC * SAMPLE_RATE) / encoder_seq_len
            real_tokens = min(encoder_seq_len, int(len(waveform) / samples_per_token))
            encoder_mask = torch.zeros(1, encoder_seq_len, dtype=torch.bool, device=self.device)
            encoder_mask[0, real_tokens:] = True

            # --- Whisper encoder ---
            audio_inputs = self.processor(
                waveform, sampling_rate=SAMPLE_RATE,
                return_tensors="pt", padding="max_length",
            )
            input_features = audio_inputs.input_features.to(self.device)
            encoder_output = self.encoder(input_features).last_hidden_state

            # --- Q-Former ---
            audio_embeds = self.projector(encoder_output, encoder_mask)  # (1, num_queries, llm_dim)

            # --- Prompt (chat template to match training) ---
            prompt_prefix = "<|im_start|>user\n" + self.prompt + "\n"
            prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"

            prefix_tokens = self.tokenizer(
                prompt_prefix, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(self.device)
            prefix_embeds = self.embed_layer(prefix_tokens)

            suffix_tokens = self.tokenizer(
                prompt_suffix, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(self.device)
            suffix_embeds = self.embed_layer(suffix_tokens)

            # --- Combine: [prefix | audio | suffix] ---
            input_embeds = torch.cat(
                [prefix_embeds, audio_embeds, suffix_embeds], dim=1
            ).to(self.llm_dtype)

            # --- Generate ---
            attn_mask = torch.ones(
                1, input_embeds.shape[1], dtype=torch.long, device=self.device
            )

            outputs = self.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # Greedy for evaluation
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        latency = (time.time() - t0) * 1000
        transcript = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "transcript": transcript,
            "latency_ms": round(latency, 1),
            "audio_duration_s": round(audio_duration, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="AnyProjector Inference")
    parser.add_argument("--audio", required=True, help="Audio file or folder path")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Projector checkpoint path")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="Whisper model ID")
    parser.add_argument("--llm", default=DEFAULT_LLM, help="LLM model ID")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text")
    args = parser.parse_args()

    model = AnyProjectorInference(
        checkpoint_path=args.checkpoint,
        encoder_id=args.encoder,
        llm_id=args.llm,
        prompt=args.prompt,
    )

    audio_path = Path(args.audio)
    if audio_path.is_file():
        files = [audio_path]
    elif audio_path.is_dir():
        files = sorted(audio_path.glob("*.wav")) + sorted(audio_path.glob("*.mp3")) + sorted(audio_path.glob("*.flac"))
    else:
        print(f"Error: {audio_path} not found")
        return

    print(f"Processing {len(files)} file(s)...\n")
    print(f"{'File':<40} {'Duration':>8} {'Latency':>10}  Transcript")
    print("-" * 100)

    for f in files:
        result = model.transcribe(str(f))
        print(f"{f.name:<40} {result['audio_duration_s']:>6.1f}s {result['latency_ms']:>8.0f}ms  {result['transcript']}")


if __name__ == "__main__":
    main()
