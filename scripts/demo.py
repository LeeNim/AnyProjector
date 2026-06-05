"""
demo.py - AnyProjector Interactive Demo

Features:
  - A/B comparison: Projector ON vs OFF (same faster-whisper encoder)
  - Tool calling: LLM outputs JSON to trigger actions
  - Streaming mic: Real-time speech detection with Silero VAD
  - Custom prompts: Edit system/user prompts
  - Timing breakdown: Encode / Project / Generate

Usage:
    python scripts/demo.py
    python scripts/demo.py --checkpoint path/to/projector_best.pt
    python scripts/demo.py --whisper-size large-v3
"""

import sys
import os

# Fix Windows encoding
if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

import argparse
import time
import json
import tempfile
import logging
import asyncio
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

# Suppress Windows ProactorEventLoop connection-reset noise
if sys.platform == "win32":
    _orig_er = asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost
    def _quiet_connection_lost(self, exc=None):
        try:
            _orig_er(self, exc)
        except (ConnectionResetError, OSError):
            pass
    asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = _quiet_connection_lost

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.projector import AnyProjector
from scripts.demo_tools import ToolRegistry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Defaults ──────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = "projectorTrained/final_011.pt"
DEFAULT_LORA = "lora/best011/best"
DEFAULT_ENCODER_CKPT = ""  # v0.11 LoRA: encoder frozen (lr_enc=0), no encoder checkpoint
DEFAULT_WHISPER = "medium"
DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 30.0

PRESET_PROMPTS = {
    "Transcribe (Vietnamese)": "Transcribe the following audio in Vietnamese:",
    "Transcribe (English)": "Transcribe the following audio in English:",
    "Translate to English": "Listen to this Vietnamese audio and translate it to English:",
    "Summarize": "Listen to the audio and provide a brief summary of its content:",
    "Q&A": "Listen to the audio and answer: What is the main topic being discussed?",
}


def _normalize_audio(audio) -> np.ndarray:
    """Normalize audio input to float32 mono 16kHz numpy array.

    Accepts:
        - (sample_rate, numpy_array) tuple from Gradio
        - filepath string
        - numpy array (assumed 16kHz float32)
    """
    if isinstance(audio, tuple):
        sr, data = audio
        # int16/int32 → float32
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        elif data.dtype != np.float32:
            data = data.astype(np.float32)
        # Stereo → mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Resample if not 16kHz
        if sr != SAMPLE_RATE:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        return data
    elif isinstance(audio, np.ndarray):
        return audio
    else:
        # Filepath — use soundfile (10x faster than librosa for WAV)
        import soundfile as sf
        data, sr = sf.read(str(audio), dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        return data


# ══════════════════════════════════════════════════════════════════════
#  Demo Engine — manages all models
# ══════════════════════════════════════════════════════════════════════

class DemoEngine:
    """Manages faster-whisper + Q-Former projector + LLM for inference."""

    def __init__(self, checkpoint_path: str, whisper_size: str = DEFAULT_WHISPER,
                 llm_id: str = DEFAULT_LLM, encoder_backend: str = "faster-whisper",
                 lora_path: str = None, encoder_ckpt: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder_backend = encoder_backend
        self.whisper_size = whisper_size
        self.llm_id = llm_id
        self.checkpoint_path = checkpoint_path
        self.lora_path = lora_path
        self.encoder_ckpt = encoder_ckpt

        self.fw_model = None       # faster-whisper model
        self.hf_encoder = None     # HF whisper encoder (fallback)
        self.hf_processor = None
        self.projector = None
        self.llm = None
        self.tokenizer = None
        self.embed_layer = None
        self.llm_dtype = None
        self.vad_model = None
        self.tool_registry = ToolRegistry()

        self._load_all()

    def _load_all(self):
        """Load all models."""
        t0 = time.time()
        self._load_whisper()
        self._load_projector()
        self._load_llm()
        self._load_vad()
        elapsed = time.time() - t0
        logger.info(f"All models loaded in {elapsed:.1f}s")
        self._print_vram()

    def _load_whisper(self):
        """Load whisper models.

        Always loads:
          - HF Whisper encoder (for projector path — must match training)
          - FasterWhisper (for cascade transcription — fast)
        """
        # 1. HF Whisper encoder — projector was trained on these outputs
        from transformers import WhisperProcessor
        model_id = f"openai/whisper-{self.whisper_size}"
        logger.info(f"Loading HF Whisper encoder ({model_id}) for projector...")
        self.hf_processor = WhisperProcessor.from_pretrained(model_id)
        whisper_full = __import__("transformers").WhisperModel.from_pretrained(model_id)
        self.hf_encoder = whisper_full.encoder.to(self.device).eval()
        del whisper_full

        # Load fine-tuned encoder weights (Phase 3)
        if self.encoder_ckpt and Path(self.encoder_ckpt).exists():
            logger.info(f"  Loading fine-tuned encoder: {self.encoder_ckpt}")
            enc_sd = torch.load(self.encoder_ckpt, map_location=self.device, weights_only=True)
            self.hf_encoder.load_state_dict(enc_sd)
            del enc_sd
            logger.info("  ✅ Fine-tuned encoder loaded")
        elif self.encoder_ckpt:
            logger.warning(f"  Encoder checkpoint not found: {self.encoder_ckpt}")

        logger.info("  HF Whisper encoder loaded")
        self._print_vram()

        # 2. FasterWhisper — for cascade mode transcription (fast)
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper ({self.whisper_size}) for cascade...")
            self.fw_model = WhisperModel(
                self.whisper_size, device="cuda" if self.device == "cuda" else "cpu",
                compute_type="float16" if self.device == "cuda" else "float32",
            )
            logger.info("  faster-whisper loaded")
        except Exception as e:
            logger.warning(f"  faster-whisper failed: {e} — cascade will use HF")
            self.fw_model = None

    def _load_projector(self):
        """Load Q-Former projector from checkpoint."""
        if not Path(self.checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {self.checkpoint_path}")
            return

        logger.info(f"Loading projector: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})

        # Auto-detect layers from state_dict
        proj_sd = ckpt["projector_state_dict"]
        layer_indices = {int(k.split(".")[1]) for k in proj_sd if k.startswith("layers.")}
        num_layers = max(layer_indices) + 1 if layer_indices else 4

        encoder_dim = config.get("encoder_dim", 1024)
        llm_dim = 1536  # Qwen2.5-1.5B

        # Try to get llm_dim from checkpoint
        # Output projection weight shape tells us: (llm_dim, qformer_dim)
        if "output_proj.weight" in proj_sd:
            llm_dim = proj_sd["output_proj.weight"].shape[0]

        self.projector = AnyProjector(
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
            num_queries=config.get("num_queries", 64),
            qformer_dim=config.get("qformer_dim", 768),
            num_layers=num_layers,
            num_heads=config.get("qformer_heads", 16),
            dropout=config.get("dropout", 0.0),
        )
        self.projector.load_state_dict(proj_sd)
        self.projector.to(self.device).eval()
        logger.info(f"  Projector: {self.projector.count_parameters():,} params, {num_layers} layers")

    def _load_llm(self):
        """Load LLM with int4 quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading LLM: {self.llm_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Try int4 quantization first (saves VRAM)
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_id, quantization_config=bnb_config,
                device_map="auto", torch_dtype=torch.float16,
            )
            logger.info("  LLM loaded (int4 quantization)")
        except Exception as e:
            logger.warning(f"  int4 failed ({e}), falling back to float16")
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_id, torch_dtype=torch.float16, device_map="auto",
            )
            logger.info("  LLM loaded (float16)")

        # Load LoRA adapter if provided
        if self.lora_path and Path(self.lora_path).exists():
            from peft import PeftModel
            logger.info(f"  Loading LoRA adapter: {self.lora_path}")
            self.llm = PeftModel.from_pretrained(self.llm, self.lora_path)
            logger.info("  LoRA adapter applied")
        elif self.lora_path:
            logger.warning(f"  LoRA path not found: {self.lora_path}")

        self.llm.eval()
        self.embed_layer = self.llm.get_input_embeddings()
        self.llm_dtype = next(self.llm.parameters()).dtype

    def _load_vad(self):
        """Load Silero VAD."""
        try:
            self.vad_model, self.vad_utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True,
            )
            self.get_speech_timestamps = self.vad_utils[0]
            logger.info("  Silero VAD loaded")
        except Exception as e:
            logger.warning(f"  VAD load failed: {e}")
            self.vad_model = None

    def _print_vram(self):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"  VRAM: {alloc:.1f}/{total:.1f} GB")

    # ── Encoder methods ───────────────────────────────────────────────

    def _encode_audio(self, waveform: np.ndarray) -> tuple[torch.Tensor, float]:
        """Encode audio → encoder hidden states.

        Always uses HF Whisper encoder (matches projector training).

        Returns:
            (encoder_output, encode_time_ms)
        """
        t0 = time.time()

        inputs = self.hf_processor(
            waveform, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding="max_length",
        )
        input_features = inputs.input_features.to(self.device)
        with torch.no_grad():
            encoder_out = self.hf_encoder(input_features).last_hidden_state

        encode_ms = (time.time() - t0) * 1000
        return encoder_out, encode_ms

    def _make_encoder_mask(self, waveform: np.ndarray, enc_seq_len: int) -> torch.Tensor:
        """Create padding mask for encoder output."""
        samples_per_token = (MAX_AUDIO_SEC * SAMPLE_RATE) / enc_seq_len
        real_tokens = min(enc_seq_len, int(len(waveform) / samples_per_token))
        mask = torch.zeros(1, enc_seq_len, dtype=torch.bool, device=self.device)
        mask[0, real_tokens:] = True
        return mask

    # ── Transcription methods ─────────────────────────────────────────

    def transcribe_standalone(self, audio, prompt: str = None,
                               temperature: float = 0.1, max_tokens: int = 256) -> dict:
        """Cascade pipeline: FasterWhisper → text → LLM.

        Transcribes audio with FasterWhisper, then feeds the transcript
        as text input to the same LLM for a fair A/B comparison with
        the projector pipeline (audio embeddings → LLM).

        Args:
            audio: filepath, numpy array, or (sample_rate, data) tuple.
            prompt: Same prompt used for projector mode.
            temperature: LLM sampling temperature.
            max_tokens: Max tokens to generate.
        """
        breakdown = {}

        # 1. Whisper transcription
        t0 = time.time()
        if self.fw_model is not None:
            audio_input = _normalize_audio(audio) if not isinstance(audio, str) else audio
            segments, info = self.fw_model.transcribe(
                audio_input, language="vi", beam_size=1, vad_filter=True,
            )
            whisper_text = " ".join(s.text for s in segments).strip()
        else:
            whisper_text = "[FasterWhisper not loaded]"
        breakdown["whisper_ms"] = round((time.time() - t0) * 1000, 1)

        # 2. Feed transcript into LLM (same model as projector path)
        t0 = time.time()
        prompt = prompt or "Transcribe the following audio in Vietnamese:"

        # Rewrite prompt for cascade: LLM receives text, not audio,
        # so "listen to the audio" would confuse it. Wrap transcript clearly.
        llm_input = (
            f"<|im_start|>user\n"
            f"The user said the following (transcribed from audio):\n"
            f"\"{whisper_text}\"\n\n"
            f"Instruction: {prompt}\n"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )

        input_ids = self.tokenizer(
            llm_input, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self.device)

        with torch.no_grad():
            gen_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
            else:
                gen_kwargs["do_sample"] = False

            outputs = self.llm.generate(**gen_kwargs)

        # Decode only new tokens (skip the input)
        new_tokens = outputs[0][input_ids.shape[1]:]
        llm_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        breakdown["llm_ms"] = round((time.time() - t0) * 1000, 1)

        breakdown["total_ms"] = round(sum(breakdown.values()), 1)
        return {
            "text": llm_text,
            "whisper_text": whisper_text,
            "total_ms": breakdown["total_ms"],
            "breakdown": breakdown,
        }

    def transcribe_projector(self, audio, prompt: str = None,
                              temperature: float = 0.1, max_tokens: int = 256) -> dict:
        """FasterWhisper encoder → Q-Former → LLM.

        Args:
            audio: filepath, numpy array, or (sample_rate, data) tuple.
        """
        if self.projector is None:
            return {"text": "[Projector not loaded]", "total_ms": 0, "breakdown": {}}

        waveform = _normalize_audio(audio)
        audio_duration = len(waveform) / SAMPLE_RATE
        breakdown = {}

        with torch.no_grad():
            # 1. Encode
            encoder_out, encode_ms = self._encode_audio(waveform)
            breakdown["encode_ms"] = round(encode_ms, 1)

            # 2. Project
            t0 = time.time()
            enc_mask = self._make_encoder_mask(waveform, encoder_out.shape[1])
            audio_embeds = self.projector(encoder_out.float(), enc_mask)
            breakdown["project_ms"] = round((time.time() - t0) * 1000, 1)

            # 3. Build prompt
            t0 = time.time()
            prompt = prompt or "Transcribe the following audio in Vietnamese:"
            prefix = "<|im_start|>user\n" + prompt + "\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n"

            prefix_ids = self.tokenizer(
                prefix, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(self.device)
            suffix_ids = self.tokenizer(
                suffix, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(self.device)

            prefix_embeds = self.embed_layer(prefix_ids)
            suffix_embeds = self.embed_layer(suffix_ids)

            input_embeds = torch.cat(
                [prefix_embeds, audio_embeds, suffix_embeds], dim=1,
            ).to(self.llm_dtype)
            attn_mask = torch.ones(
                1, input_embeds.shape[1], dtype=torch.long, device=self.device,
            )

            # 4. Generate
            gen_kwargs = dict(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
            else:
                gen_kwargs["do_sample"] = False

            outputs = self.llm.generate(**gen_kwargs)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            breakdown["generate_ms"] = round((time.time() - t0) * 1000, 1)

        breakdown["total_ms"] = round(sum(breakdown.values()), 1)
        return {
            "text": text.strip(),
            "total_ms": breakdown["total_ms"],
            "audio_duration": round(audio_duration, 2),
            "breakdown": breakdown,
        }

    # ── Tool calling ──────────────────────────────────────────────────

    def transcribe_with_tools(self, audio_path: str, enabled_tools: list[str],
                               user_prompt: str = "", temperature: float = 0.3,
                               max_tokens: int = 256) -> dict:
        """Audio → LLM with tool definitions → detect & execute tools."""
        tool_prompt = self.tool_registry.get_tool_prompt(enabled_tools)
        base_prompt = user_prompt or "Listen to the audio and respond. Use tools if appropriate."
        full_prompt = f"{tool_prompt}\n\n{base_prompt}"

        result = self.transcribe_projector(
            audio_path, prompt=full_prompt,
            temperature=temperature, max_tokens=max_tokens,
        )

        # Check for tool call in output
        tool_call, tool_result = self.tool_registry.detect_and_execute(result["text"])
        result["tool_call"] = tool_call
        result["tool_result"] = tool_result

        return result

    # ── Streaming transcription ────────────────────────────────────────

    def transcribe_stream_chunk(self, audio_chunk: tuple, state: dict) -> tuple[dict, str]:
        """Process a streaming audio chunk from Gradio mic.

        Uses Silero VAD to detect speech boundaries. Buffers speech frames,
        transcribes when silence is detected after speech.

        Args:
            audio_chunk: (sample_rate, numpy_array) from Gradio streaming.
            state: dict with 'buffer', 'transcripts', 'is_speaking' keys.

        Returns:
            (updated_state, new_transcript_line_or_empty)
        """
        if audio_chunk is None:
            return state, ""

        sr, data = audio_chunk

        # Convert to float32 mono
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample if needed
        if sr != SAMPLE_RATE:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

        # VAD check
        speech_detected = False
        if self.vad_model is not None and len(data) >= 512:
            wav_tensor = torch.from_numpy(data).float()
            try:
                speech_prob = self.vad_model(wav_tensor, SAMPLE_RATE).item()
                speech_detected = speech_prob > 0.5
            except Exception:
                speech_detected = True  # Assume speech if VAD fails
        else:
            speech_detected = True  # No VAD → always process

        new_line = ""

        if speech_detected:
            state["buffer"].append(data)
            state["is_speaking"] = True
        elif state["is_speaking"] and len(state["buffer"]) > 0:
            # Silence after speech → transcribe buffer
            full_audio = np.concatenate(state["buffer"])
            state["buffer"] = []
            state["is_speaking"] = False

            # Only transcribe if long enough (> 0.3s)
            if len(full_audio) > SAMPLE_RATE * 0.3:
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, full_audio, SAMPLE_RATE)
                    temp_path = f.name

                try:
                    if state.get("use_projector", False):
                        r = self.transcribe_projector(temp_path)
                    else:
                        r = self.transcribe_standalone(temp_path)

                    timestamp = time.strftime("%H:%M:%S")
                    duration = len(full_audio) / SAMPLE_RATE
                    new_line = f"[{timestamp}] ({duration:.1f}s, {r['total_ms']:.0f}ms) {r['text']}"
                    state["transcripts"].append(new_line)
                finally:
                    os.unlink(temp_path)

        return state, new_line


# ══════════════════════════════════════════════════════════════════════
#  Gradio UI
# ══════════════════════════════════════════════════════════════════════

def format_timing(breakdown: dict) -> str:
    """Format timing breakdown as a nice string."""
    parts = []
    for key in ["encode_ms", "project_ms", "generate_ms", "whisper_ms"]:
        if key in breakdown:
            label = key.replace("_ms", "").capitalize()
            parts.append(f"{label}: {breakdown[key]:.0f}ms")
    total = breakdown.get("total_ms", sum(breakdown.values()))
    return " | ".join(parts) + f" | **Total: {total:.0f}ms**"


def build_ui(engine: DemoEngine):
    """Build Gradio interface."""
    import gradio as gr

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(
        title="AnyProjector Demo",
        theme=theme,
        css="""
        .timing-bar { background: linear-gradient(90deg, #6366f1, #8b5cf6); border-radius: 8px; padding: 8px 14px; color: white; font-family: monospace; }
        .result-box { border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; background: #f8fafc; }
        .tool-json { background: #1e1e2e; color: #cdd6f4; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 13px; }
        footer { display: none !important; }
        """,
    ) as app:

        # ── Header ──
        gr.Markdown("""
        # 🎙️ AnyProjector Demo
        ### Whisper → Q-Former Projector → LLM Pipeline
        Compare **Projector ON** (audio understanding via LLM) vs **Projector OFF** (standalone Whisper).
        """)

        # ══════════════════════════════════════════════════════════════
        #  Tab 1: A/B Transcription
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("🎤 Transcription"):
            with gr.Row():
                # Left column — Controls
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"], type="filepath",
                        format="wav",
                        label="Audio Input",
                    )
                    with gr.Row():
                        use_projector = gr.Checkbox(label="Projector ON", value=True)
                        use_standalone = gr.Checkbox(label="Projector OFF (Whisper)", value=True)

                    encoder_dropdown = gr.Dropdown(
                        choices=["faster-whisper", "HuggingFace Whisper"],
                        value="faster-whisper", label="Encoder Backend",
                        info="Switch if projector quality is poor with faster-whisper",
                    )
                    prompt_dropdown = gr.Dropdown(
                        choices=list(PRESET_PROMPTS.keys()),
                        value="Transcribe (Vietnamese)",
                        label="Prompt Preset",
                    )
                    prompt_text = gr.Textbox(
                        value=PRESET_PROMPTS["Transcribe (Vietnamese)"],
                        label="Custom Prompt (editable)", lines=2,
                    )
                    with gr.Row():
                        temperature = gr.Slider(0, 1, value=0.1, step=0.05, label="Temperature")
                        max_tokens = gr.Slider(32, 512, value=256, step=32, label="Max Tokens")

                    run_btn = gr.Button("🚀 Run", variant="primary", size="lg")

                # Right column — Results
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("#### 🟢 Projector ON (Whisper Encoder → Q-Former → LLM)")
                        proj_output = gr.Textbox(label="Result", lines=3, interactive=False)
                        proj_timing = gr.Markdown("*Waiting...*")

                    with gr.Group():
                        gr.Markdown("#### 🔵 Projector OFF (FasterWhisper Standalone)")
                        standalone_output = gr.Textbox(label="Result", lines=3, interactive=False)
                        standalone_timing = gr.Markdown("*Waiting...*")

                    speed_comparison = gr.Markdown("*Run transcription to see comparison*")

            # ── Preset prompt update ──
            def update_prompt(preset_name):
                return PRESET_PROMPTS.get(preset_name, "")
            prompt_dropdown.change(update_prompt, prompt_dropdown, prompt_text)

            # ── Encoder backend switch ──
            def switch_encoder(backend):
                new_backend = "faster-whisper" if backend == "faster-whisper" else "hf-whisper"
                if new_backend != engine.encoder_backend:
                    engine.encoder_backend = new_backend
                    if new_backend == "hf-whisper" and engine.hf_encoder is None:
                        engine._load_whisper()  # Load HF encoder on demand
                return f"Encoder: {backend}"
            encoder_dropdown.change(switch_encoder, encoder_dropdown, gr.Markdown(visible=False))

            # ── Run transcription ──
            def run_transcription(audio, do_proj, do_standalone, prompt, temp, max_tok):
                if audio is None:
                    return "", "*No audio*", "", "*No audio*", "*Upload or record audio first*"

                proj_text, proj_time_md = "", "*Skipped*"
                stan_text, stan_time_md = "", "*Skipped*"
                comparison = ""

                proj_ms, stan_ms = 0, 0

                if do_proj:
                    r = engine.transcribe_projector(
                        audio, prompt=prompt,
                        temperature=temp, max_tokens=int(max_tok),
                    )
                    proj_text = r["text"]
                    proj_ms = r["total_ms"]
                    bd = r["breakdown"]
                    dur = r.get("audio_duration", 0)
                    proj_time_md = (
                        f"⏱️ **{proj_ms:.0f}ms** total "
                        f"(Encode: {bd.get('encode_ms', 0):.0f}ms | "
                        f"Project: {bd.get('project_ms', 0):.0f}ms | "
                        f"Generate: {bd.get('generate_ms', 0):.0f}ms) "
                        f"— Audio: {dur:.1f}s"
                    )

                if do_standalone:
                    r = engine.transcribe_standalone(audio)
                    stan_text = r["text"]
                    stan_ms = r["total_ms"]
                    stan_time_md = f"⏱️ **{stan_ms:.0f}ms** total"

                if do_proj and do_standalone:
                    if stan_ms > 0:
                        ratio = proj_ms / stan_ms
                        faster = "Whisper" if ratio > 1 else "Projector"
                        comparison = (
                            f"### ⚡ Speed Comparison\n"
                            f"| Mode | Latency | Ratio |\n"
                            f"|------|---------|-------|\n"
                            f"| Projector ON | {proj_ms:.0f}ms | {ratio:.1f}x |\n"
                            f"| Projector OFF | {stan_ms:.0f}ms | 1.0x |\n"
                            f"\n**{faster}** is faster."
                        )

                return proj_text, proj_time_md, stan_text, stan_time_md, comparison

            run_btn.click(
                run_transcription,
                inputs=[audio_input, use_projector, use_standalone, prompt_text, temperature, max_tokens],
                outputs=[proj_output, proj_timing, standalone_output, standalone_timing, speed_comparison],
            )

        # ══════════════════════════════════════════════════════════════
        #  Tab 2: Tool Calling
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("🛠️ Tool Calling"):
            gr.Markdown("""
            The LLM can output JSON to call tools. Speak naturally — if the content 
            involves math, time, or translation, the LLM may invoke a tool.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    tool_audio = gr.Audio(
                        sources=["microphone", "upload"], type="filepath",
                        format="wav",
                        label="Audio Input",
                    )
                    tool_prompt = gr.Textbox(
                        value="Listen to the audio and respond. If the user asks for a calculation, use the calculator tool. If they ask for the time, use get_time.",
                        label="System Prompt", lines=3,
                    )
                    tool_checks = gr.CheckboxGroup(
                        choices=engine.tool_registry.list_tools(),
                        value=engine.tool_registry.list_tools(),
                        label="Enabled Tools",
                    )
                    tool_temp = gr.Slider(0, 1, value=0.3, step=0.05, label="Temperature")
                    tool_btn = gr.Button("🔧 Run with Tools", variant="primary", size="lg")

                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("#### 💬 LLM Raw Output")
                        tool_llm_output = gr.Textbox(label="LLM Response", lines=3, interactive=False)
                        tool_llm_timing = gr.Markdown("*Waiting...*")

                    with gr.Group():
                        gr.Markdown("#### 🔧 Tool Execution")
                        tool_call_display = gr.Code(label="Tool Call (JSON)", language="json")
                        tool_result_display = gr.Textbox(label="Tool Result", lines=2, interactive=False)

            def run_tools(audio, prompt, enabled, temp):
                if audio is None:
                    return "", "*No audio*", "", ""

                r = engine.transcribe_with_tools(
                    audio, enabled_tools=enabled,
                    user_prompt=prompt, temperature=temp,
                )

                llm_text = r["text"]
                bd = r["breakdown"]
                timing = f"⏱️ **{r['total_ms']:.0f}ms** total"

                tool_json = ""
                tool_res = ""
                if r.get("tool_call"):
                    tool_json = json.dumps(r["tool_call"], indent=2, ensure_ascii=False)
                    tool_res = r.get("tool_result", "No result")
                else:
                    tool_json = "No tool call detected"
                    tool_res = "—"

                return llm_text, timing, tool_json, tool_res

            tool_btn.click(
                run_tools,
                inputs=[tool_audio, tool_prompt, tool_checks, tool_temp],
                outputs=[tool_llm_output, tool_llm_timing, tool_call_display, tool_result_display],
            )

        # ══════════════════════════════════════════════════════════════
        #  Tab 3: Streaming Mic
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("🎙️ Streaming Mic"):
            gr.Markdown("""
            **Real-time speech recognition** — Record, then VAD auto-segments
            and transcribes each utterance. Toggle Projector to compare.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    stream_audio = gr.Audio(
                        sources=["microphone"], type="filepath",
                        format="wav",
                        label="🎙️ Record (press Stop when done)",
                    )
                    stream_use_proj = gr.Checkbox(label="Use Projector", value=False)
                    stream_btn = gr.Button("🔍 Segment & Transcribe", variant="primary", size="lg")
                    stream_clear_btn = gr.Button("🗑️ Clear Log", variant="secondary")

                with gr.Column(scale=2):
                    stream_output = gr.Textbox(
                        label="Transcript", lines=12,
                        interactive=False, placeholder="Record something then click Segment...",
                    )
                    stream_status = gr.Markdown("*Record audio, then click Segment & Transcribe*")

            def run_stream_vad(audio, use_proj):
                if audio is None:
                    return "", "*No audio recorded*"

                import soundfile as sf
                waveform, sr_file = sf.read(audio, dtype='float32')
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                if sr_file != SAMPLE_RATE:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr_file, target_sr=SAMPLE_RATE)

                # VAD segmentation
                segments_ts = []
                if engine.vad_model is not None:
                    wav_t = torch.from_numpy(waveform).float()
                    segments_ts = engine.get_speech_timestamps(
                        wav_t, engine.vad_model,
                        sampling_rate=SAMPLE_RATE,
                        threshold=0.5,
                        min_speech_duration_ms=300,
                        min_silence_duration_ms=300,
                    )

                if not segments_ts:
                    # No VAD or no segments → transcribe whole file
                    segments_ts = [{"start": 0, "end": len(waveform)}]

                lines = []
                total_ms = 0
                for i, ts in enumerate(segments_ts):
                    seg = waveform[ts["start"]:ts["end"]]
                    if len(seg) < SAMPLE_RATE * 0.2:
                        continue

                    # Save segment to temp file for transcription
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        sf.write(f.name, seg, SAMPLE_RATE)
                        tmp = f.name

                    try:
                        start_s = ts["start"] / SAMPLE_RATE
                        end_s = ts["end"] / SAMPLE_RATE
                        if use_proj:
                            r = engine.transcribe_projector(tmp)
                        else:
                            r = engine.transcribe_standalone(tmp)
                        ms = r["total_ms"]
                        total_ms += ms
                        lines.append(f"[{start_s:.1f}s-{end_s:.1f}s] ({ms:.0f}ms) {r['text']}")
                    finally:
                        import os as _os
                        _os.unlink(tmp)

                transcript = "\n".join(lines) if lines else "[No speech detected]"
                n = len(lines)
                status = f"**{n} segments** | Total: **{total_ms:.0f}ms** | Avg: **{total_ms/max(n,1):.0f}ms/seg**"
                return transcript, status

            def clear_stream():
                return "", "*Cleared*"

            stream_btn.click(
                run_stream_vad,
                inputs=[stream_audio, stream_use_proj],
                outputs=[stream_output, stream_status],
            )
            stream_clear_btn.click(
                clear_stream, outputs=[stream_output, stream_status],
            )

        # ── Footer info ──
        gr.Markdown(f"""
        ---
        **Engine**: {engine.encoder_backend} ({engine.whisper_size}) | 
        **LLM**: {engine.llm_id} | 
        **Projector**: {'Loaded' if engine.projector else 'Not loaded'} | 
        **VAD**: {'Silero' if engine.vad_model else 'Not available'} |
        **Device**: {engine.device}
        """)

    return app


# ══════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AnyProjector Interactive Demo")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Projector checkpoint path")
    parser.add_argument("--whisper-size", default=DEFAULT_WHISPER, help="Whisper model size")
    parser.add_argument("--llm", default=DEFAULT_LLM, help="LLM model ID")
    parser.add_argument("--lora", default=DEFAULT_LORA, help="Path to LoRA adapter directory")
    parser.add_argument("--encoder-ckpt", default=DEFAULT_ENCODER_CKPT, help="Fine-tuned encoder state_dict path")
    parser.add_argument("--encoder", default="faster-whisper",
                        choices=["faster-whisper", "hf-whisper"],
                        help="Encoder backend")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    print("=" * 60)
    print("  🎙️ AnyProjector Interactive Demo")
    print("=" * 60)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  LoRA:        {args.lora}")
    print(f"  Encoder:     {args.whisper_size} ({args.encoder})")
    print(f"  Encoder ckpt:{args.encoder_ckpt}")
    print(f"  LLM:         {args.llm}")
    print(f"  Port:        {args.port}")
    print("=" * 60)

    engine = DemoEngine(
        checkpoint_path=args.checkpoint,
        whisper_size=args.whisper_size,
        llm_id=args.llm,
        encoder_backend=args.encoder,
        lora_path=args.lora,
        encoder_ckpt=args.encoder_ckpt,
    )

    app = build_ui(engine)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
