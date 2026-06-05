"""
demo_phase3.py - Demo Web cho Phase 3 Inference.

Pipeline: Audio → Whisper Encoder (+ fine-tuned layer) → Projector → LLM (+ LoRA) → Response/Tool Call

Usage:
    python -m src.demo_phase3
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo_phase3")


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
@dataclass
class DemoConfig:
    encoder_id: str = "openai/whisper-medium"
    llm_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Checkpoints
    projector_ckpt: str = "projectorTrained/projector_final_128.pt"
    lora_dir: str = "lora/best128"
    encoder_ckpt: str = "lora/best128/encoder.pt"  # Fine-tuned encoder state_dict

    sample_rate: int = 16000
    max_audio_seconds: float = 30.0
    max_new_tokens: int = 256

    # Instruction mặc định (giống training)
    default_instruction: str = "Phản hồi câu nói dưới dạng tool call hoặc trả lời tự nhiên."


# ──────────────────────────────────────────────
# AnyProjector (inline, match training)
# ──────────────────────────────────────────────
class QFormerLayer(nn.Module):
    def __init__(self, qformer_dim, encoder_dim, num_heads=8, ffn_ratio=4, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=qformer_dim, num_heads=num_heads, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(qformer_dim)
        self.self_attn_drop = nn.Dropout(dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim=qformer_dim, num_heads=num_heads, kdim=encoder_dim, vdim=encoder_dim, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(qformer_dim)
        self.cross_attn_drop = nn.Dropout(dropout)
        ffn_hidden = qformer_dim * ffn_ratio
        self.ffn = nn.Sequential(nn.Linear(qformer_dim, ffn_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_hidden, qformer_dim))
        self.ffn_norm = nn.LayerNorm(qformer_dim)

    def forward(self, queries, encoder_out, encoder_mask=None):
        q = self.self_attn_norm(queries)
        q, _ = self.self_attn(q, q, q)
        queries = queries + self.self_attn_drop(q)
        q = self.cross_attn_norm(queries)
        q, _ = self.cross_attn(query=q, key=encoder_out, value=encoder_out, key_padding_mask=encoder_mask)
        queries = queries + self.cross_attn_drop(q)
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class AnyProjector(nn.Module):
    def __init__(self, encoder_dim, llm_dim, num_queries=64, qformer_dim=768, num_layers=2, num_heads=8, dropout=0.0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.num_queries = num_queries
        self.qformer_dim = qformer_dim
        self.pre_proj = nn.Sequential(nn.Linear(encoder_dim, encoder_dim), nn.GELU(), nn.LayerNorm(encoder_dim))
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, qformer_dim) * 0.02)
        self.layers = nn.ModuleList([QFormerLayer(qformer_dim, encoder_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(qformer_dim)
        self.output_proj = nn.Sequential(nn.Linear(qformer_dim, llm_dim))

    def forward(self, encoder_output, encoder_mask=None):
        B = encoder_output.shape[0]
        encoder_output = self.pre_proj(encoder_output)
        queries = self.query_tokens.expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, encoder_output, encoder_mask)
        return self.output_proj(self.output_norm(queries))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# Model Manager
# ──────────────────────────────────────────────
class Phase3InferenceEngine:
    """Load all models and run inference."""

    def __init__(self, config: DemoConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.projector = None
        self.llm = None
        self.tokenizer = None
        self.processor = None
        self.embed_layer = None
        self._loaded = False

    def load_all(self, progress_callback=None):
        """Load encoder, projector, LLM+LoRA."""
        from transformers import WhisperModel, WhisperProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import gc

        cfg = self.config

        def log(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        # --- 1. Whisper Encoder ---
        log(f"🔊 Loading encoder: {cfg.encoder_id}")
        self.processor = WhisperProcessor.from_pretrained(cfg.encoder_id)
        self.encoder = WhisperModel.from_pretrained(cfg.encoder_id).encoder.to(self.device)

        # Load fine-tuned encoder weights (Phase 3)
        encoder_ckpt_path = Path(cfg.encoder_ckpt)
        if encoder_ckpt_path.exists():
            log(f"  Loading fine-tuned encoder: {encoder_ckpt_path}")
            state_dict = torch.load(str(encoder_ckpt_path), map_location=self.device, weights_only=True)
            self.encoder.load_state_dict(state_dict)
            del state_dict
            log("  ✅ Fine-tuned encoder loaded")
        else:
            log(f"  ⚠️ No encoder checkpoint at {encoder_ckpt_path}, using base weights")

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # --- 2. Projector ---
        log(f"🔗 Loading projector: {cfg.projector_ckpt}")
        ckpt = torch.load(cfg.projector_ckpt, map_location="cpu", weights_only=False)
        proj_config = ckpt.get("config", {})
        proj_sd = ckpt.get("projector_state_dict", ckpt)

        layer_indices = {int(k.split(".")[1]) for k in proj_sd if k.startswith("layers.")}
        num_proj_layers = max(layer_indices) + 1 if layer_indices else 4

        self.projector = AnyProjector(
            encoder_dim=proj_config.get("encoder_dim", 1024),
            llm_dim=proj_config.get("llm_dim", 1536),
            num_queries=proj_config.get("num_queries", 128),
            qformer_dim=proj_config.get("qformer_dim", 768),
            num_layers=num_proj_layers,
            num_heads=proj_config.get("qformer_heads", 16),
            dropout=proj_config.get("dropout", 0.1),
        )
        self.projector.load_state_dict(proj_sd)
        self.projector.to(self.device).eval()
        for p in self.projector.parameters():
            p.requires_grad = False
        self.num_queries = proj_config.get("num_queries", 128)
        log(f"  ✅ Projector: {self.projector.count_parameters():,} params, {self.num_queries} queries")
        del ckpt, proj_sd

        # --- 3. LLM + LoRA ---
        log(f"🧠 Loading LLM: {cfg.llm_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            base_llm = AutoModelForCausalLM.from_pretrained(
                cfg.llm_id, quantization_config=bnb_config,
                device_map="auto", torch_dtype=torch.bfloat16,
            )
        else:
            base_llm = AutoModelForCausalLM.from_pretrained(cfg.llm_id, torch_dtype=torch.float32).to(self.device)

        # Load LoRA adapter
        lora_path = Path(cfg.lora_dir)
        if lora_path.exists() and (lora_path / "adapter_config.json").exists():
            log(f"  Loading LoRA adapter: {lora_path}")
            self.llm = PeftModel.from_pretrained(base_llm, str(lora_path))
            log("  ✅ LoRA loaded")
        else:
            log(f"  ⚠️ No LoRA at {lora_path}, using base LLM")
            self.llm = base_llm

        self.llm.eval()
        self.embed_layer = self.llm.get_base_model().get_input_embeddings() if hasattr(self.llm, "get_base_model") else self.llm.get_input_embeddings()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated() / 1024**3
            log(f"  VRAM: {alloc:.1f} GB")

        self._loaded = True
        log("✅ All models loaded!")

    @torch.no_grad()
    def generate(self, audio_data, instruction: str = None) -> tuple[str, float]:
        """Run full inference pipeline.

        Args:
            audio_data: Tuple (sample_rate, numpy_array) from Gradio audio component.
            instruction: Custom instruction, or None for default.

        Returns:
            Tuple (generated_text, inference_time_seconds).
        """
        if not self._loaded:
            return "❌ Models chưa load. Nhấn 'Load Models' trước.", 0.0

        cfg = self.config
        t0 = time.time()

        # --- Process audio ---
        if audio_data is None:
            return "❌ Không có audio.", 0.0

        sr, wav = audio_data
        wav = wav.astype(np.float32)

        # Stereo → mono
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # Normalize int16 → float
        if wav.max() > 1.0 or wav.min() < -1.0:
            wav = wav / 32768.0

        # Trim to max_audio_seconds
        max_samples = int(cfg.max_audio_seconds * cfg.sample_rate)
        if len(wav) > max_samples:
            wav = wav[:max_samples]

        # Resample if needed
        if sr != cfg.sample_rate:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=cfg.sample_rate)
            except ImportError:
                return f"❌ Sample rate {sr} != {cfg.sample_rate} và librosa chưa cài.", 0.0

        # --- Encode audio ---
        inputs = self.processor([wav], sampling_rate=cfg.sample_rate, return_tensors="pt", padding="max_length")
        input_features = inputs.input_features.to(self.device)
        enc_out = self.encoder(input_features).last_hidden_state
        audio_embeds = self.projector(enc_out.float())  # (1, nq, llm_dim)

        # --- Build prompt ---
        inst = instruction or cfg.default_instruction
        prefix = f"<|im_start|>user\n{inst}\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"

        prefix_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        suffix_ids = self.tokenizer(suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        prefix_embeds = self.embed_layer(prefix_ids)
        suffix_embeds = self.embed_layer(suffix_ids)

        llm_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        input_embeds = torch.cat([prefix_embeds, audio_embeds, suffix_embeds], dim=1).to(llm_dtype)
        attn_mask = torch.ones(1, input_embeds.shape[1], dtype=torch.long, device=self.device)

        # --- Generate ---
        outputs = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elapsed = time.time() - t0
        return generated, elapsed


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────
def create_demo():
    config = DemoConfig()
    engine = Phase3InferenceEngine(config)

    load_status = {"loaded": False}

    def load_models(progress=gr.Progress(track_tqdm=True)):
        try:
            logs = []
            def cb(msg):
                logs.append(msg)
            engine.load_all(progress_callback=cb)
            load_status["loaded"] = True
            return "\n".join(logs)
        except Exception as e:
            return f"❌ Load failed:\n{e}"

    def run_inference(audio, instruction):
        if not load_status["loaded"]:
            return "❌ Load models trước!", ""
        if audio is None:
            return "❌ Chưa có audio.", ""

        inst = instruction.strip() if instruction and instruction.strip() else None
        result, elapsed = engine.generate(audio, inst)

        # Format output
        meta = f"⏱️ {elapsed:.2f}s"
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            meta += f" | VRAM: {vram:.1f}GB"

        return result, meta

    # --- Build UI ---
    with gr.Blocks(
        title="AnyProjector — Phase 3 Demo",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css="""
        .output-box { font-family: 'JetBrains Mono', monospace; font-size: 14px; }
        .tool-call { background: #1a1a2e; color: #e94560; padding: 12px; border-radius: 8px; }
        """,
    ) as app:

        gr.Markdown("""
        # 🎙️ AnyProjector — Phase 3 Demo
        ### Speech → Understanding → Tool Call / Response

        **Pipeline:** Audio → Whisper (fine-tuned) → Q-Former 128T → Qwen 1.5B (LoRA) → Output
        """)

        # --- Load Models ---
        with gr.Row():
            with gr.Column(scale=1):
                load_btn = gr.Button("🚀 Load Models", variant="primary", size="lg")
            with gr.Column(scale=3):
                load_output = gr.Textbox(
                    label="Load Status", lines=8, interactive=False,
                    value="Nhấn 'Load Models' để bắt đầu..."
                )

        load_btn.click(fn=load_models, outputs=[load_output])

        gr.Markdown("---")

        # --- Inference ---
        gr.Markdown("### 🎤 Inference")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="🔊 Audio Input",
                    sources=["microphone", "upload"],
                    type="numpy",
                )
                instruction_input = gr.Textbox(
                    label="📝 Instruction (tùy chỉnh, để trống = mặc định)",
                    placeholder=config.default_instruction,
                    lines=2,
                )
                run_btn = gr.Button("▶️ Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="📤 Model Output",
                    lines=10,
                    interactive=False,
                    elem_classes=["output-box"],
                )
                meta_text = gr.Textbox(
                    label="ℹ️ Info",
                    lines=1,
                    interactive=False,
                )

        run_btn.click(
            fn=run_inference,
            inputs=[audio_input, instruction_input],
            outputs=[output_text, meta_text],
        )

        # --- Info ---
        gr.Markdown(f"""
        ---
        **Config:**
        - Encoder: `{config.encoder_id}` + fine-tuned last 1 layer
        - Projector: `{config.projector_ckpt}` (128 queries, frozen)
        - LLM: `{config.llm_id}` + LoRA from `{config.lora_dir}`
        - Max audio: {config.max_audio_seconds}s | Max tokens: {config.max_new_tokens}
        """)

    return app


if __name__ == "__main__":
    app = create_demo()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
