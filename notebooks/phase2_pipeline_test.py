# %% [markdown]
# # 🧪 Phase 2 Pipeline Test — AnyProjector
# Kiểm tra từng bước trong pipeline Alignment Training.
# Chạy từng cell để xem output mỗi khâu.

# %% Cell 1: Imports & Environment
import sys, os, json
import torch
import torchaudio
import soundfile as sf
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent if '__file__' in dir() else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %% Cell 2: Load Dataset Metadata
DATASET_DIR = PROJECT_ROOT / "dataset" / "phase2_alignment"
METADATA_FILE = DATASET_DIR / "metadata.jsonl"

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = [json.loads(line) for line in f if line.strip()]

print(f"📊 Loaded {len(metadata)} entries from metadata.jsonl")
print(f"   Original: {sum(1 for m in metadata if m.get('augmented_from') is None)}")
print(f"   Augmented: {sum(1 for m in metadata if m.get('augmented_from') is not None)}")
print(f"\n--- Sample entry ---")
print(json.dumps(metadata[0], indent=2, ensure_ascii=False))

# %% Cell 3: Load & Inspect Audio
sample = metadata[0]
audio_path = DATASET_DIR / sample["audio_file"]

# Use soundfile (avoids torchcodec DLL issues on Windows)
import numpy as np
audio_np, sr = sf.read(str(audio_path))
waveform = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)  # (1, samples)

print(f"🎤 Audio: {sample['audio_file']}")
print(f"   Transcript: {sample['transcript']}")
print(f"   Shape: {waveform.shape}  (channels, samples)")
print(f"   Sample rate: {sr} Hz")
print(f"   Duration: {waveform.shape[1] / sr:.2f}s")

# Resample to 16kHz if needed
TARGET_SR = 16000
if sr != TARGET_SR:
    resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
    waveform = resampler(waveform)
    print(f"   Resampled: {waveform.shape} @ {TARGET_SR}Hz")

# %% Cell 4: Load Whisper Encoder (frozen)
from transformers import WhisperModel, WhisperProcessor

ENCODER_ID = "openai/whisper-medium"
print(f"🔊 Loading encoder: {ENCODER_ID}")

processor = WhisperProcessor.from_pretrained(ENCODER_ID)
whisper_full = WhisperModel.from_pretrained(ENCODER_ID)
encoder = whisper_full.encoder.to(DEVICE).eval()
del whisper_full  # Free decoder memory

# Freeze
for p in encoder.parameters():
    p.requires_grad = False

encoder_dim = encoder.config.d_model
total_params = sum(p.numel() for p in encoder.parameters())
print(f"   encoder_dim (d_model): {encoder_dim}")
print(f"   Parameters: {total_params:,} (all frozen)")

# %% Cell 5: Audio → Encoder Output
# Process audio through Whisper's feature extractor → mel spectrogram → encoder
input_features = processor(
    waveform.squeeze().numpy(),
    sampling_rate=TARGET_SR,
    return_tensors="pt"
).input_features.to(DEVICE)

print(f"📥 Mel spectrogram shape: {input_features.shape}")

with torch.no_grad():
    encoder_output = encoder(input_features).last_hidden_state

print(f"📤 Encoder output shape: {encoder_output.shape}")
print(f"   = (batch={encoder_output.shape[0]}, seq_len={encoder_output.shape[1]}, encoder_dim={encoder_output.shape[2]})")

# %% Cell 6: Create Projector (trainable)
from src.projector import AnyProjector
from transformers import AutoConfig

# Auto-detect LLM hidden_size (chỉ tải config.json, không tải weights)
# ⚡ Dùng Qwen2.5-1.5B để test pipeline (pure CausalLM, hidden_size=1536, nhẹ)
# ⚠️ Gemma 4 multimodal cần: fix bitsandbytes + GPU >8GB VRAM
LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"
llm_config = AutoConfig.from_pretrained(LLM_ID)
LLM_DIM = llm_config.hidden_size
print(f"🔍 Auto-detected LLM dim from {LLM_ID}: {LLM_DIM}")

projector = AnyProjector(encoder_dim=encoder_dim, llm_dim=LLM_DIM).to(DEVICE)
projector.train()

print(f"🔗 {projector}")
print(f"\n--- Gradient check ---")
for name, p in projector.named_parameters():
    print(f"   {name}: requires_grad={p.requires_grad}, shape={list(p.shape)}")

# %% Cell 7: Projector Forward
audio_embeds = projector(encoder_output)

print(f"📥 Input:  {encoder_output.shape}  (batch, {encoder_output.shape[1]}, encoder_dim={encoder_dim})")
print(f"📤 Output: {audio_embeds.shape}  (batch, {audio_embeds.shape[1]}, llm_dim={LLM_DIM})")
print(f"   Temporal compression: {encoder_output.shape[1]} → {audio_embeds.shape[1]} tokens (÷2)")

# %% Cell 8: Load LLM + Tokenizer (4-bit on GPU)
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# LLM_ID đã được set ở Cell 6 (đổi ở đó, không đổi ở đây)
print(f"🧠 Loading LLM: {LLM_ID}")

# Free VRAM: tạm di encoder sang CPU
encoder.cpu()
torch.cuda.empty_cache()
print("   (Encoder moved to CPU to free VRAM)")

tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Detect if multimodal (Gemma4, LLaVA, etc.) vs pure CausalLM
is_multimodal = hasattr(llm_config, 'text_config') or hasattr(llm_config, 'vision_config')
print(f"   Multimodal: {is_multimodal}")

# NOTE: bitsandbytes 0.49.2 has _is_hf_initialized bug with transformers 5.7+
#       → skip 4-bit, load bfloat16 directly. Encoder đã ở CPU → 8GB VRAM đủ.

if is_multimodal:
    # Load full model → CPU (32GB RAM đủ) → extract language_model → GPU
    print("   Loading multimodal model to CPU first...")
    full_model = AutoModel.from_pretrained(
        LLM_ID,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    # Extract text backbone
    llm = full_model.language_model
    # Xóa phần không cần (vision/audio) để free RAM
    del full_model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    # Chuyển language_model lên GPU
    llm = llm.to(DEVICE)
    print(f"   ✅ Extracted language_model → {DEVICE}")
else:
    # Pure CausalLM (Qwen, Llama, Mistral...)
    print("   Loading CausalLM in bfloat16...")
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_ID, device_map="auto", torch_dtype=torch.bfloat16,
    )

# Freeze
for p in llm.parameters():
    p.requires_grad = False
llm.eval()

# Move encoder back to GPU
encoder.to(DEVICE)
print("   (Encoder moved back to GPU)")

actual_llm_dim = llm.config.hidden_size
total_llm_params = sum(p.numel() for p in llm.parameters())
print(f"   hidden_size: {actual_llm_dim}")
print(f"   Parameters: {total_llm_params:,} (all frozen)")
print(f"   Vocab size: {llm.config.vocab_size}")

assert actual_llm_dim == LLM_DIM, f"LLM dim mismatch! Expected {LLM_DIM}, got {actual_llm_dim}"
print(f"   ✅ LLM dim matches Projector output!")

# %% Cell 9: Create Text Prompt Embeddings
PROMPT_TEXT = "Phiên âm đoạn audio sau bằng tiếng Việt:"
prompt_tokens = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(DEVICE)

embed_layer = llm.get_input_embeddings()
with torch.no_grad():
    prompt_embeds = embed_layer(prompt_tokens)

print(f"📝 Prompt: \"{PROMPT_TEXT}\"")
print(f"   Token IDs: {prompt_tokens.shape} → {prompt_tokens[0].tolist()}")
print(f"   Prompt embeddings: {prompt_embeds.shape}")
print(f"   = (batch=1, {prompt_embeds.shape[1]} prompt tokens, llm_dim={LLM_DIM})")

# %% Cell 10: Combine & LLM Forward
# Concatenate: [prompt_embeds | audio_embeds]
# Cast to LLM dtype (projector=fp32, LLM=bf16 → cần match)
llm_dtype = next(llm.parameters()).dtype
combined = torch.cat([prompt_embeds, audio_embeds], dim=1).to(llm_dtype)
print(f"🔗 Combined input:")
print(f"   prompt_embeds:  {prompt_embeds.shape[1]} tokens")
print(f"   audio_embeds:   {audio_embeds.shape[1]} tokens")
print(f"   combined:       {combined.shape[1]} tokens")
print(f"   combined dtype: {combined.dtype}")
print(f"   combined shape: {combined.shape}")

# Forward through LLM
with torch.no_grad():
    llm_output = llm(inputs_embeds=combined)
    logits = llm_output.logits

print(f"\n📤 LLM output logits: {logits.shape}")
print(f"   = (batch=1, seq_len={logits.shape[1]}, vocab_size={logits.shape[2]})")

# %% Cell 11: Calculate Loss
# Target = transcript tokens
transcript = sample["transcript"]
target_tokens = tokenizer(
    transcript,
    return_tensors="pt",
    padding=False,
    add_special_tokens=False,
).input_ids.to(DEVICE)

print(f"🎯 Target transcript: \"{transcript}\"")
print(f"   Target token IDs shape: {target_tokens.shape}")

# The loss is computed on the LAST N tokens of the LLM output,
# where N = len(target_tokens). We shift logits by 1 for autoregressive prediction.
n_target = target_tokens.shape[1]
n_combined = combined.shape[1]

# We need combined to be long enough. Typically we'd concatenate target embeds too
# for teacher forcing. Let's do it properly:
target_embeds = embed_layer(target_tokens)
full_input = torch.cat([prompt_embeds, audio_embeds, target_embeds], dim=1).to(llm_dtype)

print(f"\n--- Teacher Forcing Setup ---")
print(f"   full_input: prompt({prompt_embeds.shape[1]}) + audio({audio_embeds.shape[1]}) + target({target_embeds.shape[1]}) = {full_input.shape[1]} tokens")

# Forward with full input (teacher forcing)
outputs = llm(inputs_embeds=full_input)
logits = outputs.logits  # (1, total_seq_len, vocab_size)

# Loss: predict target tokens from the positions after audio
# Shift: logits[..., audio_end:-1, :] predicts target_tokens[..., :]
audio_end = prompt_embeds.shape[1] + audio_embeds.shape[1]
predict_logits = logits[:, audio_end - 1 : audio_end - 1 + n_target, :].float()

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(
    predict_logits.reshape(-1, predict_logits.shape[-1]),
    target_tokens.reshape(-1),
)

print(f"\n📉 Loss = {loss.item():.4f}")
print(f"   (random baseline ≈ {torch.log(torch.tensor(float(llm.config.vocab_size))).item():.2f})")

# %% Cell 12: Backward — Verify Gradients
loss.backward()

print("🔙 Backward pass completed!\n")
print("--- Projector gradients ---")
for name, p in projector.named_parameters():
    grad_norm = p.grad.norm().item() if p.grad is not None else 0
    print(f"   {name}: grad_norm={grad_norm:.6f}")

print("\n--- Encoder gradients (should be None) ---")
has_grad = any(p.grad is not None for p in encoder.parameters())
print(f"   Any encoder param has grad? {has_grad} ({'❌ BUG!' if has_grad else '✅ Correct'})")

print("\n--- LLM gradients (should be None) ---")
has_grad = any(p.grad is not None for p in llm.parameters())
print(f"   Any LLM param has grad? {has_grad} ({'❌ BUG!' if has_grad else '✅ Correct'})")

# %% Cell 13: Mini Training Loop (3 steps)
projector.zero_grad()
optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-4, weight_decay=0.01)

print("🏋️ Mini training loop (3 steps, 1 sample):\n")
for step in range(1, 4):
    optimizer.zero_grad()

    # Forward pipeline
    with torch.no_grad():
        enc_out = encoder(input_features).last_hidden_state
    
    audio_emb = projector(enc_out)
    
    with torch.no_grad():
        p_emb = embed_layer(prompt_tokens)
        t_emb = embed_layer(target_tokens)
    
    full_in = torch.cat([p_emb, audio_emb, t_emb], dim=1).to(llm_dtype)
    out = llm(inputs_embeds=full_in)
    
    pred_logits = out.logits[:, audio_end - 1 : audio_end - 1 + n_target, :].float()
    step_loss = loss_fn(pred_logits.reshape(-1, pred_logits.shape[-1]), target_tokens.reshape(-1))
    
    step_loss.backward()
    torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
    optimizer.step()
    
    print(f"   Step {step}: loss = {step_loss.item():.4f}")

print("\n✅ Pipeline Phase 2 hoạt động! Projector nhận gradient, Encoder + LLM frozen.")
print("   Tiếp theo: Scale lên full dataset + nhiều epoch để loss hội tụ.")
