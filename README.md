# AnyProjector

**Xây dựng mô hình cầu nối cơ bản giữa mô hình âm thanh và mô hình tạo sinh văn bản.**

> *Any Encoder, Any LLM, One Bridge.*

AnyProjector là một hệ thống **Bridge Model** hạng nhẹ sử dụng kiến trúc **Q-Former** để chiếu (project) không gian biểu diễn đặc trưng âm thanh từ bộ mã hóa **Whisper** sang không gian embedding của LLM **Qwen2.5**, trong khi giữ nguyên trọng số đóng băng (frozen weights) của cả hai Foundation Model.

```
Audio (WAV 16kHz) → [Whisper Encoder ❄️ 769M] → [Projector 🔥 42M] → [Qwen2.5 ❄️ 1.5B] → Text
```

---

## Kiến trúc

```
AnyProjector/
├── src/                          # Core source code
│   ├── projector.py              # Q-Former Projector (4L/16H, 42M params)
│   ├── train_phase2.py           # Phase 2: Embedding Alignment training
│   ├── trainer.py                # Training loop, early stopping, logging
│   ├── dataset.py                # Dataset loading & preprocessing
│   ├── model_loader.py           # Whisper + Qwen2.5 model loading
│   ├── config.py                 # Configuration management
│   ├── system.py                 # System utilities
│   ├── demo_phase3.py            # Phase 3 demo inference
│   └── app.py                    # Gradio UI
├── scripts/                      # Pipeline scripts
│   ├── demo.py                   # CLI demo inference
│   ├── demo_web.py               # FastAPI web demo
│   ├── demo_web.html             # Web UI frontend
│   ├── demo_tools.py             # Tool definitions for demo
│   ├── inference.py              # Batch inference
│   ├── evaluate_wer.py           # WER evaluation
│   ├── plot_training_log.py      # Training curve visualization
│   ├── compare_projectors.py     # Cross-version comparison
│   ├── generate_lora_dataset.py  # LoRA dataset generation
│   ├── generate_*.py             # Dataset generation scripts
│   ├── compile_final_dataset.py  # Dataset compilation
│   └── export_colab_notebook.py  # Colab notebook export
├── notebooks/                    # Colab training notebooks
│   ├── train_phase2_*.ipynb      # Phase 2 training (multiple versions)
│   ├── train_phase3_*.ipynb      # Phase 3 LoRA training
│   └── generate_dataset_*.ipynb  # Dataset generation
├── config/
│   └── default_config.yaml       # Default hyperparameters
├── tools/
│   └── dataset_recorder/         # Audio recording tools
├── requirements.txt
├── CHANGELOG.md                  # Detailed development log
└── README.md
```

---

## Tính năng chính

- 🔌 **Modular Bridge Architecture** — Projector Q-Former kết nối bất kỳ Audio Encoder ↔ LLM nào
- 🎯 **MaxSim Token-level Alignment** — Tiên phong áp dụng ColBERT MaxSim cho Audio-Text Q-Former
- 📉 **42M Trainable Parameters** — Chưa đến 2% tổng hệ thống, huấn luyện trên 1 GPU
- 🎤 **End-to-End Voice Agent** — Audio trực tiếp → LLM response, không qua ASR trung gian
- 🧠 **Tool-calling** — Stress-test downstream: gọi hàm từ giọng nói

---

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**Phần cứng tối thiểu:**
- **Training Phase 2**: 1 GPU ≥ 8GB VRAM (Cloud GPU 95GB recommended)
- **Inference**: RTX 3060+ (6GB VRAM) với 4-bit quantization

---

## Kết quả dự án

### Tổng quan Pipeline huấn luyện

| Phase | Mục tiêu | Trainable Params | Kết quả |
|---|---|---|---|
| **Phase 2** | Embedding Alignment (Projector) | 42M (Q-Former) | cos_sim = **0.80**, Token Match = **75.4%** |
| **Phase 3** | Downstream Task (LoRA + LLM) | 14.8M (LoRA + Encoder top layer) | val_loss = **0.656** (best), Tool Name Acc = **45.7%** |

### Phase 2: Embedding Alignment — Tiến hóa kiến trúc

Projector trải qua **11 phiên bản chính** (v0.2.0 → v0.11.0), mỗi phiên bản giải quyết một bottleneck cụ thể:

| Version | Kiến trúc | Kết quả | Vấn đề phát hiện |
|---|---|---|---|
| v0.2.0 | Single Linear | val_loss = 3.27 | Underfitting nghiêm trọng |
| v0.5.0–v0.8.x | MLP 2-layer | val_loss ~1.8 | Trigger Token phenomenon |
| v0.9.4 | Q-Former 6L/16H | Collapse | Quá sâu → gradient vanishing |
| v0.9.5 | Q-Former + AlignmentHead | Collapse | Residual leak, shortcut learning |
| v0.9.6 | Q-Former 4L/16H + Cosine+InfoNCE | cos_sim = 0.76 | Overfit ratio 2.47×, gradient near-dead |
| v0.9.7 | + Dropout + SpecAugment + MoCo Queue | cos_sim = 0.75 | **cos_sim plateau ≈ 0.75** (Mean-Pool ceiling) |
| **v0.10.0** | + MaxSim Loss (từ ColBERT) | cos_sim = **0.78** | Phá trần Mean-Pool |
| **v0.11.0** | MaxSim + 128 Query Tokens | cos_sim = **0.80**, Token Match **75.4%** | Best Phase 2 result |

### Phase 2 v0.9.7: Anti-Plateau (Dropout + SpecAugment + MoCo Queue)

**3 kỹ thuật chống bão hòa đã triển khai:**

1. **Dropout 0.1** (Hinton et al., 2012) — Ngăn neuron co-adaptation, buộc network robust
2. **SpecAugment** (Park et al., 2019) — Frequency/time masking trên mel spectrogram, tạo data diversity
3. **MoCo Queue 4,096** (He et al., 2020) — FIFO queue text embeddings, mở rộng negative samples mà không tăng VRAM

| Metric | v0.9.6 (no anti-plateau) | v0.9.7 (anti-plateau) |
|---|---|---|
| Epochs chạy được | 12 (early stopped) | **43** (natural plateau) |
| Overfit ratio | 2.47× (overfit nặng) | **0.87×** (healthy) |
| Gradient cuối | 0.181 (near dead) | **0.70** (active) |
| Val contrastive | 0.433 | **0.168** (↓ 61%) |
| cos_sim | 0.76 | **0.75** |

**Phát hiện quan trọng:** cos_sim ≈ 0.75 là **trần của Mean-Pool + Cosine** — cả hai phiên bản đều converge về cùng giới hạn bằng hai con đường khác nhau. Để vượt qua, cần **token-level alignment**.

### Phase 2 v0.10.0–v0.11.0: Đột phá MaxSim

**Đóng góp chính (novelty):** Áp dụng toán tử **MaxSim** (Maximum Similarity, gốc từ ColBERT — Khattab & Zaharia, 2020) vào huấn luyện Q-Former cho Audio-Text alignment.

```
MaxSim(A, T) = (1/|T|) × Σ_j max_i cos(a_i, t_j)
```

Với mỗi text token `t_j`, tìm audio token `a_i` tương đồng nhất → ép mỗi từ trong phiên âm phải có ít nhất một audio token đại diện.

**Kết quả v0.11.0 (best):**
- cos_sim = **0.80** (vượt trần 0.75)
- Token Match Rate = **75.4%** (75.4% text tokens có audio token match > 0.7)
- Hàm loss: `L = L_cosine + L_InfoNCE + 0.5 × L_MaxSim`

### Phase 3: Downstream Task — LoRA Tool-calling

**Cấu hình:**
- LoRA: r=16, α=32, targets=[q_proj, v_proj] → 2.2M params
- Whisper encoder: last 1 layer unfrozen → 12.6M params
- Dataset: 5,122 samples (Speech-MASSIVE Vietnamese)
- Best checkpoint: val_loss = **0.656** (v0.11.0 projector)

**Kết quả stress-test:**

| Metric | v0.9.8 Projector | v0.11.0 Projector |
|---|---|---|
| Phase 3 val_loss | 0.677 | **0.656** |
| Tool Name Accuracy | 45.7% | — |
| Tool Args F1 | 4.3% | — |
| WER (speech transcription) | 0.73 | — |

**Phát hiện:** Args F1 = 4.3% cho thấy LLM **hallucinate** nội dung đối số thay vì trích xuất từ audio → Error Amplification qua 28 Transformer layers. cos_sim 0.80 ở embedding input **chưa đủ** để LLM đọc trực tiếp projector output.

**Đề xuất kiến trúc mới:** Hidden State Alignment — căn chỉnh Teacher-Student trên hidden states + KL-Divergence trên logits, không chỉ ở embedding input.

---

## Cấu hình huấn luyện

```yaml
# Phase 2: Embedding Alignment
model:
  whisper: openai/whisper-medium      # 769M params, frozen
  llm: Qwen/Qwen2.5-1.5B-Instruct    # 1.5B params, frozen (chỉ dùng embed_tokens)

projector:
  type: q-former
  layers: 4
  heads: 16
  query_tokens: 64                    # hoặc 128
  pre_projection: true                # Linear + GELU + LayerNorm

training:
  batch_size: 400
  gradient_accumulation: 2            # effective batch = 800
  lr: 5e-4
  scheduler: cosine
  warmup: 5%
  dropout: 0.1
  spec_augment:
    freq_masks: 2, width: 27
    time_masks: 2, width: 100
  moco_queue: 4096
  loss: cosine + InfoNCE + 0.5×MaxSim
  dataset: doof-ferb/vlsp2020_vinai_100h  # 51K samples
  epochs: ~40-50
  patience: 7
```

---

## Dataset

| Dataset | Nguồn | Mẫu | Mục đích |
|---|---|---|---|
| `doof-ferb/vlsp2020_vinai_100h` | HuggingFace (CC-BY-4.0) | 51,000 | Phase 2 Alignment |
| `Niem/speech-massive-vie-tool-calling` | HuggingFace (tự tạo) | 5,122 | Phase 3 LoRA Tool-calling |

---

## Chạy Demo

```bash
# CLI Demo
python scripts/demo.py --projector projectorTrained/projector_final.pt

# Web Demo (FastAPI + HTML)
python scripts/demo_web.py --projector projectorTrained/projector_final.pt --lora lora/best

# Phase 3 Demo (với LoRA + fine-tuned encoder)
python -m src.demo_phase3 \
  --projector projectorTrained/projector_final_128.pt \
  --lora lora/best128 \
  --encoder-ckpt lora/encoder.pt
```

---

## Đóng góp chính

1. **Kiến trúc Q-Former cho Audio-Text Modality Bridging** — Nén 23:1 (1,500 → 64 tokens), cos_sim 0.80
2. **MaxSim cho Audio-Text Q-Former** (novelty) — Phá trần cos_sim 0.75, Token Match 75.4%
3. **Phân tích Error Amplification** — Chẩn đoán khuếch đại sai số qua 28 Transformer layers
4. **Đề xuất Hidden State Alignment** — Kiến trúc căn chỉnh 3 tầng (Input → Hidden → Output)

---

## License

MIT
