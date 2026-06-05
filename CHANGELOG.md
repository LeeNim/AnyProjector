# Changelog

All notable changes to this project will be documented in this file.

## [2026-05-27] Phase 3 Demo — LoRA + Encoder Unfreeze Inference

### Changed — Demo Web (`demo_web.py` + `demo.py`)
- **Defaults updated**: `projector_final_128.pt` (128 queries) + `lora/best128` + `encoder.pt`
- **`DemoEngine.__init__`**: Added `encoder_ckpt` parameter for fine-tuned Whisper encoder weights
- **`DemoEngine._load_whisper`**: After loading base Whisper encoder, loads `encoder.pt` state_dict (Phase 3 fine-tuned last 1 layer)
- **CLI args**: Added `--encoder-ckpt` and `--lora` to both `demo.py` and `demo_web.py`
- **Why**: Phase 3 training unfroze last 1 Whisper encoder layer → encoder weights differ from base → must load fine-tuned weights for correct inference

### Phase 3 Training Results
- **Checkpoint**: `lora/best128` — val_loss=0.677, global_step=774
- **Config**: LoRA r=16, α=32, targets=[q_proj, v_proj] → 2.2M params
- **Encoder**: Whisper-medium last 1 layer unfrozen → 12.6M params
- **Total trainable**: 14.8M params
- **Dataset**: 5,122 samples merged → 80/20 split (4,097 train / 1,025 val)


## [2026-05-23] Phase 2 v0.9.6 — Direct Cosine + Contrastive Alignment

### Failed — v0.9.5 AlignmentHead (Cross-Attention)
- **Phương pháp**: AlignmentHead (55M params) dùng text_embeds làm Query, proj_out làm K/V, loss = MSE(reconstructed, text_embeds)
- **Vấn đề 1 — Residual Leak**: Residual connection `text_embeds + attn_out` cho phép AlignmentHead bypass proj_out hoàn toàn (text tự copy chính nó → loss ≈ 0)
- **Vấn đề 2 — Fix residual vẫn fail**: Sau khi bỏ residual, AlignmentHead vẫn nuốt hết gradient (grad ratio AlignHead/Projector ≈ 100x). Projector không nhận đủ signal → collapse
- **Kết quả 3 epoch**: `cos(proj₁,proj₂)` → 0.9999 (collapse), `cos(audio→text)` → 0.0 (zero alignment), Projector grad norm → 0.0005 (near dead)
- **Kết luận**: Cross-Attention làm auxiliary module quá mạnh — nó tìm shortcut thay vì buộc Projector học

### Changed — v0.9.6: Direct Cosine + Contrastive (CLIP-style)
- **Bỏ hoàn toàn AlignmentHead** — Trainable params: 96M → ~41M (chỉ Projector)
- **Loss mới**: `cosine_align + InfoNCE` (CLIP-style contrastive)
  - Cosine: `(1 - cos(audio_vec, text_vec)).mean()` — đẩy matched pairs lại gần nhau
  - InfoNCE: Cross-entropy trên similarity matrix (B×B) — đẩy unmatched pairs ra xa
  - Contrastive loss **tự chống collapse**: nếu tất cả audio → cùng vector → unmatched cũng cos cao → loss tăng
- **Gradient trực tiếp**: Loss → Projector (không qua module trung gian)
- **Learnable temperature**: `log_temp` parameter (init từ CLIP default 0.07)
- **LR tăng**: 1e-4 → 5e-4 (vì chỉ train Projector, gradient mạnh hơn)

### Changed — Dataset
- **Chuyển từ** `NhutP/VietSpeech` (gated, 190K, 131GB) **sang** `doof-ferb/vlsp2020_vinai_100h` (public CC-BY-4.0, 56K, 11.6GB)
- Cùng format: `audio` + `transcription` columns
- 35 parquet shards, default load 10 shards ≈ 16K samples

### Results — Run 1 (12 epochs, early stopped)
- **Config**: batch=400×2, LR=5e-4, 51K samples (35 shards full), PATIENCE=7
- **Training log**: [phase2_v096_run1_training_log.csv](file:///c:/Users/suoya/OneDrive/Documents/AnyProjector/logs/phase2_v096_run1_training_log.csv)
- **Epoch summary**:

| Epoch | Train | Val | Val cos | Val con | Ratio | Grad norm |
|-------|-------|------|---------|---------|-------|-----------|
| 1 | 6.294 | 5.096 | 0.645 | 4.451 | 0.81x | 2.391 |
| 2 | 4.038 | 3.101 | 0.579 | 2.522 | 0.77x | 5.883 |
| 3 | 2.568 | 2.126 | 0.517 | 1.609 | 0.83x | 3.685 |
| 4 | 1.766 | 1.610 | 0.472 | 1.138 | 0.91x | 1.992 |
| 5 | 1.296 | 1.327 | 0.437 | 0.890 | 1.02x | 1.351 |
| 6 | 0.970 | 1.119 | 0.404 | 0.715 | 1.15x | 1.561 |
| **7** | **0.736** | **0.974** | **0.368** | **0.606** | **1.32x** | 1.086 |
| 8 | 0.573 | 0.882 | 0.334 | 0.548 | 1.54x | 0.637 |
| 9 | 0.468 | 0.799 | 0.306 | 0.492 | 1.71x | 0.706 |
| 10 | 0.385 | 0.747 | 0.280 | 0.467 | 1.94x | 0.322 |
| 11 | 0.322 | 0.698 | 0.255 | 0.443 | 2.17x | 0.289 |
| **12** | **0.271** | **0.671** | **0.238** | **0.433** | **2.47x** | 0.181 |

- **Embedding diagnostics E3**: cos(audio→text) = 0.57, cos(proj₁,proj₂) = 0.32 → ✅ Differentiated, ✅ Aligned
- **Best checkpoint**: E7 (saved by trainer, overfit ratio < 1.5)
- **Actual best**: E12 — val loss vẫn giảm đều, val cos 0.238 = cos_sim ≈ 0.76
- **Early stop**: Triggered at E12 do overfit ratio > 1.5 kéo dài (patience=7, bắt đầu từ E8)

### Lesson Learned — Overfit Ratio sai cho Contrastive Loss
- **Phát hiện**: Overfit ratio (val/train) không phản ánh đúng chất lượng cho contrastive training
- **Lý do**: Train contrastive loss giảm nhanh hơn val vì model "nhớ mặt" các negative examples trong training batch (gặp đi gặp lại qua nhiều epoch). Đây là **bản chất của InfoNCE**, không phải overfit thật
- **Bằng chứng**: Val loss KHÔNG tăng lại ở bất kỳ epoch nào (giảm đều 5.10 → 0.67), nhưng ratio vẫn tăng vì train giảm nhanh hơn
- **Fix**: Bỏ overfit ratio check trong early stopping. Chỉ dùng val loss trực tiếp (stop khi val không giảm nữa)

## [2026-05-24] Phase 2 v0.9.7 — Anti-Plateau (Dropout + SpecAugment + MoCo Queue)

### Vấn đề cần giải quyết
- **Val contrastive plateau**: E10→E12 delta chỉ còn 0.010/epoch (vs 0.056 ở E8→E9)
- **Root cause phân tích**: 3 yếu tố cùng lúc:
  1. **Neuron memorization**: Projector không có dropout → neurons chuyên biệt hóa cho từng sample
  2. **Audio memorization**: Model "nhớ mặt" spectrogram → cùng audio qua 12 epoch = 12 lần giống hệt
  3. **Negative exhaustion**: batch=400, epoch=102 batches, sau 12 epoch = ~1,224 batch compositions → model đã "giải" hết tổ hợp negatives

### Kỹ thuật 1: Dropout (Hinton et al., 2012)
- **Nguồn gốc**: "Improving neural networks by preventing co-adaptation of feature detectors" — Geoffrey Hinton, University of Toronto
- **Lý thuyết**: Mỗi forward pass, random tắt 10% neurons → buộc TOÀN BỘ network phải đóng góp vào output, không cho neurons chuyên biệt hóa. Tương đương train ~2^N sub-networks đồng thời (N = số neurons), inference là ensemble của tất cả
- **Tại sao an toàn cho embedding alignment**: Output phải gần text embedding → dropout buộc mapping robust hơn → inference (dropout OFF) cho vector chính xác hơn training. CLIP, BLIP-2 Q-Former, Sentence-BERT đều dùng dropout trong projection layers
- **Config**: dropout=0.1 sau self-attention, cross-attention, và FFN trong Q-Former

### Kỹ thuật 2: SpecAugment (Park et al., 2019, Google Brain)
- **Nguồn gốc**: "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" — Google Brain. Lần đầu đạt SOTA trên LibriSpeech bằng augmentation thay vì model lớn hơn
- **Lý thuyết**: Trên mel spectrogram (2D: freq × time), random zero-mask 1-2 dải tần số (frequency masking) và 1-2 đoạn thời gian (time masking). Cùng 1 audio nhưng mỗi epoch "nhìn" khác → tạo data diversity miễn phí, buộc model học robust features thay vì memorize exact spectrogram
- **Tại sao hiệu quả cho contrastive**: Cùng audio + cùng text label nhưng mel spectrogram khác nhau mỗi epoch → model phải dựa vào semantic content (ngữ nghĩa) thay vì acoustic fingerprint (dấu vân tay âm thanh)
- **Config**: freq_masks=2 (width≤27), time_masks=2 (width≤100). Áp dụng SAU Whisper processor, TRƯỚC encoder. Chỉ khi training

### Kỹ thuật 3: MoCo Negative Queue (He et al., 2020, Facebook AI / FAIR)
- **Nguồn gốc**: "Momentum Contrast for Unsupervised Visual Representation Learning" — Kaiming He, FAIR. Giải quyết bottleneck lớn nhất của contrastive learning: số lượng negatives bị giới hạn bởi batch size
- **Lý thuyết**: Duy trì FIFO queue chứa embeddings từ các batch trước. Mỗi batch mới, contrastive loss so sánh không chỉ với in-batch negatives (400) mà còn với queue (4096+). Queue cung cấp diverse negatives mà không cần tăng batch size hay VRAM
- **Vấn đề staleness (thường gặp)**: Trong MoCo gốc, queue chứa embeddings từ model CŨ (đã update nhiều lần) → stale → cần momentum encoder (EMA copy, update rate 0.999) để giảm staleness
- **Tại sao KHÔNG cần momentum encoder trong trường hợp này**: Queue chứa TEXT embeddings từ **embed_layer FROZEN** → cùng 1 text sẽ LUÔN cho cùng 1 vector → KHÔNG CÓ staleness. Đây là lợi thế đặc biệt của kiến trúc "frozen embed_layer" — cho phép MoCo queue đơn giản hóa triệt để
- **Asymmetric design**: 
  - audio→text: dùng queue (4096 text negatives) → contrastive signal cực mạnh
  - text→audio: chỉ in-batch (vì audio embeddings thay đổi theo projector updates → sẽ stale)
- **Config**: queue_size=4096, ~24MB RAM

### Thay đổi Early Stopping
- **Bỏ overfit ratio check** (sai cho contrastive, xem lesson learned ở trên)
- **Thêm slope-based stopping**: Dừng nếu val loss giảm < MIN_DELTA trong PATIENCE epoch liên tiếp
- **Chỉ dùng val loss trực tiếp** làm tiêu chí save best

### Bugfix — MoCo Queue Inplace Gradient Error
- **Lỗi**: `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [1536, 4096]]`
- **Nguyên nhân**: `self.text_queue[:queue_size]` tạo **view** (không copy) → `_enqueue` ghi đè inplace → autograd phát hiện tensor đã bị modified trong computation graph
- **Fix**: `self.text_queue[:queue_size].clone().detach()` — tạo bản copy riêng, cắt khỏi computation graph

### Results — Run 1 (43 epochs total: 2 fresh + 41 resumed)
- **Config**: batch=400×2=800 effective, LR=5e-4 (cosine decay), 51K samples, dropout=0.1
- **SpecAugment**: freq=2×27, time=2×100
- **MoCo Queue**: 4096 text embeddings, ~24MB
- **Trainable**: 41,663,488 params (Projector only)
- **VRAM**: 2.2-2.7GB / 95GB
- **Training log**: [phase2_v097_run1_training_log.csv](file:///c:/Users/suoya/OneDrive/Documents/AnyProjector/logs/phase2_v097_run1_training_log.csv)
- **Tổng thời gian**: ~43 × 21.7 phút ≈ **15.5 giờ**

#### Epoch Summary (milestones)

| Epoch (overall) | Epoch (CSV) | Train | Val | Val cos | Val con | Ratio | cos_sim | Grad |
|:---:|:---:|-------|------|---------|---------|-------|---------|------|
| **1** | — | 7.320 | 4.748 | 0.656 | 4.092 | 0.65x | 0.33 | 7.69 |
| **2** | — | 5.315 | 2.843 | 0.583 | 2.260 | 0.53x | 0.41 | 3.69 |
| 3 | 1* | 4.039 | 2.040 | 0.536 | 1.504 | 0.51x | — | 2.37 |
| 7 | 5 | 2.113 | 0.894 | 0.454 | 0.441 | 0.42x | — | 1.72 |
| 12 | 10 | 1.332 | 0.613 | 0.401 | 0.213 | 0.46x | — | 1.35 |
| 17 | 15 | 0.993 | 0.537 | 0.367 | 0.170 | 0.54x | — | 1.08 |
| 22 | 20 | 0.805 | 0.492 | 0.340 | 0.153 | 0.61x | — | 0.86 |
| 27 | 25 | 0.666 | 0.464 | 0.300 | 0.164 | 0.70x | — | 0.89 |
| 32 | 30 | 0.585 | 0.437 | 0.278 | 0.159 | 0.75x | — | 0.72 |
| 34 | 32 | 0.554 | 0.431 | 0.268 | 0.164 | 0.78x | **0.73** | 0.76 |
| 37 | 35 | 0.530 | 0.429 | 0.262 | 0.167 | 0.81x | — | 0.77 |
| **43** | **41** | **0.478** | **0.414** | **0.246** | **0.168** | **0.87x** | **~0.75** | 0.70 |

\* CSV epoch 1 = overall epoch 3 (resumed từ E2 checkpoint, epoch counter reset về 1)

#### Embedding Diagnostics
- **E1**: cos(audio→text)=0.46, cos(proj₁,proj₂)=0.65 → differentiated ngay từ đầu (nhờ contrastive)
- **E2**: cos(audio→text)=0.51, cos(proj₁,proj₂)=0.34 → diversity tốt
- **E34** (CSV E32): cos(audio→text)=0.73 → alignment rõ ràng

#### Phân tích 3 giai đoạn training

**Phase A — Rapid Learning (E1→E12):**
- Val: 4.748 → 0.613 (↓ 87%)
- Tốc độ giảm ~0.4/epoch
- Queue chưa đầy (E1: 1633/4096, E2: 3266/4096)
- Gradient rất mạnh (1.3-7.7)
- Model đang học các features cơ bản: phân biệt ngôn ngữ, nhận diện phoneme patterns

**Phase B — Steady Improvement (E12→E27):**
- Val: 0.613 → 0.464 (↓ 24%)
- Tốc độ giảm ~0.01/epoch
- Queue đã đầy, contrastive signal ổn định
- Val cosine giảm đều (0.40 → 0.30) = alignment cải thiện
- Val contrastive bắt đầu đi ngang (0.21 → 0.16)

**Phase C — Plateau (E27→E43):**
- Val: 0.464 → 0.414 (↓ 10.8% trong 16 epoch)
- Tốc độ giảm ~0.003/epoch
- Val cosine: 0.30 → 0.25 (chậm, ~0.003/epoch)
- **Val contrastive: 0.164 → 0.168 (TĂNG nhẹ, đã plateau thật)**
- Ratio: 0.70 → 0.87 (train bắt kịp val nhưng chưa overfit)

#### Đánh giá hiệu quả Anti-Plateau

| So sánh | v0.9.6 (no anti-plateau) | v0.9.7 (anti-plateau) |
|---|---|---|
| Epochs chạy được | 12 (early stopped) | **43** (plateau tự nhiên) |
| Val cos cuối | 0.238 (cos_sim 0.76) | 0.246 (cos_sim **0.75**) |
| Val con cuối | 0.433 | **0.168** (↓ 61%) |
| Ratio cuối | 2.47x (overfit nặng) | **0.87x** (healthy) |
| Grad cuối | 0.181 (near dead) | **0.70** (active) |

**Kết luận**: Anti-plateau thành công trong việc:
- ✅ **Ngăn overfit hoàn toàn** (ratio 0.87 vs 2.47)
- ✅ **Duy trì gradient khỏe** suốt 43 epoch (0.70 vs 0.181)
- ✅ **Contrastive tốt hơn nhiều** (0.168 vs 0.433)
- ⚠️ **Nhưng KHÔNG cải thiện alignment (cosine)** so với v0.9.6 (0.246 vs 0.238)

#### Lesson Learned — Giới hạn kiến trúc Mean-Pool + Cosine

- **Phát hiện**: cos_sim ≈ 0.75 là trần của approach "mean-pool 64 tokens → 1 vector → cosine similarity"
- **Lý do**: Mean-pooling mất sequence information. 64 query tokens chứa thông tin khác nhau (phoneme, prosody, timing...) nhưng bị nén thành 1 vector trung bình → chi tiết bị mất → alignment chỉ ở mức "chủ đề chung"
- **Bằng chứng**: Cả v0.9.6 (overfit) lẫn v0.9.7 (regularized) đều converge về cos_sim ≈ 0.75, bằng 2 con đường khác nhau, cùng 1 giới hạn
- **Kết luận**: Để cải thiện alignment hơn nữa, cần thay đổi kiến trúc (token-level alignment, hoặc chuyển sang Phase 3 LoRA để LLM học đọc 64 tokens trực tiếp)
- **Quyết định**: Dừng Phase 2 tại checkpoint E43. cos_sim ≈ 0.75 đủ tốt để Phase 3 LoRA tiếp tục fine-tune

---




## [2026-05-23] Phase 3 Dataset & Phase 2 Architectural Redesign


### Added — Phase 3 Tool-Calling Dataset
- **HuggingFace Dataset**: `Niem/speech-massive-vie-tool-calling`
  - Source: `doof-ferb/Speech-MASSIVE_vie` (Vietnamese speech + intent labels)
  - Response generation: `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` on Colab T4
  - Structure: `train=115`, `validation=2033`, `test=2974` → **Total 5,122 samples**
  - Columns: `id`, `audio` (16kHz), `instruction`, `input` (utt), `output` (tool-call response), `intent_str`, `scenario_str`, + speaker metadata
  - **Audio preserved**: Dùng `.map()` trên dataset gốc để giữ nguyên cột audio khi push
  - Covers 60+ intents across alarm, audio, calendar, cooking, email, iot, lists, music, news, play, qa, recommendation, social, takeaway, transport, weather, general
- **Notebook**: [generate_dataset_colab.ipynb](file:///c:/Users/suoya/OneDrive/Documents/AnyProjector/notebooks/generate_dataset_colab.ipynb) — Self-contained Colab pipeline for dataset generation
- **Notebook**: [train_phase3_end2end.ipynb](file:///c:/Users/suoya/OneDrive/Documents/AnyProjector/notebooks/train_phase3_end2end.ipynb) — Phase 3 End-to-End training (Freeze Projector + LoRA LLM + Unfreeze Whisper top layers)

### Discovered — Phase 2 Trigger Token Phenomenon
- **Problem**: Ở Phase 2 (Alignment), LLM bị frozen hoàn toàn. Projector bị ép (qua CE Loss) phải làm LLM sinh ra text phiên âm. Kết quả: Projector học được cách tạo ra **"trigger tokens"** — các vector đặc biệt hack vào attention layers của LLM, lấn át hoàn toàn prefix prompt, buộc LLM chỉ dịch audio → text mà phớt lờ mọi instruction.
- **Evidence**: Ở Demo, khi kết hợp Projector + LLM, LLM chỉ nhả ra phiên âm (ASR) bất kể prompt prefix nói gì. LLM không "không hiểu" prefix — mà Projector đã lấn át nó.
- **Impact**: Phase 3 LoRA vẫn sẽ khắc phục được (LLM học lại cách cân bằng attention), nhưng trigger tokens là "technical debt" không cần thiết.

### Changed — Phase 2 Redesign: Embedding Alignment (Cross-Attention)
- **Quyết định**: Chuyển Phase 2 từ "Full LLM Forward + CE Loss" sang "Embedding Alignment" để loại bỏ trigger token problem.
- **Các hướng đã phân tích**:
  1. **Mean Pooling + MSE/Cosine**: Đơn giản nhất, nhưng mất thông tin thứ tự (sequence info).
  2. **CTC Loss**: Giữ thứ tự, nhưng cần classification head → output không nằm trong embedding space của LLM.
  3. **Token-wise Pad/Truncate**: Đơn giản, giữ sequence, nhưng text > 64 tokens bị cắt.
  4. **Token-wise Cross-Attention** ✅: Text embeds làm Query, Projector output làm Key/Value. Module cross-attention chỉ dùng khi train (bỏ khi inference). Giữ sequence info, handle variable-length text, output nằm trong LLM embedding space.
- **Lựa chọn**: Token-wise Cross-Attention Alignment (Cách B)
- **Lợi ích**:
  - Không cần load full LLM khi train Phase 2 (chỉ cần embed_layer) → tiết kiệm ~3GB VRAM
  - Projector học "pure semantic alignment" thay vì hack LLM
  - Train nhanh hơn nhiều (không forward qua 28 transformer layers)
- **Notebook**: [train_phase2_embedding_align.ipynb](file:///c:/Users/suoya/OneDrive/Documents/AnyProjector/notebooks/train_phase2_embedding_align.ipynb)

## [Unreleased] - 2026-05-22

### Added
- **LoRA Dataset Generation**: Created [generate_lora_dataset.py](file:///c:/Users/suoya/OneDrive/Documents/AnyProjector/scripts/generate_lora_dataset.py) to load `doof-ferb/Speech-MASSIVE_vie` dataset and map transcriptions (`utt`) to standardized tool calling and conversational responses.
  - Supports multiple standard tool call representations: `xml_json` (standard XML tags), `json_only` (universal wrapper), `react` (standard ReAct Action/Response blocks), and `plain_text`.
  - Implemented configurable chunk sizes (default 500 samples) using `--limit` and `--offset` to prevent memory and LLM context window issues during conversion.
  - Supports JSON, CSV, and Parquet formats for easy loading.
- **Premium Gemini Validation Dataset**: Overwrote and fully expanded [generate_val_gemini.py](file:///c:/Users/suoya/OneDrive/Documents/AnyProjector/scripts/generate_val_gemini.py) to replace dynamic templates with 100% hand-crafted premium Gemini 3.5 Flash-quality Vietnamese responses.
  - Created a comprehensive registry of high-quality, diverse conversational responses with accurate XML-JSON tool calling format.
  - Successfully generated [speech_massive_lora_500_val.json](file:///c:/Users/suoya/OneDrive/Documents/AnyProjector/dataset/speech_massive_lora_500_val.json) containing 500 premium validation records.

- **Premium Gemini Training Dataset Upgrade**: Directly upgraded [speech_massive_lora_500_offset_0.json](file:///c:/Users/suoya/OneDrive/Documents/AnyProjector/dataset/speech_massive_lora_500_offset_0.json) by replacing the basic/dummy generator outputs with 100% hand-crafted premium Gemini 3.5 Flash-level conversational and XML-JSON tool calling responses in Vietnamese.
  - Ensured all responses are extremely natural, contextualized, and adhere to standard tool-calling representation for robust LoRA training.

### Fixed
- CORS error when calling `/api/info` and `/api/transcribe` from `file://` protocols in `demo_web.html` by enabling standard CORS middleware in FastAPI backend.
- CT2 encoder incompatibility by migrating projector pipeline entirely to Hugging Face Whisper encoder backend.
- Audio volume and playback issues by redesigning the recording backend using standard Web Audio APIs.

## [2026-05-22] Dataset Generation (Next Chunk)
- **Changed**: Extracted and annotated the next 50 samples from validation split (offset 500-549).
- **Files Edited**: dataset/speech_massive_lora_50_offset_500.json
- **Why**: Continued manual generation of high-quality Gemini responses directly into the file as requested by the user.

## [2026-05-22] Gemini Canvas/Web Integration Tools
- **Added**: Created [convert_canvas_csv_to_json.py](file:///c:/Users/suoya/OneDrive/Documents/AnyProjector/scripts/convert_canvas_csv_to_json.py) to parse and normalize the CSV output generated via Gemini Web / Canvas, converting it directly into standardized JSON chunks.
- **Changed**: Upgraded [compile_final_dataset.py](file:///c:/Users/suoya/OneDrive/Documents/AnyProjector/scripts/compile_final_dataset.py) to automatically scan, match, and compile all `speech_massive_lora_*.json` files (including the newly generated canvas chunk) into the final LoRA training set, avoiding hardcoding.
- **Why**: Transitioned dataset generation to a hybrid flow utilizing the Gemini Web Consumer App (Canvas) to bypass API quota constraints while retaining automated processing and strict schema validation in the local environment.
