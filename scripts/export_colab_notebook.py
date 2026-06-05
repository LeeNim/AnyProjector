"""Export train_phase2.py + projector.py as a Colab notebook.

v0.9.4: Pre-projection + VietSpeech 200K + 4L/16H Q-Former + Frozen encoder
SIMPLIFIED: Stream 5 subsets → RAM → Train. No cache, no Drive backup complexity.
"""
import json
import sys
from pathlib import Path


def build_notebook() -> tuple[list, str]:
    """Build notebook cells for v0.9.4 VietSpeech training.

    Returns (cells, filename).
    """
    cells = []

    def md(lines):
        if isinstance(lines, str):
            lines = [lines]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})

    def code(lines):
        if isinstance(lines, str):
            lines = [lines]
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})

    # ── Cell 1: Title ──
    md([
        "# 🚀 AnyProjector v0.9.4 — Phase 2 Alignment\n",
        "\n",
        "Train AnyProjector Q-Former to align Whisper audio embeddings with LLM text space.\n",
        "\n",
        "**Architecture:**\n",
        "- Pre-Projection: Linear + GELU + LayerNorm (feature transform)\n",
        "- Q-Former: 4 layers × 16 heads, 64 queries\n",
        "- Encoder: Whisper-medium (fully frozen)\n",
        "- LLM: Qwen2.5-1.5B-Instruct (frozen)\n",
        "\n",
        "**Dataset:** VietSpeech — stream N samples via HF (no full download)\n",
        "\n",
        "**Pipeline:** Audio → Whisper(frozen) → Pre-Proj → Q-Former → LLM(frozen) → Loss\n",
    ])

    # ── Cell 2: Install ──
    code([
        "# Install dependencies + hf_transfer for fast downloads (3-10x)\n",
        "!pip install -q transformers datasets torch accelerate hf_transfer huggingface_hub matplotlib\n",
        "\n",
        "# Enable hf_transfer (Rust-based parallel downloader)\n",
        "import os\n",
        "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n",
        "print('✅ hf_transfer enabled for fast downloads')\n",
    ])

    # ── Cell 3: GPU check ──
    code([
        "# Verify GPU\n",
        "import torch\n",
        "if torch.cuda.is_available():\n",
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
        "    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')\n",
        "else:\n",
        "    print('⚠️ No GPU! Go to Runtime → Change runtime type → GPU')\n",
        "print(f'PyTorch: {torch.__version__}')\n",
    ])

    # ── Cell 4: HuggingFace Login ──
    md(["## 🔑 HuggingFace Login\n",
        "VietSpeech is a gated dataset — you must:\n",
        "1. Go to https://huggingface.co/datasets/NhutP/VietSpeech and accept terms\n",
        "2. Create a token at https://huggingface.co/settings/tokens\n",
        "3. Paste it below\n"])
    code([
        "from huggingface_hub import login\n",
        "\n",
        "# Option 1: Login interactively\n",
        "login()\n",
        "\n",
        "# Option 2: Set token directly (uncomment)\n",
        "# login(token='hf_YOUR_TOKEN_HERE')\n",
    ])

    # ── Cell 5: Config ──
    md(["## ⚙️ Configuration\n", "Edit these settings before running."])
    config_lines = [
        '# ============================================\n',
        '# 🔧 AnyProjector v0.9.4 Config\n',
        '# ============================================\n',
        'ENCODER_ID    = "openai/whisper-medium"\n',
        'LLM_ID        = "Qwen/Qwen2.5-1.5B-Instruct"\n',
        '\n',
        '# Dataset: VietSpeech — load N parquet shards (no full download!)\n',
        'DATASET_ID    = "NhutP/VietSpeech"\n',
        '# VietSpeech has 27 parquet shards. Each shard ≈ 3,900 samples.\n',
        '# NUM_SHARDS controls how many to download (5 ≈ 20K, 13 ≈ 50K, 27 = all 190K)\n',
        'NUM_SHARDS    = 5        # 5 shards ≈ 20K samples\n',
        '\n',
        'NUM_EPOCHS  = 50\n',
        'BATCH_SIZE  = 32     # H100: 64, A100: 32, T4: 4\n',
        'LR          = 1e-4\n',
        'GRAD_ACCUM  = 2      # Effective batch = BATCH_SIZE * GRAD_ACCUM\n',
        'SAVE_DIR    = "checkpoints/phase2/v094_vietspeech"\n',
        'PATIENCE    = 7      # Early stopping patience\n',
        'MIN_DELTA   = 0.01   # Min improvement to count\n',
        'PROMPT      = "Transcribe the following audio in Vietnamese:"\n',
        'RESUME_FROM = None   # Set to checkpoint path to resume\n',
        'NUM_WORKERS = 4      # Parallel data loading\n',
        'PRELOAD_RAM = True   # Cache all audio in RAM\n',
        '\n',
        '# --- Q-Former (4 layers × 16 heads) ---\n',
        'QFORMER_LAYERS = 4\n',
        'QFORMER_HEADS  = 16\n',
        '\n',
        '# --- LoRA (disabled for alignment) ---\n',
        'LORA_ENABLED = False\n',
        'LORA_RANK    = 8\n',
        'LORA_ALPHA   = 16\n',
        '\n',
        '# --- Encoder (fully frozen) ---\n',
        'UNFREEZE_ENCODER_LAYERS = 0\n',
    ]
    code(config_lines)

    # ── Cell 6: Mount Drive ──
    md(["## 📁 Mount Google Drive\n", "Mount Drive để lưu checkpoint trực tiếp — tránh mất khi disconnect."])
    code([
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "import os\n",
        "DRIVE_SAVE = f'/content/drive/MyDrive/AnyProjector/{SAVE_DIR}'\n",
        "os.makedirs(DRIVE_SAVE, exist_ok=True)\n",
        "print(f'✅ Drive backup folder: {DRIVE_SAVE}')\n",
    ])

    md(["## 📊 Load Dataset\n",
        "Tải N shards parquet từ VietSpeech. **Không tải full 131GB!**\n",
        "\n",
        "VietSpeech có 27 shards, mỗi shard ~3,900 samples.\n",
        "- 5 shards ≈ 20K samples\n",
        "- 13 shards ≈ 50K samples\n",
        "- 27 shards = tất cả ~190K samples\n"])
    code([
        "import time\n",
        "from datasets import load_dataset as hf_load_dataset\n",
        "\n",
        "print(f'⬇️ Loading {NUM_SHARDS} / 27 shards from {DATASET_ID}...')\n",
        "print(f'  (Each shard ≈ 3,900 samples → ~{NUM_SHARDS * 3900} total)')\n",
        "t0 = time.time()\n",
        "\n",
        "# Generate list of shard files to download\n",
        "TOTAL_SHARDS = 27\n",
        "shard_files = [f'data/train-{i:05d}-of-{TOTAL_SHARDS:05d}.parquet' for i in range(NUM_SHARDS)]\n",
        "\n",
        "ds = hf_load_dataset(\n",
        "    DATASET_ID,\n",
        "    data_files=shard_files,\n",
        "    split='train',\n",
        "    token=True,\n",
        "    verification_mode='no_checks',  # Skip split size validation for partial load\n",
        ")\n",
        "\n",
        "elapsed = time.time() - t0\n",
        "print(f'✅ Loaded {len(ds)} samples in {elapsed/60:.1f} min')\n",
        "print(f'  Columns: {ds.column_names}')\n",
        "print(f'  Example: {ds[0][\"transcription\"][:80]}...')\n",
    ])

    # ── Cell 8: Projector ──
    md(["## 🧱 AnyProjector Q-Former Module\n",
        "Pre-Projection → Q-Former (4L/16H) → Output Projection"])
    projector_src = Path("src/projector.py").read_text(encoding="utf-8")
    code([projector_src])

    # ── Cell 9: Imports ──
    md(["## 📦 Imports"])
    import_lines = [
        "import gc\n",
        "import json\n",
        "import logging\n",
        "import math\n",
        "import time\n",
        "from dataclasses import dataclass\n",
        "from pathlib import Path\n",
        "\n",
        "import numpy as np\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from datasets import load_dataset as hf_load_dataset, load_from_disk\n",
        "from torch.utils.data import Dataset, DataLoader, random_split\n",
        "\n",
        "logging.basicConfig(\n",
        "    level=logging.INFO,\n",
        '    format="%(asctime)s [%(levelname)s] %(message)s",\n',
        '    datefmt="%H:%M:%S",\n',
        "    force=True,\n",
        ")\n",
        'logger = logging.getLogger("phase2")\n',
        "logger.setLevel(logging.INFO)\n",
    ]
    code(import_lines)

    # ── Cell 10: Config dataclass ──
    md(["## 🔧 Config Dataclass"])
    code([
        "@dataclass\n",
        "class Phase2Config:\n",
        "    encoder_id: str = ENCODER_ID\n",
        "    llm_id: str = LLM_ID\n",
        f"    datasets: tuple = ((DATASET_ID, 'transcription', 'auto_split'),)\n",
        "    max_samples_per_dataset: int = 0  # 0 = use all loaded samples\n",
        "    auto_val_ratio: float = 0.1\n",
        "    sample_rate: int = 16000\n",
        "    max_audio_seconds: float = 30.0\n",
        "    num_queries: int = 64\n",
        "    qformer_dim: int = 768\n",
        "    qformer_layers: int = QFORMER_LAYERS\n",
        "    qformer_heads: int = QFORMER_HEADS\n",
        "    lora_enabled: bool = LORA_ENABLED\n",
        "    lora_rank: int = LORA_RANK\n",
        "    lora_alpha: int = LORA_ALPHA\n",
        '    lora_target_modules: tuple = ("q_proj", "v_proj")\n',
        "    unfreeze_encoder_layers: int = UNFREEZE_ENCODER_LAYERS\n",
        "    num_epochs: int = NUM_EPOCHS\n",
        "    batch_size: int = BATCH_SIZE\n",
        "    learning_rate: float = LR\n",
        "    weight_decay: float = 0.01\n",
        "    warmup_ratio: float = 0.05\n",
        "    max_grad_norm: float = 1.0\n",
        "    gradient_accumulation_steps: int = GRAD_ACCUM\n",
        "    prompt_text: str = PROMPT\n",
        "    save_dir: str = SAVE_DIR\n",
        "    save_every: int = 5\n",
        "    early_stopping_patience: int = PATIENCE\n",
        "    early_stopping_min_delta: float = MIN_DELTA\n",
        "    resume_from: str = RESUME_FROM\n",
        "    num_workers: int = NUM_WORKERS\n",
        "    preload_ram: bool = PRELOAD_RAM\n",
        "    local_cache_dir: str = ''  # Not used — dataset already in RAM\n",
        "    streaming: bool = False    # Not streaming — loaded via shards\n",
        "    streaming_val_samples: int = 0\n",
        "    steps_per_epoch: int = 0\n",
        "    streaming_shuffle_buffer: int = 0\n",
    ])

    # ── Cell 11: Dataset + Collate ──
    md(["## 📊 Dataset"])
    train_src = Path("src/train_phase2.py").read_text(encoding="utf-8")
    train_lines = train_src.split("\n")

    ds_start = next(i for i, l in enumerate(train_lines) if "class Phase2Dataset" in l)
    trainer_start = next(i for i, l in enumerate(train_lines) if "class Phase2Trainer" in l)
    dataset_code = "\n".join(train_lines[ds_start:trainer_start - 4])
    code([dataset_code])

    # ── Cell 12: Trainer class ──
    md(["## 🏋️ Trainer"])
    cli_start = next(i for i, l in enumerate(train_lines) if "def parse_args" in l)
    trainer_code = "\n".join(train_lines[trainer_start:cli_start - 4])
    code([trainer_code])

    # ── Cell 13: Run ──
    md(["## 🚀 Run Training\n",
        "Dataset `ds` đã load ở trên sẽ được truyền trực tiếp — **không tải lại!**"])
    code([
        "config = Phase2Config()\n",
        "print(config)\n",
        "print()\n",
        "trainer = Phase2Trainer(config)\n",
        "trainer.train(preloaded_dataset=ds)  # Pass already-loaded dataset\n",
    ])

    # ── Cell 14: Download ──
    md(["## 💾 Download Best Checkpoint"])
    code([
        "from google.colab import files\n",
        'files.download(f"{SAVE_DIR}/projector_best.pt")\n',
    ])

    filename = "train_phase2_v094_vietspeech.ipynb"
    return cells, filename


def export_notebook():
    """Generate and save the notebook."""
    cells, filename = build_notebook()

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
            "gpuClass": "standard",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "cells": cells,
    }

    out_path = Path("notebooks") / filename
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"  OK: {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    print("Generating v0.9.4 VietSpeech notebook:")
    export_notebook()
