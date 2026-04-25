"""
model_loader.py - Tải và cache model từ Hugging Face.

Phase 0: Tải Audio Encoder & LLM Decoder về cache cục bộ.
Đọc config.json của model để trích xuất encoder_dim / llm_dim (chuẩn bị cho Phase 1).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Thông tin model đã tải về."""
    hf_id: str
    local_path: Path
    config: dict = field(default_factory=dict)
    hidden_size: int | None = None
    model_type: str | None = None


def validate_hf_id(hf_id: str) -> bool:
    """Kiểm tra xem Hugging Face ID có tồn tại không.

    Args:
        hf_id: Hugging Face model ID (VD: 'openai/whisper-base').

    Returns:
        True nếu model tồn tại trên HF Hub.
    """
    try:
        api = HfApi()
        api.model_info(hf_id)
        return True
    except Exception:
        return False


def download_model(hf_id: str, cache_dir: Path, progress_callback=None) -> Path:
    """Tải model từ Hugging Face về cache cục bộ.

    Args:
        hf_id: Hugging Face model ID.
        cache_dir: Thư mục cache.
        progress_callback: Hàm callback để báo tiến trình (optional).

    Returns:
        Đường dẫn tới thư mục model đã tải.
    """
    logger.info(f"Downloading model: {hf_id} -> {cache_dir}")

    # Tạo tên thư mục con từ HF ID (vd: openai/whisper-base -> openai--whisper-base)
    model_subdir = hf_id.replace("/", "--")
    local_dir = cache_dir / model_subdir

    if progress_callback:
        progress_callback(f"⏳ Đang tải {hf_id}...")

    try:
        snapshot_download(
            repo_id=hf_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        if progress_callback:
            progress_callback(f"✅ Đã tải xong {hf_id}")
    except Exception as e:
        logger.error(f"Failed to download {hf_id}: {e}")
        raise

    return local_dir


def read_model_config(model_path: Path) -> dict:
    """Đọc config.json từ thư mục model.

    Args:
        model_path: Đường dẫn thư mục model đã tải.

    Returns:
        dict chứa nội dung config.json.
    """
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_hidden_size(config: dict) -> int:
    """Trích xuất kích thước hidden (encoder_dim hoặc llm_dim) từ config.

    Hỗ trợ nhiều kiến trúc model khác nhau:
    - Whisper: d_model (encoder) 
    - LLM (Qwen, Llama, etc.): hidden_size

    Args:
        config: dict từ config.json.

    Returns:
        Kích thước hidden dimension.

    Raises:
        KeyError: Nếu không tìm thấy key phù hợp.
    """
    # Thứ tự ưu tiên tìm kiếm
    candidates = [
        "d_model",       # Whisper encoder
        "hidden_size",   # Qwen, Llama, GPT-2, BERT, etc.
        "n_embd",        # GPT-2 variant
        "dim",           # Some custom models
    ]

    # Với Whisper, d_model nằm trong config gốc (dùng cho encoder)
    # Một số model có encoder config nested
    search_dicts = [config]
    if "encoder" in config:
        search_dicts.append(config["encoder"])

    for d in search_dicts:
        for key in candidates:
            if key in d:
                return int(d[key])

    raise KeyError(
        f"Cannot find hidden size in config. "
        f"Tried keys: {candidates}. "
        f"Available keys: {list(config.keys())}"
    )


def load_and_inspect_model(hf_id: str, cache_dir: Path, progress_callback=None) -> ModelInfo:
    """Tải model và đọc thông tin cấu hình.

    Workflow hoàn chỉnh cho Phase 0:
    1. Validate HF ID
    2. Download model
    3. Read config.json
    4. Extract hidden_size

    Args:
        hf_id: Hugging Face model ID.
        cache_dir: Thư mục cache.
        progress_callback: Hàm callback tiến trình.

    Returns:
        ModelInfo chứa thông tin model.
    """
    # 1. Validate
    if progress_callback:
        progress_callback(f"🔍 Kiểm tra {hf_id} trên HuggingFace Hub...")

    if not validate_hf_id(hf_id):
        raise ValueError(f"Model '{hf_id}' không tồn tại trên HuggingFace Hub.")

    # 2. Download
    local_path = download_model(hf_id, cache_dir, progress_callback)

    # 3. Read config
    if progress_callback:
        progress_callback(f"📖 Đọc config.json...")

    config = read_model_config(local_path)

    # 4. Extract hidden size
    hidden_size = extract_hidden_size(config)
    model_type = config.get("model_type", "unknown")

    if progress_callback:
        progress_callback(
            f"✅ {hf_id}\n"
            f"   Model type: {model_type}\n"
            f"   Hidden size: {hidden_size}"
        )

    return ModelInfo(
        hf_id=hf_id,
        local_path=local_path,
        config=config,
        hidden_size=hidden_size,
        model_type=model_type,
    )
