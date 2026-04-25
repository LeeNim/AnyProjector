"""
config.py - Quản lý cấu hình cho AnyProjector.

Đọc file YAML config và cung cấp các giá trị mặc định.
"""

import os
from pathlib import Path

import yaml


# Đường dẫn gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"


def load_config(config_path: str | None = None) -> dict:
    """Đọc và trả về config dict từ file YAML.

    Args:
        config_path: Đường dẫn tới file config. Nếu None, dùng default.

    Returns:
        dict chứa toàn bộ cấu hình.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_default_encoder_id(config: dict) -> str:
    """Lấy Hugging Face ID mặc định cho Audio Encoder."""
    return config["models"]["audio_encoder"]["default_id"]


def get_default_llm_id(config: dict) -> str:
    """Lấy Hugging Face ID mặc định cho LLM Decoder."""
    return config["models"]["llm_decoder"]["default_id"]


def get_cache_dir(config: dict) -> Path:
    """Lấy đường dẫn cache cho model đã tải."""
    cache_dir = Path(config["cache"]["models_dir"])
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
