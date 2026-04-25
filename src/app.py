"""
app.py - Giao diện Web cho AnyProjector.

Phase 0: Tải model từ HuggingFace Hub.
Phase 1: Xây dựng kiến trúc mạng (Encoder frozen + LLM 4-bit + Projector trainable).
"""

import logging
import sys
from pathlib import Path

import gradio as gr

from src.config import load_config, get_default_encoder_id, get_default_llm_id, get_cache_dir
from src.model_loader import load_and_inspect_model, ModelInfo
from src.system import AnyProjectorSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- State ----------
loaded_models: dict[str, ModelInfo] = {}
system: AnyProjectorSystem | None = None


# ---------- Core Logic ----------
def initialize_system(
    encoder_id: str,
    llm_id: str,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[str, str, str]:
    """Khởi tạo hệ thống: validate & tải cả 2 model.

    Args:
        encoder_id: HuggingFace ID cho Audio Encoder.
        llm_id: HuggingFace ID cho LLM Decoder.
        progress: Gradio progress tracker.

    Returns:
        Tuple (status_text, encoder_info, llm_info).
    """
    global loaded_models

    config = load_config()
    cache_dir = get_cache_dir(config)

    encoder_id = encoder_id.strip()
    llm_id = llm_id.strip()

    if not encoder_id or not llm_id:
        return (
            "❌ Lỗi: Vui lòng nhập cả 2 Hugging Face ID.",
            "",
            "",
        )

    results = []

    # --- Tải Audio Encoder ---
    try:
        progress(0.0, desc="Đang tải Audio Encoder...")
        encoder_info = load_and_inspect_model(
            hf_id=encoder_id,
            cache_dir=cache_dir,
            progress_callback=lambda msg: logger.info(msg),
        )
        loaded_models["encoder"] = encoder_info
        encoder_summary = (
            f"✅ **{encoder_id}**\n"
            f"- Model type: `{encoder_info.model_type}`\n"
            f"- Hidden size (encoder_dim): **{encoder_info.hidden_size}**\n"
            f"- Local path: `{encoder_info.local_path}`"
        )
        results.append(("encoder", True))
    except Exception as e:
        encoder_summary = f"❌ **Lỗi tải Audio Encoder:** {e}"
        results.append(("encoder", False))
        logger.error(f"Encoder load failed: {e}")

    # --- Tải LLM Decoder ---
    try:
        progress(0.5, desc="Đang tải LLM Decoder...")
        llm_info = load_and_inspect_model(
            hf_id=llm_id,
            cache_dir=cache_dir,
            progress_callback=lambda msg: logger.info(msg),
        )
        loaded_models["llm"] = llm_info
        llm_summary = (
            f"✅ **{llm_id}**\n"
            f"- Model type: `{llm_info.model_type}`\n"
            f"- Hidden size (llm_dim): **{llm_info.hidden_size}**\n"
            f"- Local path: `{llm_info.local_path}`"
        )
        results.append(("llm", True))
    except Exception as e:
        llm_summary = f"❌ **Lỗi tải LLM Decoder:** {e}"
        results.append(("llm", False))
        logger.error(f"LLM load failed: {e}")

    # --- Tổng kết ---
    all_ok = all(ok for _, ok in results)
    if all_ok:
        enc_dim = loaded_models["encoder"].hidden_size
        llm_dim = loaded_models["llm"].hidden_size
        status = (
            f"## ✅ Khởi tạo thành công!\n\n"
            f"**Projector sẽ được xây dựng với:**\n"
            f"- `encoder_dim` = **{enc_dim}**\n"
            f"- `llm_dim` = **{llm_dim}**\n"
            f"- Conv1d: ({enc_dim} → {enc_dim}, stride=2)\n"
            f"- Semantic Route: Linear({enc_dim} → {llm_dim})\n"
            f"- VAD Route: Linear({enc_dim} → 1)\n\n"
            f"*Sẵn sàng! Nhấn **Xây dựng Kiến trúc** bên dưới.*"
        )
    else:
        status = "## ⚠️ Khởi tạo chưa hoàn tất\nVui lòng kiểm tra lỗi bên dưới."

    progress(1.0, desc="Hoàn tất!")
    return status, encoder_summary, llm_summary


def build_architecture(progress=gr.Progress(track_tqdm=True)) -> str:
    """Phase 1: Xây dựng kiến trúc mạng.

    Load Encoder (frozen) + LLM (4-bit, frozen) + tạo Projector (trainable).

    Returns:
        Markdown string mô tả kiến trúc đã xây dựng.
    """
    global system

    if "encoder" not in loaded_models or "llm" not in loaded_models:
        return "❌ Chưa tải model. Vui lòng nhấn **Khởi tạo Hệ thống** trước."

    try:
        progress(0.0, desc="Đang xây dựng kiến trúc...")

        system = AnyProjectorSystem(
            encoder_info=loaded_models["encoder"],
            llm_info=loaded_models["llm"],
        )

        progress(0.1, desc="Đang tải Encoder lên GPU...")
        info = system.build(
            progress_callback=lambda msg: logger.info(msg),
        )

        progress(1.0, desc="Hoàn tất!")

        return (
            f"## ✅ Kiến trúc đã sẵn sàng!\n\n"
            f"| Thành phần | Giá trị |\n"
            f"|---|---|\n"
            f"| Device | `{info.device}` |\n"
            f"| Encoder | `{info.encoder_type}` (encoder_dim={info.encoder_dim}) |\n"
            f"| LLM | `{info.llm_type}` (llm_dim={info.llm_dim}) |\n"
            f"| LLM Quantization | {info.llm_quantization} |\n"
            f"| **Projector Params** | **{info.projector_params:,}** (trainable) |\n\n"
            f"```\n{system.projector}\n```\n\n"
            f"*Sẵn sàng cho Phase 2: Huấn luyện đa nhiệm.*"
        )
    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=True)
        return f"❌ **Lỗi xây dựng kiến trúc:**\n```\n{e}\n```"


# ---------- UI ----------
def create_ui() -> gr.Blocks:
    """Xây dựng giao diện Gradio."""
    config = load_config()
    default_encoder = get_default_encoder_id(config)
    default_llm = get_default_llm_id(config)

    with gr.Blocks(
        title="AnyProjector - Cấu hình Hệ thống",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    ) as app:

        # --- Header ---
        gr.Markdown(
            """
            # 🎙️ AnyProjector
            ### Dynamic Voice Agent — Ghép nối bất kỳ Audio Encoder ↔ LLM
            
            Nhập Hugging Face ID cho 2 model rồi nhấn **Khởi tạo** để tải về cache cục bộ.
            """
        )

        with gr.Row():
            # --- Cột trái: Input ---
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Cấu hình Model")

                encoder_input = gr.Textbox(
                    label="🔊 Audio Encoder (HuggingFace ID)",
                    placeholder="VD: openai/whisper-base",
                    value=default_encoder,
                    info="Model STT để mã hóa âm thanh thành vector.",
                )
                llm_input = gr.Textbox(
                    label="🧠 LLM Decoder (HuggingFace ID)",
                    placeholder="VD: Qwen/Qwen2.5-1.5B-Instruct",
                    value=default_llm,
                    info="Model ngôn ngữ để xử lý ngữ nghĩa và sinh text.",
                )

                init_btn = gr.Button(
                    value="🚀 Khởi tạo Hệ thống",
                    variant="primary",
                    size="lg",
                )

            # --- Cột phải: Status ---
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Trạng thái")
                status_output = gr.Markdown(
                    value="*Chưa khởi tạo. Nhấn nút để bắt đầu.*"
                )

        # --- Chi tiết model ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔊 Audio Encoder")
                encoder_output = gr.Markdown(value="")
            with gr.Column():
                gr.Markdown("### 🧠 LLM Decoder")
                llm_output = gr.Markdown(value="")

        # --- Phase 1: Build Architecture ---
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🏗️ Phase 1: Xây dựng Kiến trúc")
                build_btn = gr.Button(
                    value="🔗 Xây dựng Kiến trúc",
                    variant="secondary",
                    size="lg",
                )
            with gr.Column(scale=2):
                build_output = gr.Markdown(
                    value="*Tải model trước, sau đó nhấn nút xây dựng.*"
                )

        # --- Events ---
        init_btn.click(
            fn=initialize_system,
            inputs=[encoder_input, llm_input],
            outputs=[status_output, encoder_output, llm_output],
        )
        build_btn.click(
            fn=build_architecture,
            inputs=[],
            outputs=[build_output],
        )

    return app


# ---------- Entry Point ----------
if __name__ == "__main__":
    config = load_config()
    server_cfg = config.get("server", {})

    app = create_ui()
    app.launch(
        server_name=server_cfg.get("host", "0.0.0.0"),
        server_port=server_cfg.get("port", 7860),
        share=server_cfg.get("share", False),
    )
