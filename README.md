# AnyProjector

**Voice Agent End-to-End** với giao diện tùy biến cho phép ghép nối trực tiếp Latent Space của **bất kỳ Audio Encoder** nào vào **bất kỳ LLM** nào.

## Tính năng chính

- 🔌 **Dynamic Architecture**: Tự động co giãn Projector theo kích thước encoder/LLM
- 🎤 **VAD (Voice Activity Detection)**: Phát hiện ngắt lời / kết thúc câu nói
- 🧠 **Agentic Tool Calling**: Phân tích ý định và gọi hàm từ giọng nói
- ⚡ **Streaming Inference**: Xử lý luồng âm thanh thời gian thực với Barge-in

## Cài đặt

```bash
# Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
python -m src.app
```

## Kiến trúc

```
AnyProjector/
├── src/
│   ├── app.py              # Gradio UI (Phase 0)
│   ├── model_loader.py     # Tải & cache model từ HuggingFace
│   └── config.py           # Quản lý cấu hình
├── config/
│   └── default_config.yaml # Cấu hình mặc định
├── requirements.txt
└── README.md
```

## Phần cứng khuyến nghị

- **RAM**: 32GB
- **GPU**: RTX 5060 (8GB VRAM)

## License

MIT
