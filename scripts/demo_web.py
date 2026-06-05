"""
demo_web.py - AnyProjector Demo (FastAPI + Vanilla HTML)

Lightweight alternative to Gradio. Uses native browser audio recording
(Web Audio API + MediaRecorder) → WAV → FastAPI → Engine → JSON response.

Usage:
    python scripts/demo_web.py
    python scripts/demo_web.py --checkpoint path/to/best.pt --port 8000
"""

import sys
import os

if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

import argparse
import time
import csv
import io
import tempfile
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Import engine components from demo.py (no Gradio UI created on import)
from scripts.demo import DemoEngine, SAMPLE_RATE, PRESET_PROMPTS

# ── FastAPI App ───────────────────────────────────────────────────────

app = FastAPI(title="AnyProjector Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine: DemoEngine | None = None
HTML_PATH = Path(__file__).parent / "demo_web.html"
LOG_PATH = Path(__file__).parent / "demo_log.csv"

# In-memory transcription log
transcription_log: list[dict] = []

CSV_COLUMNS = [
    "id", "timestamp", "mode", "prompt", "temperature", "max_tokens",
    "proj_text", "proj_total_ms", "proj_encode_ms", "proj_project_ms", "proj_generate_ms",
    "cascade_whisper_text", "cascade_llm_text", "cascade_total_ms", "cascade_whisper_ms", "cascade_llm_ms",
]


def _append_log(entry: dict):
    """Append entry to in-memory log and save CSV."""
    entry["id"] = len(transcription_log) + 1
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    transcription_log.append(entry)
    _save_csv()


def _save_csv():
    """Persist log to CSV file."""
    try:
        with open(LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(transcription_log)
        logger.info(f"  Log saved: {LOG_PATH} ({len(transcription_log)} rows)")
    except Exception as e:
        logger.warning(f"  Failed to save CSV: {e}")


def _save_upload(upload: UploadFile) -> str:
    """Save uploaded audio to temp file, return path."""
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(upload.file.read())
        return f.name


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/api/info")
async def info():
    return {
        "encoder": engine.encoder_backend,
        "whisper_size": engine.whisper_size,
        "llm": engine.llm_id,
        "projector_loaded": engine.projector is not None,
        "vad_loaded": engine.vad_model is not None,
        "device": engine.device,
        "presets": PRESET_PROMPTS,
    }


@app.post("/api/transcribe")
async def api_transcribe(
    audio: UploadFile = File(...),
    mode: str = Form("both"),
    prompt: str = Form("Transcribe the following audio in Vietnamese:"),
    temperature: float = Form(0.1),
    max_tokens: int = Form(100),
    use_tools: bool = Form(False),
    enabled_tools: str = Form("calculator,get_time,translate,search"),
):
    """Transcribe audio. mode: 'projector', 'standalone', or 'both'."""
    tmp = _save_upload(audio)
    try:
        result = {}
        if mode in ("projector", "both"):
            if use_tools:
                tools_list = [t.strip() for t in enabled_tools.split(",") if t.strip()]
                tool_prompt = engine.tool_registry.get_tool_prompt(tools_list)
                full_prompt = f"{tool_prompt}\n\n{prompt}"
                proj_result = engine.transcribe_projector(
                    tmp, prompt=full_prompt, temperature=temperature, max_tokens=int(max_tokens),
                )
                tool_call, tool_result = engine.tool_registry.detect_and_execute(proj_result["text"])
                proj_result["tool_call"] = tool_call
                proj_result["tool_result"] = tool_result
                result["projector"] = proj_result
            else:
                result["projector"] = engine.transcribe_projector(
                    tmp, prompt=prompt, temperature=temperature, max_tokens=int(max_tokens),
                )
        if mode in ("standalone", "both"):
            result["standalone"] = engine.transcribe_standalone(
                tmp, prompt=prompt, temperature=temperature, max_tokens=int(max_tokens),
            )

        # Log the result
        log_entry = {
            "mode": mode, "prompt": prompt,
            "temperature": temperature, "max_tokens": max_tokens,
        }
        if p := result.get("projector"):
            bd = p.get("breakdown", {})
            log_entry.update({
                "proj_text": p.get("text", ""),
                "proj_total_ms": p.get("total_ms", 0),
                "proj_encode_ms": bd.get("encode_ms", 0),
                "proj_project_ms": bd.get("project_ms", 0),
                "proj_generate_ms": bd.get("generate_ms", 0),
            })
        if s := result.get("standalone"):
            bd = s.get("breakdown", {})
            log_entry.update({
                "cascade_whisper_text": s.get("whisper_text", ""),
                "cascade_llm_text": s.get("text", ""),
                "cascade_total_ms": s.get("total_ms", 0),
                "cascade_whisper_ms": bd.get("whisper_ms", 0),
                "cascade_llm_ms": bd.get("llm_ms", 0),
            })
        _append_log(log_entry)
        result["log_count"] = len(transcription_log)

        return JSONResponse(result)
    finally:
        os.unlink(tmp)


@app.get("/api/log")
async def api_log():
    """Return all log entries."""
    return JSONResponse({"count": len(transcription_log), "entries": transcription_log})


@app.get("/api/log/csv")
async def api_log_csv():
    """Download log as CSV file."""
    from fastapi.responses import StreamingResponse

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(transcription_log)
    buf.seek(0)

    filename = f"demo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/log/clear")
async def api_log_clear():
    """Clear all log entries."""
    transcription_log.clear()
    return JSONResponse({"status": "cleared"})


@app.post("/api/tools")
async def api_tools(
    audio: UploadFile = File(...),
    prompt: str = Form(""),
    enabled_tools: str = Form("calculator,get_time,translate,search"),
    temperature: float = Form(0.3),
):
    """Transcribe with tool calling."""
    tmp = _save_upload(audio)
    try:
        tools_list = [t.strip() for t in enabled_tools.split(",") if t.strip()]
        result = engine.transcribe_with_tools(
            tmp, enabled_tools=tools_list, user_prompt=prompt, temperature=temperature,
        )
        return JSONResponse(result)
    finally:
        os.unlink(tmp)


@app.post("/api/vad")
async def api_vad(
    audio: UploadFile = File(...),
    use_projector: bool = Form(False),
):
    """VAD segmentation + transcription."""
    import soundfile as sf

    tmp = _save_upload(audio)
    try:
        waveform, sr = sf.read(tmp, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

        # VAD
        segments_ts = []
        if engine.vad_model is not None:
            wav_t = torch.from_numpy(waveform).float()
            segments_ts = engine.get_speech_timestamps(
                wav_t, engine.vad_model, sampling_rate=SAMPLE_RATE,
                threshold=0.5, min_speech_duration_ms=300, min_silence_duration_ms=300,
            )
        if not segments_ts:
            segments_ts = [{"start": 0, "end": len(waveform)}]

        results = []
        total_ms = 0
        for ts in segments_ts:
            seg = waveform[ts["start"]:ts["end"]]
            if len(seg) < SAMPLE_RATE * 0.2:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, seg, SAMPLE_RATE)
                seg_tmp = f.name
            try:
                r = engine.transcribe_projector(seg_tmp) if use_projector else engine.transcribe_standalone(seg_tmp)
                ms = r["total_ms"]
                total_ms += ms
                results.append({
                    "start": round(ts["start"] / SAMPLE_RATE, 2),
                    "end": round(ts["end"] / SAMPLE_RATE, 2),
                    "text": r["text"],
                    "ms": round(ms),
                })
            finally:
                os.unlink(seg_tmp)

        return JSONResponse({"segments": results, "total_ms": round(total_ms), "count": len(results)})
    finally:
        os.unlink(tmp)


# ── Entry Point ───────────────────────────────────────────────────────

def main():
    global engine

    parser = argparse.ArgumentParser(description="AnyProjector Demo (Web)")
    parser.add_argument("--checkpoint", default="projectorTrained/final_011.pt")
    parser.add_argument("--lora", default="lora/best011/best", help="Path to LoRA adapter directory")
    parser.add_argument("--encoder-ckpt", default="", help="Fine-tuned encoder weights (empty if encoder frozen)")
    parser.add_argument("--whisper-size", default="medium")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--encoder", default="faster-whisper", choices=["faster-whisper", "hf-whisper"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("=" * 50)
    print("  🎙️ AnyProjector Demo (FastAPI)")
    print("=" * 50)

    engine = DemoEngine(
        checkpoint_path=args.checkpoint,
        whisper_size=args.whisper_size,
        llm_id=args.llm,
        encoder_backend=args.encoder,
        lora_path=args.lora,
        encoder_ckpt=args.encoder_ckpt,
    )

    print(f"\n  → http://localhost:{args.port}\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
