# -*- coding: utf-8 -*-
"""
generate_dataset_ollama.py - Uses Ollama's local API to run Gemma 4 E4B
for GPU-accelerated dataset generation via llama.cpp engine.

Ollama handles CUDA/GPU acceleration natively with the latest llama.cpp
that fully supports the Gemma 4 architecture.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4-e4b-local"


def ollama_generate(prompt, max_tokens=192):
    """Call Ollama's generate API and return the response text."""
    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": max_tokens,
            "repeat_penalty": 1.1,
            "stop": ["<end_of_turn>"],
            "num_ctx": 2048,
            "num_gpu": 99,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_API,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result.get("response", "").strip()


def make_gemma_prompt(utt, intent, scenario, tool_hint_str):
    """Build the Gemma 4 instruction prompt for a single sample."""
    prompt = f"""<start_of_turn>user
Bạn là trợ lý ảo tiếng Việt chuyên nghiệp, thông minh và lịch sự.
Nhiệm vụ của bạn là phản hồi câu nói của người dùng dưới dạng câu lệnh công cụ (tool call) kết hợp câu nói tự nhiên, hoặc câu nói tự nhiên trực tiếp (nếu không có công cụ phù hợp).

HƯỚNG DẪN ĐỊNH DẠNG:
- Nếu có Gợi ý Công cụ (Suggested Tool Call Hint):
  Bạn PHẢI sử dụng đúng định dạng thẻ XML chứa JSON như sau:
  <tool_call>{{"name": "tên_công_cụ", "arguments": {{các_tham_số}}}}</tool_call> Phản hồi tự nhiên bằng tiếng Việt lịch sự, thân thiện.
  Ví dụ: <tool_call>{{"name": "alarm_set", "arguments": {{"time": "06:00"}}}}</tool_call> Dạ, tôi đã đặt báo thức lúc 6 giờ sáng cho bạn rồi nhé.

- Nếu KHÔNG có Gợi ý Công cụ (General Quirky/chém gió/tán gẫu):
  Phản hồi tự nhiên bằng tiếng Việt lịch sự, thân thiện trực tiếp, TUYỆT ĐỐI KHÔNG dùng thẻ <tool_call>.
  Ví dụ: Chào bạn! Hôm nay tôi có thể giúp gì cho bạn ạ?

Yêu cầu chất lượng câu trả lời:
- Luôn sử dụng giọng văn tự nhiên của người bản xứ, lịch sự, trôi chảy. Tránh dịch thô cứng.
- Dùng các từ đệm tự nhiên cuối câu như: "ạ", "nhé", "nha", "giúp bạn nhé".

Dữ liệu mẫu cần xử lý:
- Câu nói (Utt): {utt}
- Ý định (Intent): {intent}
- Bối cảnh (Scenario): {scenario}
- Gợi ý Công cụ: {tool_hint_str}

Hãy đưa ra câu trả lời (Response) hoàn chỉnh, cực kỳ tự nhiên:<end_of_turn>
<start_of_turn>model
"""
    return prompt


def main():
    print("=" * 60)
    print("  Speech-MASSIVE Ollama Dataset Generator (Gemma 4 E4B)")
    print("=" * 60)

    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    labels_path = Path("data/agent_labels/speech_massive_labels.json")
    completed_output_file = dataset_dir / "speech_massive_lora_local_gguf_completed.json"

    # 1. Verify Ollama is running and model is available
    print(f"\n[1/4] Verifying Ollama API at {OLLAMA_API}...")
    try:
        test_req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(test_req, timeout=5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in tags.get("models", [])]
            print(f"   Available models: {models}")
            if not any(MODEL_NAME.split(":")[0] in m for m in models):
                print(f"❌ Model '{MODEL_NAME}' not found. Please run: ollama pull {MODEL_NAME}")
                return
        print(f"✅ Ollama is running with model '{MODEL_NAME}' available!")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is running (check system tray or run 'ollama serve')")
        return

    # 2. Quick dry run to verify generation works
    print(f"\n[2/4] Running dry-run test...")
    try:
        test_response = ollama_generate("Xin chào! Bạn là ai?", max_tokens=50)
        print(f"   Test response: {test_response[:100]}...")
        print("✅ Dry run successful!")
    except Exception as e:
        print(f"❌ Dry run failed: {e}")
        return

    # 3. Load completed sample IDs (from gold chunks and previous runs)
    completed_ids = set()
    gold_files = [
        dataset_dir / "speech_massive_lora_115_offset_0.json",
        dataset_dir / "speech_massive_lora_500_val.json",
        dataset_dir / "speech_massive_lora_50_offset_500.json",
        dataset_dir / "speech_massive_lora_validation_offset_550.json"
    ]

    for gf in gold_files:
        if gf.exists():
            try:
                with open(gf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        completed_ids.add(item.get("id"))
                print(f"  Loaded {len(data)} completed gold IDs from {gf.name}")
            except Exception as e:
                print(f"  ⚠️ Error reading {gf.name}: {e}")

    # Load previously generated samples for auto-resume
    existing_samples = []
    if completed_output_file.exists():
        try:
            with open(completed_output_file, "r", encoding="utf-8") as f:
                existing_samples = json.load(f)
                for item in existing_samples:
                    completed_ids.add(item.get("id"))
            print(f"  Loaded {len(existing_samples)} already generated samples. (Auto-Resume active)")
        except Exception as e:
            print(f"  ⚠️ Error reading {completed_output_file.name}: {e}")

    print(f"  ✨ Total already processed samples: {len(completed_ids)}")

    # 4. Load all raw labels & Filter unprocessed samples
    if not labels_path.exists():
        print(f"❌ Error: Raw labels file not found at {labels_path}")
        return

    with open(labels_path, "r", encoding="utf-8") as f:
        all_labels = json.load(f)

    unprocessed_samples = []
    for item in all_labels:
        sample_id = item.get("id")
        split = item.get("split")

        # Only process validation and test splits
        if split not in ["validation", "test"]:
            continue

        if sample_id in completed_ids:
            continue

        # Extract suggested tool call
        orig_output = item.get("output", {})
        orig_calls = orig_output.get("calls", [])
        suggested_call = None
        if orig_calls:
            suggested_call = {
                "name": orig_calls[0].get("name"),
                "arguments": orig_calls[0].get("args")
            }

        unprocessed_samples.append({
            "id": sample_id,
            "utt": item.get("transcript"),
            "intent": item.get("intent"),
            "scenario": item.get("scenario"),
            "suggested_tool_call": suggested_call
        })

    total_to_process = len(unprocessed_samples)
    print(f"\n[3/4] 🔍 Found {total_to_process} remaining unprocessed samples.")
    if total_to_process == 0:
        print("🎉 All samples have already been completed!")
        return

    # 5. Run inference loop
    print(f"\n[4/4] Starting Ollama dataset generation for {total_to_process} samples...")
    print("      Checkpoint will be saved automatically after every 50 samples.\n")

    start_time = time.time()
    checkpoint_counter = 0
    error_count = 0

    for idx, sample in enumerate(unprocessed_samples, 1):
        utt = sample["utt"]
        intent = sample["intent"]
        scenario = sample["scenario"]
        suggested = sample["suggested_tool_call"]

        tool_hint_str = json.dumps(suggested, ensure_ascii=False) if suggested else "Không có (General Quirky)"

        prompt = make_gemma_prompt(utt, intent, scenario, tool_hint_str)

        # Generate via Ollama API
        try:
            response_text = ollama_generate(prompt)

            # Clean up: remove trailing <end_of_turn> if present
            if "<end_of_turn>" in response_text:
                response_text = response_text.split("<end_of_turn>")[0].strip()

        except Exception as e:
            print(f"\n❌ Error generating sample ID {sample['id']}: {e}")
            error_count += 1
            if error_count > 10:
                print("⛔ Too many consecutive errors. Stopping.")
                break
            # Brief pause before retry
            time.sleep(2)
            continue

        # Post-processing: Ensure it has tool call if suggested
        if suggested and "<tool_call>" not in response_text:
            # Inject tool call if model failed to output it but hint was provided
            tool_call_xml = f'<tool_call>{json.dumps(suggested, ensure_ascii=False)}</tool_call> '
            response_text = tool_call_xml + response_text

        new_sample = {
            "id": sample["id"],
            "utt": utt,
            "intent": intent,
            "scenario": scenario,
            "response": response_text
        }

        existing_samples.append(new_sample)
        checkpoint_counter += 1
        error_count = 0  # Reset error counter on success

        # Print progress
        elapsed = time.time() - start_time
        avg_speed = elapsed / idx
        est_remaining = avg_speed * (total_to_process - idx)

        sys.stdout.write(
            f"\rProcessing: {idx}/{total_to_process} (ID: {sample['id']}) | "
            f"Speed: {avg_speed:.2f}s/sample | "
            f"ETA: {est_remaining/3600:.2f}h"
        )
        sys.stdout.flush()

        # Print first 3 samples for quality check
        if idx <= 3:
            print(f"\n   📝 Sample {idx} response: {response_text[:150]}...")

        # Save Checkpoint after every 50 samples
        if checkpoint_counter >= 50 or idx == total_to_process:
            with open(completed_output_file, "w", encoding="utf-8") as f:
                json.dump(existing_samples, f, ensure_ascii=False, indent=2)
            checkpoint_counter = 0
            print(f"\n💾 Checkpoint saved! ({len(existing_samples)} total samples processed)")

    print(f"\n\n🎉 All generation completed in {(time.time() - start_time)/3600:.2f} hours!")
    print(f"📁 Completed file: {completed_output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
