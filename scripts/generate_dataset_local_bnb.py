# -*- coding: utf-8 -*-
"""
generate_dataset_local_bnb.py - Uses unsloth/gemma-4-E4B-it-unsloth-bnb-4bit
via HuggingFace Transformers + bitsandbytes for GPU-accelerated dataset generation.

This approach bypasses llama-cpp-python entirely, leveraging PyTorch's native CUDA
support which is fully compatible with the RTX 5060 (Blackwell).
"""

import json
import os
import sys
import time
import gc
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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
    print("  Speech-MASSIVE Local BNB-4bit Dataset Generator (Gemma 4)")
    print("=" * 60)

    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    labels_path = Path("data/agent_labels/speech_massive_labels.json")
    completed_output_file = dataset_dir / "speech_massive_lora_local_gguf_completed.json"

    # 1. Load model and tokenizer
    repo_id = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
    print(f"\n[1/4] Loading model and tokenizer from: {repo_id}")
    print("       (Attempting full GPU load, fallback to CPU offload)")

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)

        # Strategy 1: Force everything onto GPU (4-bit model ~4.5GB should fit in 8GB)
        try:
            print("   Trying: Full GPU load (device_map={'': 0})...")
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                device_map={"": 0},
            )
            print(f"✅ Model loaded fully on GPU!")
        except Exception as gpu_err:
            print(f"   ⚠️ Full GPU failed: {gpu_err}")
            print("   Trying: CPU offload with patched quantization config...")

            # Strategy 2: Patch the baked-in quantization config for CPU offloading
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(repo_id)
            if hasattr(config, 'quantization_config'):
                config.quantization_config['llm_int8_enable_fp32_cpu_offload'] = True

            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                config=config,
                device_map="auto",
                max_memory={0: "6500MiB", "cpu": "16GiB"},
            )
            print(f"✅ Model loaded with GPU + CPU offload!")

        print(f"   Device map: {getattr(model, 'hf_device_map', 'all on cuda:0')}")
        print(f"   GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Load completed sample IDs (from gold chunks and previous runs)
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

    # 3. Load all raw labels & Filter unprocessed samples
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

    # 4. Run inference loop
    print(f"\n[4/4] Starting local dataset generation for {total_to_process} samples...")
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

        # Tokenize and generate
        try:
            # With device_map="auto", inputs go to the first device (cuda:0)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=192,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                )

            # Decode only the new tokens (skip the prompt)
            response_ids = outputs[0][input_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

            # Clean up: remove trailing <end_of_turn> if present
            if "<end_of_turn>" in response_text:
                response_text = response_text.split("<end_of_turn>")[0].strip()

        except Exception as e:
            print(f"\n❌ Error generating sample ID {sample['id']}: {e}")
            error_count += 1
            if error_count > 10:
                print("⛔ Too many consecutive errors. Stopping.")
                break
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

        # Save Checkpoint after every 50 samples
        if checkpoint_counter >= 50 or idx == total_to_process:
            with open(completed_output_file, "w", encoding="utf-8") as f:
                json.dump(existing_samples, f, ensure_ascii=False, indent=2)
            checkpoint_counter = 0
            print(f"\n💾 Checkpoint saved! ({len(existing_samples)} total samples processed)")

        # Periodically clear CUDA cache to prevent OOM
        if idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n\n🎉 All generation completed in {(time.time() - start_time)/3600:.2f} hours!")
    print(f"📁 Completed file: {completed_output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
