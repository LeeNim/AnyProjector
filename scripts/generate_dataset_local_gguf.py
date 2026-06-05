# -*- coding: utf-8 -*-
"""
generate_dataset_local_gguf.py - Downloads Gemma 4 E4B GGUF model and runs local GPU acceleration
to process the remaining 4,432 samples of the Speech-MASSIVE dataset with strict format validation.
"""

import json
import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Windows CUDA DLL injection helper
# Dynamically add PyTorch CUDA DLLs directory to DLL search path to prevent "llama.dll not found" errors on Windows
if sys.platform == 'win32':
    try:
        import os
        torch_lib_path = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
        if torch_lib_path.exists():
            os.add_dll_directory(str(torch_lib_path.resolve()))
            print(f"✨ PyTorch CUDA DLLs injected dynamically from virtual environment!")
    except Exception as dll_err:
        pass

# Ensure we import llama-cpp safely
try:
    from llama_cpp import Llama
except ImportError:
    print("❌ Error: llama-cpp-python is not installed. Please install it first.")
    sys.exit(1)

def main():
    print("=" * 60)
    print("  Speech-MASSIVE Local LLM Dataset Generator (Gemma 4)")
    print("=" * 60)

    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    labels_path = Path("data/agent_labels/speech_massive_labels.json")
    completed_output_file = dataset_dir / "speech_massive_lora_local_gguf_completed.json"

    # 1. Download Gemma 4 GGUF Model using huggingface_hub
    repo_id = "unsloth/gemma-4-E4B-it-GGUF"
    model_filename = "gemma-4-E4B-it-Q4_K_M.gguf"
    
    print(f"\n[1/4] Checking and downloading model from Hugging Face: {repo_id}/{model_filename}...")
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            resume_download=True
        )
        print(f"✅ Model downloaded successfully! Path: {model_path}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return

    # 2. Initialize Llama-CPP with CUDA GPU acceleration
    print(f"\n[2/4] Initializing Llama model with GPU acceleration...")
    try:
        # n_gpu_layers=-1 offloads all layers to GPU (RTX 5060)
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=-1,
            verbose=False  # Keep logs clean
        )
        print("✅ Llama model loaded successfully with GPU layers offloaded!")
    except Exception as e:
        print(f"⚠️ GPU Initialization failed ({e}). Falling back to CPU...")
        try:
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=8,
                n_gpu_layers=0,
                verbose=False
            )
            print("✅ Llama model loaded successfully on CPU.")
        except Exception as cpu_e:
            print(f"❌ CPU Initialization failed: {cpu_e}")
            return

    # 3. Load completed sample IDs (from gold chunks and previous GGUF runs)
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
                print(f"Loaded {len(data)} completed gold IDs from {gf.name}")
            except Exception as e:
                print(f"⚠️ Error reading {gf.name}: {e}")

    # Load previously generated GGUF samples for auto-resume
    existing_samples = []
    if completed_output_file.exists():
        try:
            with open(completed_output_file, "r", encoding="utf-8") as f:
                existing_samples = json.load(f)
                for item in existing_samples:
                    completed_ids.add(item.get("id"))
            print(f"Loaded {len(existing_samples)} already generated local GGUF samples. (Auto-Resume active)")
        except Exception as e:
            print(f"⚠️ Error reading {completed_output_file.name}: {e}")

    print(f"✨ Total already processed samples: {len(completed_ids)}")

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
    print(f"🔍 Found {total_to_process} remaining unprocessed samples.")
    if total_to_process == 0:
        print("🎉 All samples have already been completed!")
        return

    # 5. Run inference loop
    print(f"\n[4/4] Starting local dataset generation for {total_to_process} samples...")
    print("Checkpoint will be saved automatically after every 50 samples.")
    
    start_time = time.time()
    
    # Gemma 4 instruction template helper
    def make_gemma_prompt(utt, intent, scenario, tool_hint_str):
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

    checkpoint_counter = 0
    
    for idx, sample in enumerate(unprocessed_samples, 1):
        utt = sample["utt"]
        intent = sample["intent"]
        scenario = sample["scenario"]
        suggested = sample["suggested_tool_call"]
        
        tool_hint_str = json.dumps(suggested, ensure_ascii=False) if suggested else "Không có (General Quirky)"
        
        prompt = make_gemma_prompt(utt, intent, scenario, tool_hint_str)
        
        # Generation with greedy decoding for consistent formatting
        try:
            output = llm(
                prompt,
                max_tokens=256,
                temperature=0.1,
                top_p=0.9,
                stop=["<end_of_turn>", "user", " trợ lý ảo"],
                echo=False
            )
            response_text = output["choices"][0]["text"].strip()
        except Exception as e:
            print(f"\n❌ Error generating sample ID {sample['id']}: {e}")
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

    print(f"\n\n🎉 All generation completed in {(time.time() - start_time)/3600:.2f} hours!")
    print(f"📁 Completed GGUF file: {completed_output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
