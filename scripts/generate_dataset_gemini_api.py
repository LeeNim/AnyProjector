# -*- coding: utf-8 -*-
"""
generate_dataset_gemini_api.py - Premium Gemini 3.5 Flash-powered Vietnamese dataset generator for Speech-MASSIVE dataset.
Uses dynamic few-shot matching, exponential backoff, and state-of-the-art prompt templates to guarantee gold-standard outputs.
"""

import os
import sys
import json
import time
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Load system instructions
SYSTEM_INSTRUCTION = """
Bạn là một trợ lý ảo tiếng Việt siêu cấp. Nhiệm vụ của bạn là nhận câu nói của người dùng (utt), kết hợp với ý định (intent) và kịch bản (scenario) để sinh ra câu trả lời theo đúng định dạng tool calling và phản hồi tự nhiên bằng tiếng Việt.

QUY TẮC ĐỊNH DẠNG ĐẦU RA (BẮT BUỘC):
1. Nếu câu nói cần gọi công cụ (tool call):
Định dạng đầu ra PHẢI là:
<tool_call>{"name": "tên_công_cụ", "arguments": {các_tham_số}}</tool_call> Phản hồi tự nhiên bằng tiếng Việt.

Ví dụ:
<tool_call>{"name": "alarm_set", "arguments": {"time": "06:00"}}</tool_call> Dạ, tôi đã đặt báo thức lúc 6 giờ sáng cho bạn rồi nhé.

2. Nếu câu nói chỉ là giao tiếp thông thường, chém gió, chào hỏi (ví dụ: intent là 'general_quirky' hoặc scenario là 'general' mà không cần hành động gì):
Định dạng đầu ra PHẢI là phản hồi tự nhiên bằng tiếng Việt trực tiếp, KHÔNG có thẻ <tool_call>.

Ví dụ:
Nghiên cứu không gian là một lĩnh vực vô tận và kỳ thú! Bạn đang quan tâm đến chủ đề nào cụ thể thế?

YÊU CẦU CHẤT LƯỢNG PHẢN HỒI:
- Phản hồi tự nhiên phải là tiếng Việt bản xứ, cực kỳ trôi chảy, ấm áp, lịch sự. Tránh dịch máy thô cứng.
- Sử dụng các từ đệm tự nhiên như "nhé", "nha", "ạ", "giúp bạn nhé", "đây ạ", "nhé bạn".
- Trích xuất tham số (arguments) chính xác từ câu nói. Không tự bịa ra thông tin.
"""

def select_few_shots(intent, scenario, completed_samples, k=5):
    """
    Select the top k most relevant few-shot examples based on intent and scenario.
    """
    matches = [s for s in completed_samples if s.get("intent") == intent]
    if len(matches) < k:
        scenario_matches = [s for s in completed_samples if s.get("scenario") == scenario and s not in matches]
        matches.extend(scenario_matches)
    if len(matches) < k:
        general_matches = [s for s in completed_samples if s not in matches]
        matches.extend(general_matches)
    return matches[:k]

def call_gemini_with_backoff(model, prompt, max_retries=5, initial_delay=2.0):
    """
    Calls the Gemini API with exponential backoff for rate limits.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            else:
                print(f"⚠️ Empty response from model, retrying...")
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "limit" in err_str.lower():
                print(f"⏳ Rate limit hit (429). Retrying in {delay:.1f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                delay *= 2.0
            else:
                print(f"❌ Gemini API Error: {e}")
                time.sleep(delay)
                delay *= 1.5
    return None

def main():
    parser = argparse.ArgumentParser(description="Augment Speech-MASSIVE dataset with premium Gemini responses.")
    parser.add_argument("--split", choices=["validation", "test"], default="validation", help="Split to generate.")
    parser.add_argument("--limit", type=int, default=10, help="Max samples to generate in this run.")
    parser.add_argument("--offset", type=int, default=550, help="Starting index in the split.")
    parser.add_argument("--model", type=str, default="models/gemini-3.5-flash", help="Gemini model name.")
    parser.add_argument("--output", type=str, default="", help="Output json file path.")
    
    args = parser.parse_args()
    
    # Load .env variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file!")
        sys.exit(1)
        
    genai.configure(api_key=api_key)
    
    # Initialize Gemini model
    print(f"🤖 Initializing model {args.model}...")
    try:
        model = genai.GenerativeModel(
            model_name=args.model,
            system_instruction=SYSTEM_INSTRUCTION
        )
    except Exception as e:
        print(f"❌ Failed to initialize Gemini model: {e}")
        sys.exit(1)
        
    # 2. Load completed gold-standard samples for few-shot learning
    completed_samples = []
    gold_files = [
        Path("dataset/speech_massive_lora_115_offset_0.json"),
        Path("dataset/speech_massive_lora_500_val.json"),
        Path("dataset/speech_massive_lora_50_offset_500.json")
    ]
    
    for gf in gold_files:
        if gf.exists():
            try:
                with open(gf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    completed_samples.extend(data)
                print(f"✅ Loaded {len(data)} gold samples from {gf.name}")
            except Exception as e:
                print(f"⚠️ Error loading {gf.name}: {e}")
                
    print(f"✨ Total gold-standard examples in registry: {len(completed_samples)}")
    
    # 3. Load input labels dataset
    labels_path = Path("data/agent_labels/speech_massive_labels.json")
    if not labels_path.exists():
        print(f"❌ Error: {labels_path} does not exist!")
        sys.exit(1)
        
    with open(labels_path, "r", encoding="utf-8") as f:
        all_labels = json.load(f)
        
    # Filter by split
    split_labels = [item for item in all_labels if item.get("split") == args.split]
    total_split_samples = len(split_labels)
    print(f"📦 Split '{args.split}' contains {total_split_samples} samples in total.")
    
    # Determine offset range
    start_idx = args.offset
    end_idx = min(start_idx + args.limit, total_split_samples)
    
    if start_idx >= total_split_samples:
        print(f"❌ Offset {start_idx} is out of bounds for split '{args.split}'!")
        sys.exit(1)
        
    print(f"🚀 Processing samples from offset {start_idx} to {end_idx - 1} (Total: {end_idx - start_idx} samples)")
    
    # 4. Initialize output file and loaded progress
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"dataset/speech_massive_lora_{args.split}_offset_{start_idx}.json")
        
    progress_data = []
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
            print(f"♻️ Resuming: Loaded {len(progress_data)} existing samples from {out_path.name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not parse existing output file, starting fresh: {e}")
            
    # Convert progress data to dict for fast lookup
    progress_dict = {item["id"]: item for item in progress_data}
    
    # 5. Generation loop
    count_success = 0
    count_skipped = 0
    consecutive_failures = 0
    
    for i in range(start_idx, end_idx):
        sample = split_labels[i]
        sample_id = sample.get("id")
        utt = sample.get("transcript")
        intent = sample.get("intent")
        scenario = sample.get("scenario")
        
        # Check if already processed in this session/file
        if sample_id in progress_dict:
            count_skipped += 1
            continue
            
        print(f"⏳ [{i - start_idx + 1}/{end_idx - start_idx}] Generating response for ID {sample_id} | Intent: {intent} | Utt: '{utt}'")
        
        # Select few-shot examples
        few_shots = select_few_shots(intent, scenario, completed_samples, k=5)
        
        # Parse original tool call suggestion
        orig_output = sample.get("output", {})
        orig_calls = orig_output.get("calls", [])
        suggested_call = None
        if orig_calls:
            suggested_call = {"name": orig_calls[0].get("name"), "arguments": orig_calls[0].get("args")}
            
        # Build prompt
        prompt_parts = []
        prompt_parts.append("Dưới đây là một số ví dụ mẫu chuẩn về cách phản hồi:")
        for idx_fs, fs in enumerate(few_shots):
            prompt_parts.append(f"\n[Ví dụ {idx_fs+1}]")
            prompt_parts.append(f"Yêu cầu: Utt='{fs['utt']}', Intent='{fs['intent']}', Scenario='{fs['scenario']}'")
            prompt_parts.append(f"Trả lời: {fs['response']}")
            
        prompt_parts.append("\n" + "="*40)
        prompt_parts.append("HÃY SINH CÂU TRẢ LỜI CHO YÊU CẦU DƯỚI ĐÂY:")
        prompt_parts.append(f"Yêu cầu: Utt='{utt}', Intent='{intent}', Scenario='{scenario}'")
        if suggested_call:
            prompt_parts.append(f"Gợi ý tool call gốc: {json.dumps(suggested_call, ensure_ascii=False)}")
        prompt_parts.append("Trả lời:")
        
        prompt = "\n".join(prompt_parts)
        
        # Call Gemini API
        response_str = call_gemini_with_backoff(model, prompt)
        
        if response_str:
            # Basic validation: ensure tag is closed if it is opened
            if "<tool_call>" in response_str and "</tool_call>" not in response_str:
                response_str += "</tool_call>"
                
            new_item = {
                "id": sample_id,
                "utt": utt,
                "intent": intent,
                "scenario": scenario,
                "response": response_str
            }
            progress_data.append(new_item)
            progress_dict[sample_id] = new_item
            count_success += 1
            consecutive_failures = 0
            
            # Auto-save checkpoint every single success to prevent data loss
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(progress_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ Error saving checkpoint: {e}")
        else:
            print(f"❌ Failed to generate response for ID {sample_id} after retries. Skipping.")
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("🚨 CRITICAL: Encountered 3 consecutive API failures! This indicates that the daily quota has been fully exhausted or there is a persistent network error. Exiting gracefully to save state.")
                break
            
        # Respect API Rate limits by pausing slightly between requests (increased to 3.5s to stay under Free Tier 15 RPM limit)
        time.sleep(3.5)
        
    print(f"\n🎉 Generation Batch Complete!")
    print(f"  Processed range: {start_idx} -> {end_idx - 1}")
    print(f"  Successfully generated: {count_success} samples")
    print(f"  Skipped (already exist): {count_skipped} samples")
    print(f"  Saved progress to: {out_path.name}")
    print("=" * 60)

if __name__ == "__main__":
    main()
