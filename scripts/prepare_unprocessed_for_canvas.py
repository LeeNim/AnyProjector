# -*- coding: utf-8 -*-
"""
prepare_unprocessed_for_canvas.py - Extracts remaining unprocessed samples, formats them cleanly, 
and provides a ready-to-copy file for the user to upload to Gemini Web/Canvas.
"""

import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=" * 60)
    print("  Speech-MASSIVE Unprocessed Dataset Extractor")
    print("=" * 60)
    
    dataset_dir = Path("dataset")
    labels_path = Path("data/agent_labels/speech_massive_labels.json")
    
    # 1. Load already completed sample IDs
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
                print(f"Loaded {len(data)} completed IDs from {gf.name}")
            except Exception as e:
                print(f"⚠️ Error reading {gf.name}: {e}")
                
    print(f"✨ Total already processed samples: {len(completed_ids)}")
    
    # 2. Load all raw labels
    if not labels_path.exists():
        print(f"❌ Error: Raw labels file not found at {labels_path}")
        return
        
    with open(labels_path, "r", encoding="utf-8") as f:
        all_labels = json.load(f)
        
    print(f"Total raw labels loaded: {len(all_labels)}")
    
    # 3. Filter unprocessed samples (Only for validation and test splits)
    unprocessed_samples = []
    
    for item in all_labels:
        sample_id = item.get("id")
        split = item.get("split")
        
        # Skip if already completed
        if sample_id in completed_ids:
            continue
            
        # We only care about validation and test splits
        if split not in ["validation", "test"]:
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
            "split": split,
            "utt": item.get("transcript"),
            "intent": item.get("intent"),
            "scenario": item.get("scenario"),
            "suggested_tool_call": suggested_call
        })
        
    print(f"🔍 Found {len(unprocessed_samples)} remaining unprocessed samples.")
    
    # 4. Save as a clean JSON file
    output_json = dataset_dir / "unprocessed_for_canvas.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(unprocessed_samples, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved unprocessed JSON to: {output_json}")
    
    # 5. Save as a simple CSV file for easy spreadsheet viewing
    import csv
    output_csv = dataset_dir / "unprocessed_for_canvas.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Split", "Transcript (Utt)", "Intent", "Scenario", "Suggested Tool Call Hint"])
        for s in unprocessed_samples:
            tool_hint = json.dumps(s["suggested_tool_call"], ensure_ascii=False) if s["suggested_tool_call"] else ""
            writer.writerow([s["id"], s["split"], s["utt"], s["intent"], s["scenario"], tool_hint])
            
    print(f"💾 Saved unprocessed CSV to: {output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    main()
