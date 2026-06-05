# -*- coding: utf-8 -*-
"""
compile_final_dataset.py - Consolidates all dataset chunks (train, validation, test) into a single unified JSON file for LoRA training.
"""

import json
from pathlib import Path

def main():
    print("=" * 60)
    print("  Speech-MASSIVE LoRA Dataset Compiler")
    print("=" * 60)
    
    dataset_dir = Path("dataset")
    output_path = dataset_dir / "speech_massive_lora_final.json"
    
    # 1. Base gold chunks we manually created
    chunks = [
        dataset_dir / "speech_massive_lora_115_offset_0.json",
        dataset_dir / "speech_massive_lora_500_val.json",
        dataset_dir / "speech_massive_lora_50_offset_500.json",
        dataset_dir / "speech_massive_lora_validation_offset_550.json"
    ]
    
    # 2. Automatically find all other generated chunks
    # Look for any files starting with speech_massive_lora_ and ending with .json
    for f in dataset_dir.glob("speech_massive_lora_*.json"):
        if f.name != "speech_massive_lora_final.json" and f not in chunks:
            chunks.append(f)
            
    print(f"Found {len(chunks)} chunk files to compile:")
    all_samples = []
    seen_ids = set()
    
    for chunk_path in sorted(chunks, key=lambda p: p.name):
        if not chunk_path.exists():
            continue
        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  - {chunk_path.name}: Loaded {len(data)} samples")
            for item in data:
                sample_id = item.get("id")
                if sample_id in seen_ids:
                    # Avoid duplicates
                    continue
                seen_ids.add(sample_id)
                all_samples.append(item)
        except Exception as e:
            print(f"  ❌ Error reading {chunk_path.name}: {e}")
            
    print(f"\nTotal compiled unique samples: {len(all_samples)}")
    
    # Write final compiled file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 Successfully compiled final consolidated dataset!")
        print(f"📁 Output file: {output_path}")
        print(f"📊 Total samples: {len(all_samples)} / 5122")
    except Exception as e:
        print(f"❌ Error writing final compiled file: {e}")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
