# -*- coding: utf-8 -*-
"""
convert_canvas_csv_to_json.py - Converts the completed CSV downloaded from Gemini Web / Canvas
into the standardized JSON format ready for compilation.
"""

import csv
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
    print("  Canvas CSV to JSON Converter for Speech-MASSIVE")
    print("=" * 60)
    
    dataset_dir = Path("dataset")
    
    # Prompt user for input file or use default
    default_input = dataset_dir / "completed_canvas_output.csv"
    print(f"Default input path: {default_input}")
    
    input_str = input("Enter path to the completed CSV file (Press Enter to use default): ").strip()
    input_path = Path(input_str) if input_str else default_input
    
    if not input_path.exists():
        print(f"❌ Error: File not found at: {input_path}")
        print("Please place the downloaded CSV file in the 'dataset' directory and try again.")
        return
        
    output_json = dataset_dir / "speech_massive_lora_canvas_completed.json"
    
    print(f"\nReading {input_path.name}...")
    samples = []
    skipped_headers = 0
    missing_response = 0
    
    with open(input_path, "r", encoding="utf-8-sig") as f:
        # We use dict reader to handle different column name casings/formats
        reader = csv.reader(f)
        
        # Read header and normalize column names
        header = next(reader)
        header_normalized = [h.strip().lower() for h in header]
        
        print("CSV Header columns detected:", header)
        
        # Find column indices
        try:
            id_idx = next(i for i, h in enumerate(header_normalized) if 'id' in h)
            utt_idx = next(i for i, h in enumerate(header_normalized) if any(x in h for x in ['transcript', 'utt']))
            intent_idx = next(i for i, h in enumerate(header_normalized) if 'intent' in h)
            scenario_idx = next(i for i, h in enumerate(header_normalized) if 'scenario' in h)
            response_idx = next(i for i, h in enumerate(header_normalized) if 'response' in h)
        except StopIteration as e:
            print("❌ Error: Missing required columns in CSV header.")
            print("Make sure your CSV contains columns for: ID, Transcript/Utt, Intent, Scenario, and Response.")
            return
            
        for row_num, row in enumerate(reader, start=2):
            if not row or len(row) <= max(id_idx, utt_idx, intent_idx, scenario_idx, response_idx):
                continue
                
            sample_id = row[id_idx].strip()
            utt = row[utt_idx].strip()
            intent = row[intent_idx].strip()
            scenario = row[scenario_idx].strip()
            response = row[response_idx].strip()
            
            if not response:
                missing_response += 1
                # Still add it, but print warning later
                
            samples.append({
                "id": sample_id,
                "utt": utt,
                "intent": intent,
                "scenario": scenario,
                "response": response
            })
            
    print(f"Parsed {len(samples)} samples successfully.")
    if missing_response > 0:
        print(f"⚠️ Warning: {missing_response} samples have empty 'response' field.")
        
    # Write to standardized JSON file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        
    print(f"💾 Standardized JSON file saved to: {output_json}")
    print("\n🎉 Conversion completed successfully!")
    print("Now you can run compile_final_dataset.py to generate the final LoRA training set!")
    print("=" * 60)

if __name__ == "__main__":
    main()
