import json
import sys

def main():
    try:
        with open("dataset/val_500_raw.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"Total validation samples: {len(data)}")
        # Save a clean text summary of all 500 samples
        with open("dataset/val_500_summary.txt", "w", encoding="utf-8") as out:
            for item in data:
                out.write(f"ID: {item['id']} | INTENT: {item['intent']} | UTT: {item['utt']}\n")
        print("Successfully saved summary to dataset/val_500_summary.txt")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
