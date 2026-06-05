import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

d = json.load(open("evaluation_results.json", "r", encoding="utf-8"))

print("=" * 80)
print("ANYPROJECTOR SAMPLES")
print("=" * 80)
for i, s in enumerate(d["anyprojector"]["samples"]):
    print(f"\n--- Sample {i+1} | WER={s['wer']:.2f} ---")
    print(f"REF: {s['reference'][:120]}")
    print(f"AP:  {s['hypothesis'][:120]}")

print("\n" + "=" * 80)
print("WHISPER SAMPLES")
print("=" * 80)
for i, s in enumerate(d["whisper"]["samples"]):
    print(f"\n--- Sample {i+1} | WER={s['wer']:.2f} ---")
    print(f"REF: {s['reference'][:120]}")
    print(f"WSP: {s['hypothesis'][:120]}")
