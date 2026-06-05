"""
compare_projectors.py — So sánh 64 vs 128 token projector
Dùng DemoEngine (đã test hoạt động trên máy), swap projector giữa 2 pass.
Output: results/compare_64_vs_128.md
"""

import sys, os, json, re, time, gc
sys.path.insert(0, os.getcwd())

# IMPORTANT: datasets must be imported BEFORE scripts.demo
# (bitsandbytes + datasets import order causes segfault on Windows otherwise)
from datasets import load_dataset, concatenate_datasets

import torch
import numpy as np
from src.projector import AnyProjector
from scripts.demo import DemoEngine, _normalize_audio, SAMPLE_RATE

# ═══════════════════════════════════════════
# Config
# ═══════════════════════════════════════════
PROJ_64_PATH  = r"C:\Users\suoya\Downloads\projector_final.pt"
PROJ_128_PATH = r"C:\Users\suoya\Downloads\projector_final (1).pt"
LORA_PATH     = "lora/best08"

DATASET_ID  = "Niem/speech-massive-vie-tool-calling"
N_SAMPLES   = 100
OUTPUT_PATH = "results/compare_64_vs_128.md"

# ═══════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════
def parse_tool_call(text):
    match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass
    return None

def compute_word_f1(pred, ref):
    pw = set(pred.lower().split())
    rw = set(ref.lower().split())
    if not pw or not rw: return 0.0
    c = len(pw & rw)
    p, r = c/len(pw), c/len(rw)
    return 2*p*r/(p+r) if (p+r)>0 else 0.0

def args_f1(gen_t, exp_t):
    if not gen_t or not exp_t: return 0.0
    ga = set(str(v).lower() for v in gen_t.get("arguments", gen_t.get("args", {})).values())
    ea = set(str(v).lower() for v in exp_t.get("arguments", exp_t.get("args", {})).values())
    if not ga or not ea: return 0.0
    c = len(ga & ea)
    p, r = c/len(ga), c/len(ea)
    return 2*p*r/(p+r) if (p+r)>0 else 0.0

def clean_text(t):
    return re.sub(r'<tool_call>.*?</tool_call>', '', t, flags=re.DOTALL).strip()

def load_projector(path, device):
    """Load projector checkpoint → AnyProjector on device."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    sd = ckpt.get("projector_state_dict", ckpt)
    layer_idx = {int(k.split(".")[1]) for k in sd if k.startswith("layers.")}
    nl = max(layer_idx) + 1 if layer_idx else 4
    proj = AnyProjector(
        encoder_dim=config.get("encoder_dim", 1024),
        llm_dim=config.get("llm_dim", 1536),
        num_queries=config.get("num_queries", 64),
        qformer_dim=config.get("qformer_dim", 768),
        num_layers=nl,
        num_heads=config.get("qformer_heads", 16),
        dropout=config.get("dropout", 0.0),
    )
    proj.load_state_dict(sd)
    nq = config.get("num_queries", 64)
    print(f"  Projector: {proj.count_parameters():,} params, num_queries={nq}")
    del ckpt, sd
    return proj.to(device).eval(), nq

# ═══════════════════════════════════════════
# Init engine (loads Whisper + LLM + LoRA + default projector)
# ═══════════════════════════════════════════
print("=" * 50)
print("  Initializing DemoEngine...")
print("=" * 50)
engine = DemoEngine(
    checkpoint_path=PROJ_64_PATH,  # start with 64
    lora_path=LORA_PATH,
)
print("Engine ready.\n")

# ═══════════════════════════════════════════
# Load dataset
# ═══════════════════════════════════════════
print(f"Loading dataset: {DATASET_ID}")
from datasets import Audio as AudioFeature
raw_ds = load_dataset(DATASET_ID)
# Disable auto audio decoding (avoids torchcodec requirement)
for split in raw_ds:
    raw_ds[split] = raw_ds[split].cast_column("audio", AudioFeature(decode=False))
merged = concatenate_datasets([raw_ds[s] for s in raw_ds.keys()]).shuffle(seed=42)
eval_ds = merged.select(range(min(N_SAMPLES, len(merged))))
print(f"  Eval: {len(eval_ds)} samples")

# Pre-extract audio (decode with soundfile)
import soundfile as sf
import io
print("Pre-extracting audio...")
samples = []
for i in range(len(eval_ds)):
    row = eval_ds[i]
    audio_info = row["audio"]
    # Decode audio bytes with soundfile
    audio_bytes = audio_info["bytes"]
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if sr != SAMPLE_RATE:
        # Simple resample if needed
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    instruction = str(row.get("instruction", row.get("utt", "")))[:120]
    expected = str(row.get("output", row.get("response", "")))
    samples.append({"wav": wav, "instruction": instruction, "expected": expected})
    if (i+1) % 20 == 0:
        print(f"  Extracted {i+1}/{len(eval_ds)}")
print(f"  {len(samples)} samples ready\n")

# ═══════════════════════════════════════════
# PASS 1: 64 tokens (already loaded)
# ═══════════════════════════════════════════
print("=" * 50)
print("  PASS 1: 64 tokens")
print("=" * 50)

gen_64_list = []
t0_total = time.time()
for i, s in enumerate(samples):
    result = engine.transcribe_projector(s["wav"], temperature=0.0, max_tokens=192)
    gen_64_list.append(result.get("text", ""))
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(samples)} done")
time_64 = time.time() - t0_total
print(f"  Total: {time_64:.0f}s ({time_64/len(samples)*1000:.0f}ms/sample)\n")

# ═══════════════════════════════════════════
# PASS 2: Swap to 128 tokens
# ═══════════════════════════════════════════
print("=" * 50)
print("  PASS 2: 128 tokens (swapping projector)")
print("=" * 50)

# Swap projector
del engine.projector
gc.collect(); torch.cuda.empty_cache()
engine.projector, nq_128 = load_projector(PROJ_128_PATH, engine.device)

gen_128_list = []
t0_total = time.time()
for i, s in enumerate(samples):
    result = engine.transcribe_projector(s["wav"], temperature=0.0, max_tokens=192)
    gen_128_list.append(result.get("text", ""))
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(samples)} done")
time_128 = time.time() - t0_total
print(f"  Total: {time_128:.0f}s ({time_128/len(samples)*1000:.0f}ms/sample)\n")

# ═══════════════════════════════════════════
# Compute metrics
# ═══════════════════════════════════════════
print("Computing metrics...")
results = []
for i, s in enumerate(samples):
    exp = s["expected"]
    g64 = gen_64_list[i]
    g128 = gen_128_list[i]

    exp_tool = parse_tool_call(exp)
    t64 = parse_tool_call(g64)
    t128 = parse_tool_call(g128)
    exp_name = exp_tool["name"] if exp_tool else None

    results.append({
        "idx": i,
        "instruction": s["instruction"],
        "expected": exp[:150],
        "gen_64": g64[:150],
        "gen_128": g128[:150],
        "exp_name": exp_name,
        "name_64_ok": (t64["name"] == exp_name) if (t64 and exp_name) else None,
        "name_128_ok": (t128["name"] == exp_name) if (t128 and exp_name) else None,
        "args_f1_64": args_f1(t64, exp_tool),
        "args_f1_128": args_f1(t128, exp_tool),
        "text_f1_64": compute_word_f1(clean_text(g64), clean_text(exp)),
        "text_f1_128": compute_word_f1(clean_text(g128), clean_text(exp)),
    })

tool_samples = [r for r in results if r["exp_name"] is not None]
n_tool = len(tool_samples)
n_total = len(results)

name_acc_64 = sum(1 for r in tool_samples if r["name_64_ok"]) / max(n_tool, 1)
name_acc_128 = sum(1 for r in tool_samples if r["name_128_ok"]) / max(n_tool, 1)
avg_args_64 = sum(r["args_f1_64"] for r in tool_samples) / max(n_tool, 1)
avg_args_128 = sum(r["args_f1_128"] for r in tool_samples) / max(n_tool, 1)
avg_text_64 = sum(r["text_f1_64"] for r in results) / max(n_total, 1)
avg_text_128 = sum(r["text_f1_128"] for r in results) / max(n_total, 1)

# ═══════════════════════════════════════════
# Write MD
# ═══════════════════════════════════════════
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def winner(a, b, higher_better=True):
    if higher_better: return "128" if b>a else ("64" if a>b else "tie")
    return "64" if a<b else ("128" if b<a else "tie")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("# Projector Comparison: 64 vs 128 Tokens\n\n")
    f.write(f"**Samples:** {n_total} | **Tool-call samples:** {n_tool}\n")
    f.write(f"**LoRA:** `{LORA_PATH}` (shared)\n\n")

    f.write("## Aggregate Metrics\n\n")
    f.write("| Metric | 64 Tokens | 128 Tokens | Winner |\n")
    f.write("|--------|-----------|------------|--------|\n")
    f.write(f"| Tool Name Accuracy | {name_acc_64:.1%} | {name_acc_128:.1%} | **{winner(name_acc_64, name_acc_128)}** |\n")
    f.write(f"| Tool Args F1 | {avg_args_64:.1%} | {avg_args_128:.1%} | **{winner(avg_args_64, avg_args_128)}** |\n")
    f.write(f"| Response Text F1 | {avg_text_64:.1%} | {avg_text_128:.1%} | **{winner(avg_text_64, avg_text_128)}** |\n")
    f.write(f"| Total Time (s) | {time_64:.0f} | {time_128:.0f} | **{winner(time_64, time_128, False)}** |\n")
    f.write(f"| ms/sample | {time_64/n_total*1000:.0f} | {time_128/n_total*1000:.0f} | **{winner(time_64, time_128, False)}** |\n")

    f.write("\n## Sample Results (first 30)\n\n")
    for r in results[:30]:
        i64 = "✅" if r["name_64_ok"] else ("❌" if r["name_64_ok"] is not None else "—")
        i128 = "✅" if r["name_128_ok"] else ("❌" if r["name_128_ok"] is not None else "—")
        f.write(f"### [{r['idx']+1}] {r['instruction']}\n\n")
        f.write(f"**Expected:** `{r['expected']}`\n\n")
        f.write(f"| | 64 Tokens | 128 Tokens |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| Output | `{r['gen_64']}` | `{r['gen_128']}` |\n")
        f.write(f"| Tool | {i64} | {i128} |\n")
        f.write(f"| Args F1 | {r['args_f1_64']:.0%} | {r['args_f1_128']:.0%} |\n")
        f.write(f"| Text F1 | {r['text_f1_64']:.0%} | {r['text_f1_128']:.0%} |\n\n")

    f.write("\n## Per-Tool Breakdown\n\n")
    tool_names = sorted(set(r["exp_name"] for r in tool_samples if r["exp_name"]))
    f.write("| Tool | Count | Acc 64 | Acc 128 | Winner |\n")
    f.write("|------|-------|--------|---------|--------|\n")
    for tn in tool_names:
        sub = [r for r in tool_samples if r["exp_name"] == tn]
        cnt = len(sub)
        a64 = sum(1 for r in sub if r["name_64_ok"]) / cnt
        a128 = sum(1 for r in sub if r["name_128_ok"]) / cnt
        f.write(f"| {tn} | {cnt} | {a64:.0%} | {a128:.0%} | {winner(a64, a128)} |\n")

print(f"\n{'='*50}")
print(f"  RESULTS: {OUTPUT_PATH}")
print(f"{'='*50}")
print(f"  Tool Name: 64={name_acc_64:.1%} vs 128={name_acc_128:.1%}")
print(f"  Args F1:   64={avg_args_64:.1%} vs 128={avg_args_128:.1%}")
print(f"  Text F1:   64={avg_text_64:.1%} vs 128={avg_text_128:.1%}")
