"""
evaluate_wer.py - WER/CER Evaluation: AnyProjector vs Whisper Standalone

Run same test set on both systems, compare results.
Uses HuggingFace dataset as test set.

Usage:
    python scripts/evaluate_wer.py
    python scripts/evaluate_wer.py --checkpoint path/to/ckpt.pt --num_samples 200
    python scripts/evaluate_wer.py --dataset "doof-ferb/fpt_fosd" --transcript_field transcription
"""

import sys
import os

# Fix Windows cp1252 encoding crash with Vietnamese text
if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EvalConfig:
    checkpoint: str = "projectorTrained/projector_best.pt"
    encoder_id: str = "openai/whisper-medium"
    llm_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset: str = "nguyendv02/ViMD_Dataset"
    dataset_split: str = "test"  # Use test split (unseen data)
    transcript_field: str = "text"
    num_samples: int = 200  # Test set size
    seed: int = 42
    prompt: str = "Transcribe the following audio in Vietnamese:"
    output_file: str = "evaluation_results.json"
    sample_rate: int = 16000


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    # Dynamic programming -- Levenshtein distance at word level
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) between reference and hypothesis."""
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())

    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0

    # Levenshtein distance at character level
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


def load_test_dataset(config: EvalConfig) -> list[dict]:
    """Load test samples from HuggingFace dataset.

    Uses soundfile for audio decoding to avoid torchcodec issues on Windows.
    """
    from datasets import load_dataset
    import soundfile as sf
    import io

    print(f"Loading dataset: {config.dataset} (split={config.dataset_split})")

    # Load WITHOUT automatic audio decoding (avoids torchcodec crash on Windows)
    from datasets import Audio
    ds = load_dataset(
        config.dataset, split=config.dataset_split,
        streaming=True,
    ).cast_column("audio", Audio(decode=False))

    # Shuffle and take test samples
    ds = ds.shuffle(seed=config.seed)

    samples = []
    skipped = 0
    for item in ds:
        if len(samples) >= config.num_samples:
            break

        # Get transcript
        transcript = item.get(config.transcript_field, "")
        if not transcript or not transcript.strip():
            skipped += 1
            continue

        # Get audio -- decode from the audio dict
        audio_data = item.get("audio")
        if not audio_data:
            skipped += 1
            continue

        try:
            # decode=False gives {'bytes': ..., 'path': ...}
            if isinstance(audio_data, dict) and "bytes" in audio_data and audio_data["bytes"]:
                audio_bytes = audio_data["bytes"]
                waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                # Convert stereo to mono if needed
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
            elif isinstance(audio_data, dict) and "array" in audio_data:
                # Pre-decoded (non-streaming fallback)
                waveform = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data["sampling_rate"]
            else:
                skipped += 1
                continue

            # Resample if needed
            if sr != config.sample_rate:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=config.sample_rate)

            samples.append({
                "waveform": waveform,
                "reference": transcript.strip(),
                "duration_s": len(waveform) / config.sample_rate,
            })
        except Exception as e:
            print(f"\n  Warning: Failed to decode audio: {e}")
            skipped += 1
            continue

        print(f"\r  Loading samples: {len(samples)}/{config.num_samples} (skipped {skipped})", end="", flush=True)

    print(f"\n  Loaded {len(samples)} test samples (skipped {skipped})")
    return samples


def evaluate_whisper_standalone(samples: list[dict], encoder_id: str,
                                sample_rate: int) -> list[dict]:
    """Run Whisper standalone (full encoder+decoder) on test samples."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import gc

    print(f"\n{'='*60}")
    print(f"  WHISPER STANDALONE: {encoder_id}")
    print(f"{'='*60}")

    processor = WhisperProcessor.from_pretrained(encoder_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        encoder_id,
        torch_dtype=dtype,
    )
    model = model.to(device).eval()

    # Force Vietnamese
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="vi", task="transcribe"
    )

    results = []
    total = len(samples)

    for i, sample in enumerate(samples):
        t0 = time.time()

        with torch.no_grad():
            inputs = processor(
                sample["waveform"], sampling_rate=sample_rate,
                return_tensors="pt", padding="max_length",
            )
            input_features = inputs.input_features.to(device=device, dtype=dtype)

            generated_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=128,
            )

        latency = (time.time() - t0) * 1000
        hypothesis = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        wer = compute_wer(sample["reference"], hypothesis)
        cer = compute_cer(sample["reference"], hypothesis)

        results.append({
            "reference": sample["reference"],
            "hypothesis": hypothesis,
            "wer": wer,
            "cer": cer,
            "latency_ms": round(latency, 1),
        })

        # Progress
        print(f"\r  {i+1}/{total} | WER={wer:.2f} CER={cer:.2f} | {hypothesis[:50]}...", end="", flush=True)

    print()

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def evaluate_anyprojector(samples: list[dict], config: EvalConfig) -> list[dict]:
    """Run AnyProjector (Whisper encoder -> Q-Former -> LLM) on test samples."""
    import torch
    # Import inference module
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.inference import AnyProjectorInference

    print(f"\n{'='*60}")
    print(f"  ANYPROJECTOR: {config.checkpoint}")
    print(f"{'='*60}")

    model = AnyProjectorInference(
        checkpoint_path=config.checkpoint,
        encoder_id=config.encoder_id,
        llm_id=config.llm_id,
        prompt=config.prompt,
    )

    results = []
    total = len(samples)

    for i, sample in enumerate(samples):
        t0 = time.time()

        with torch.no_grad():
            # Manually run inference on waveform (bypass file loading)
            waveform = sample["waveform"]

            # Encoder mask
            encoder_seq_len = 1500
            samples_per_token = (30.0 * config.sample_rate) / encoder_seq_len
            real_tokens = min(encoder_seq_len, int(len(waveform) / samples_per_token))
            encoder_mask = torch.zeros(1, encoder_seq_len, dtype=torch.bool, device=model.device)
            encoder_mask[0, real_tokens:] = True

            # Whisper encoder
            audio_inputs = model.processor(
                waveform, sampling_rate=config.sample_rate,
                return_tensors="pt", padding="max_length",
            )
            input_features = audio_inputs.input_features.to(model.device)
            encoder_output = model.encoder(input_features).last_hidden_state

            # Q-Former
            audio_embeds = model.projector(encoder_output, encoder_mask)

            # Prompt (chat template matching training)
            prompt_prefix = "<|im_start|>user\n" + config.prompt + "\n"
            prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"

            prefix_tokens = model.tokenizer(
                prompt_prefix, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(model.device)
            prefix_embeds = model.embed_layer(prefix_tokens)

            suffix_tokens = model.tokenizer(
                prompt_suffix, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(model.device)
            suffix_embeds = model.embed_layer(suffix_tokens)

            # Combine + generate
            input_embeds = torch.cat(
                [prefix_embeds, audio_embeds, suffix_embeds], dim=1
            ).to(model.llm_dtype)
            attn_mask = torch.ones(1, input_embeds.shape[1], dtype=torch.long, device=model.device)

            outputs = model.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=model.tokenizer.eos_token_id,
                pad_token_id=model.tokenizer.pad_token_id,
            )

        latency = (time.time() - t0) * 1000
        hypothesis = model.tokenizer.decode(outputs[0], skip_special_tokens=True)

        wer = compute_wer(sample["reference"], hypothesis)
        cer = compute_cer(sample["reference"], hypothesis)

        results.append({
            "reference": sample["reference"],
            "hypothesis": hypothesis,
            "wer": wer,
            "cer": cer,
            "latency_ms": round(latency, 1),
        })

        print(f"\r  {i+1}/{total} | WER={wer:.2f} CER={cer:.2f} | {hypothesis[:50]}...", end="", flush=True)

    print()
    return results


def print_summary(name: str, results: list[dict]):
    """Print summary statistics for a set of results."""
    wers = [r["wer"] for r in results]
    cers = [r["cer"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    avg_wer = sum(wers) / len(wers)
    avg_cer = sum(cers) / len(cers)
    avg_latency = sum(latencies) / len(latencies)

    # WER distribution
    perfect = sum(1 for w in wers if w == 0.0)
    good = sum(1 for w in wers if 0 < w <= 0.2)
    moderate = sum(1 for w in wers if 0.2 < w <= 0.5)
    poor = sum(1 for w in wers if w > 0.5)

    print(f"\n+---------------------------------------------+")
    print(f"|  {name:<43}|")
    print(f"+---------------------------------------------+")
    print(f"|  Avg WER:      {avg_wer:<28.4f}|")
    print(f"|  Avg CER:      {avg_cer:<28.4f}|")
    print(f"|  Avg Latency:  {avg_latency:<25.0f} ms |")
    print(f"+---------------------------------------------+")
    print(f"|  WER Distribution:                          |")
    print(f"|    Perfect (0%):     {perfect:>3}/{len(wers)} samples          |")
    print(f"|    Good (0-20%):     {good:>3}/{len(wers)} samples          |")
    print(f"|    Moderate (20-50%): {moderate:>2}/{len(wers)} samples          |")
    print(f"|    Poor (>50%):      {poor:>3}/{len(wers)} samples          |")
    print(f"+---------------------------------------------+")

    return {"avg_wer": avg_wer, "avg_cer": avg_cer, "avg_latency_ms": avg_latency}


def main():
    parser = argparse.ArgumentParser(description="WER/CER Evaluation: AnyProjector vs Whisper")
    parser.add_argument("--checkpoint", default=EvalConfig.checkpoint, help="Projector checkpoint")
    parser.add_argument("--encoder", default=EvalConfig.encoder_id, help="Whisper model ID")
    parser.add_argument("--llm", default=EvalConfig.llm_id, help="LLM model ID")
    parser.add_argument("--dataset", default=EvalConfig.dataset, help="HuggingFace dataset name")
    parser.add_argument("--dataset_split", default=EvalConfig.dataset_split, help="Dataset split (train/test)")
    parser.add_argument("--transcript_field", default=EvalConfig.transcript_field, help="Transcript field name")
    parser.add_argument("--num_samples", type=int, default=EvalConfig.num_samples, help="Number of test samples")
    parser.add_argument("--output", default=EvalConfig.output_file, help="Output JSON file")
    args = parser.parse_args()

    config = EvalConfig(
        checkpoint=args.checkpoint,
        encoder_id=args.encoder,
        llm_id=args.llm,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        transcript_field=args.transcript_field,
        num_samples=args.num_samples,
        output_file=args.output,
    )

    # Load test data
    samples = load_test_dataset(config)

    if len(samples) == 0:
        print("Error: No valid samples found!")
        return

    # -- 1. AnyProjector (run FIRST for faster error feedback) --
    anyprojector_results = evaluate_anyprojector(samples, config)
    ap_summary = print_summary("AnyProjector (Q-Former)", anyprojector_results)

    # -- 2. Whisper Standalone (with cache) --
    whisper_cache = Path(config.output_file).with_name("whisper_cache.json")
    if whisper_cache.exists():
        import json as _json
        cached = _json.loads(whisper_cache.read_text(encoding="utf-8"))
        if cached.get("num_samples") == len(samples) and cached.get("encoder") == config.encoder_id:
            print(f"\n{'='*60}")
            print(f"  WHISPER STANDALONE: {config.encoder_id} (CACHED)")
            print(f"{'='*60}")
            whisper_results = cached["results"]
            whisper_summary = cached["summary"]
            print(f"  Loaded from cache: {whisper_cache}")
            print(f"  Avg WER: {whisper_summary['avg_wer']:.4f}")
            print(f"  Avg CER: {whisper_summary['avg_cer']:.4f}")
        else:
            whisper_results = evaluate_whisper_standalone(
                samples, config.encoder_id, config.sample_rate
            )
            whisper_summary = print_summary("Whisper Standalone", whisper_results)
    else:
        whisper_results = evaluate_whisper_standalone(
            samples, config.encoder_id, config.sample_rate
        )
        whisper_summary = print_summary("Whisper Standalone", whisper_results)

    # Save Whisper cache
    whisper_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(whisper_cache, "w", encoding="utf-8") as f:
        json.dump({
            "encoder": config.encoder_id,
            "num_samples": len(samples),
            "summary": whisper_summary,
            "results": whisper_results[:50],
        }, f, ensure_ascii=False, indent=2)

    # -- 3. Comparison --
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Whisper':>12} {'AnyProjector':>12} {'Delta':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Avg WER':<20} {whisper_summary['avg_wer']:>12.4f} {ap_summary['avg_wer']:>12.4f} {ap_summary['avg_wer'] - whisper_summary['avg_wer']:>+10.4f}")
    print(f"  {'Avg CER':<20} {whisper_summary['avg_cer']:>12.4f} {ap_summary['avg_cer']:>12.4f} {ap_summary['avg_cer'] - whisper_summary['avg_cer']:>+10.4f}")
    print(f"  {'Avg Latency (ms)':<20} {whisper_summary['avg_latency_ms']:>12.0f} {ap_summary['avg_latency_ms']:>12.0f} {ap_summary['avg_latency_ms'] - whisper_summary['avg_latency_ms']:>+10.0f}")

    # -- 4. Save results --
    output = {
        "config": {
            "checkpoint": config.checkpoint,
            "encoder": config.encoder_id,
            "llm": config.llm_id,
            "dataset": config.dataset,
            "num_samples": len(samples),
        },
        "whisper": {
            "summary": whisper_summary,
            "samples": whisper_results[:10],
        },
        "anyprojector": {
            "summary": ap_summary,
            "samples": anyprojector_results[:10],
        },
    }

    out_path = Path(config.output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        with open("_eval_error.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        sys.exit(1)
