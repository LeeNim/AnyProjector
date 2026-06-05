"""
prepare_fpt_dataset.py — Convert FPT Open Speech Data → metadata.jsonl for Phase 2.

Reads transcriptAll.txt (pipe-delimited) and creates metadata.jsonl
compatible with train_phase2.py.

Filters:
    - Skip entries with -N (noise marker)
    - Only use original files (skip " 1", " 2", " 3" copies)
    - Skip entries with multi-segment timestamps (contains space in timestamp)

Usage:
    python scripts/prepare_fpt_dataset.py
"""

import json
import os
from pathlib import Path


def main(max_samples: int = None):
    src_dir = Path("dataset/k9sxg2twv4-4")
    transcript_path = src_dir / "transcriptAll.txt"
    output_path = src_dir / "metadata.jsonl"
    mp3_dir = src_dir / "mp3"

    print(f"Reading: {transcript_path}")
    lines = transcript_path.read_text(encoding="utf-8").strip().split("\n")
    print(f"Total lines: {len(lines)}")

    entries = []
    skipped_noise = 0
    skipped_missing = 0
    skipped_multi_seg = 0

    for line in lines:
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue

        filename = parts[0].strip()
        transcript = parts[1].strip()
        timestamps = parts[2].strip()

        # Skip noise markers
        if "-N" in transcript:
            skipped_noise += 1
            continue

        # Skip multi-segment timestamps (e.g. "0.00-3.5 3.5-7.0")
        if " " in timestamps:
            skipped_multi_seg += 1
            continue

        # Check file exists
        audio_path = mp3_dir / filename
        if not audio_path.exists():
            skipped_missing += 1
            continue

        # Clean transcript (remove trailing spaces, normalize)
        transcript = transcript.strip()
        if not transcript:
            continue

        entries.append({
            "audio_file": f"mp3/{filename}",
            "transcript": transcript,
        })

    # Random sample if needed
    if max_samples and len(entries) > max_samples:
        import random
        random.seed(42)
        entries = random.sample(entries, max_samples)
        print(f"Random sampled: {max_samples} from {len(lines)} total")

    # Write metadata.jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"Created: {output_path}")
    print(f"Total entries: {len(entries)}")
    print(f"Skipped (noise -N): {skipped_noise}")
    print(f"Skipped (multi-seg): {skipped_multi_seg}")
    print(f"Skipped (missing file): {skipped_missing}")
    print(f"{'='*50}")

    # Stats
    import statistics
    transcript_lens = [len(e["transcript"]) for e in entries]
    print(f"\nTranscript length stats:")
    print(f"  Min: {min(transcript_lens)} chars")
    print(f"  Max: {max(transcript_lens)} chars")
    print(f"  Mean: {statistics.mean(transcript_lens):.1f} chars")
    print(f"  Median: {statistics.median(transcript_lens):.1f} chars")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()
    main(max_samples=args.max_samples)

