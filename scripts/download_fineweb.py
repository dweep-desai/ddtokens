"""
download_fineweb.py — Downloads a subset of FineWeb 10BT as plain text

What this file does:
  Streams FineWeb sample-10BT from HuggingFace Hub and saves each document's
  text content to a plain text file. Uses streaming mode so the full dataset
  never needs to fit in memory.

Where it gets its data:
  HuggingFaceFW/fineweb sample-10BT subset via the datasets library.

Who consumes its output:
  clean_fineweb.py processes the raw text into a cleaned version suitable
  for BPE training via build_freqs.

Why this file exists:
  Downloads a manageable subset of FineWeb as plain text, avoiding the need
  to stream from HuggingFace every time we want to rebuild frequencies.
  Having local files also enables offline iteration.

Prerequisites:
  pip install datasets

Usage:
  python scripts/download_fineweb.py                   # default 2M docs
  python scripts/download_fineweb.py --max-docs 500000 # custom limit
"""

import os
import sys
import time
import argparse
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "fineweb_raw.txt")


def main():
    parser = argparse.ArgumentParser(description="Download FineWeb subset as plain text")
    parser.add_argument(
        "--max-docs", type=int, default=2_000_000,
        help="Maximum number of documents to download (default: 2,000,000)"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Downloading FineWeb sample-10BT ({args.max_docs:,} docs max)")
    print(f"Output: {OUTPUT_PATH}")
    print()

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    start = time.time()
    count = 0
    total_bytes = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for example in dataset:
            text = example.get("text", "")
            if not text or len(text.strip()) < 50:
                continue

            out.write(text.rstrip())
            out.write("\n\n")  # double newline separates documents

            count += 1
            total_bytes += len(text)

            if count % 50_000 == 0:
                elapsed = time.time() - start
                gb = total_bytes / (1024 ** 3)
                rate = count / elapsed if elapsed > 0 else 0
                eta_min = (args.max_docs - count) / rate / 60 if rate > 0 else 0
                print(
                    f"  {count:>10,} docs | {gb:.2f} GB | "
                    f"{rate:.0f} docs/s | ETA {eta_min:.0f} min",
                    flush=True,
                )

            if count >= args.max_docs:
                break

    elapsed = time.time() - start
    gb = total_bytes / (1024 ** 3)
    file_gb = os.path.getsize(OUTPUT_PATH) / (1024 ** 3)

    print()
    print(f"Done. {count:,} documents downloaded in {elapsed/60:.1f} min")
    print(f"  Content: {gb:.2f} GB")
    print(f"  File on disk: {file_gb:.2f} GB")
    print(f"  Saved to: {OUTPUT_PATH}")
    print()
    print("Next step: python scripts/clean_fineweb.py")


if __name__ == "__main__":
    main()
