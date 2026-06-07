"""
stream_fineweb.py — Streams FineWeb text content to stdout for piping into build_freqs

What this file does:
  Loads the FineWeb sample-10BT dataset (from local cache if available, otherwise
  streams from HuggingFace Hub) and writes each document's text to stdout,
  one line per line of text. This output is piped directly into the C++ build_freqs
  executable to build the word frequency hashmap.

Where it gets its data:
  HuggingFaceFW/fineweb sample-10BT subset via the datasets library.
  Uses streaming mode so the full dataset doesn't need to fit in memory.

Who consumes its output:
  build_freqs reads from stdin (via the "-" argument) and feeds each line
  into BPETokenizer::build_word_frequencies().

Why this file exists:
  Bridges the HuggingFace Python ecosystem with the C++ tokenizer.
  Streaming avoids downloading the entire ~30GB dataset to disk first.

Prerequisites:
  pip install datasets

Usage:
  python scripts/stream_fineweb.py | ./build/build_freqs word_freqs.ddfreq -
"""

import sys
from datasets import load_dataset

def main():
    # Stream mode — never loads the full dataset into memory.
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    count = 0
    for example in dataset:
        text = example.get("text", "")
        if text:
            # Write the document text directly to stdout.
            # build_word_frequencies() handles line-by-line splitting internally.
            sys.stdout.write(text)
            sys.stdout.write("\n")

            count += 1
            if count % 100000 == 0:
                print(f"  [stream_fineweb] streamed {count // 1000}k documents", file=sys.stderr)

    print(f"  [stream_fineweb] done — {count} documents total", file=sys.stderr)

if __name__ == "__main__":
    main()
