"""
clean_fineweb.py — Cleans raw FineWeb text for BPE tokenizer training

What this file does:
  Reads the raw FineWeb text dump and produces a cleaned version by:
  - Removing/replacing URLs, email addresses, and file paths
  - Filtering out lines that are mostly non-alphanumeric (boilerplate, nav bars)
  - Collapsing excessive whitespace and blank lines
  - Skipping documents that are too short after cleaning
  - Normalizing unicode whitespace to ASCII equivalents

Where it gets its data:
  datasets/fineweb_raw.txt produced by download_fineweb.py

Who consumes its output:
  datasets/fineweb_clean.txt is fed into build_freqs to build the word
  frequency hashmap for BPE training.

Why this file exists:
  Raw web crawl text contains enormous amounts of junk — URLs, base64 blobs,
  navigation menus, cookie banners — that inflate the unique word count
  without improving tokenizer quality. Cleaning reduces unique words from
  200M+ to a manageable size that fits in 16GB RAM.

Usage:
  python scripts/clean_fineweb.py
"""

import os
import re
import sys
import time
import unicodedata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "fineweb_raw.txt")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "fineweb_clean.txt")

# --- Compiled regex patterns ---

# URLs (http, https, ftp, www)
URL_RE = re.compile(
    r"https?://[^\s\)\]\}\"\'<>]{2,200}"
    r"|ftp://[^\s\)\]\}\"\'<>]{2,200}"
    r"|www\.[^\s\)\]\}\"\'<>]{2,200}",
    re.IGNORECASE,
)

# Email addresses
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]{1,64}@[a-zA-Z0-9.\-]{1,255}\.[a-zA-Z]{2,10}")

# Long hex/base64 blobs (32+ chars of hex-like content)
HEX_BLOB_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")
BASE64_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/=]{40,}\b")

# Collapse runs of 3+ whitespace-separated special chars (nav bars, separators)
NAV_JUNK_RE = re.compile(r"(?:[^\w\s][\s]*){5,}")

# Collapse 3+ consecutive blank lines to 2
BLANK_COLLAPSE_RE = re.compile(r"\n{3,}")

# Lines that are mostly non-alpha (e.g. "|||", "---", "***")
JUNK_LINE_RE = re.compile(r"^[^a-zA-Z]*$")


def clean_line(line: str) -> str:
    """Clean a single line of text."""
    # Remove all special characters (punctuation, bullets, etc.) at the start of a line
    line = re.sub(r"^[ \t]*([^\w\s]+[ \t]*)+", "", line)

    # Replace URLs and emails with nothing (they just inflate unique word count)
    line = URL_RE.sub("", line)
    line = EMAIL_RE.sub("", line)

    # Remove hex/base64 blobs
    line = HEX_BLOB_RE.sub("", line)
    line = BASE64_BLOB_RE.sub("", line)

    # Collapse 4+ repeated special characters to 3 and surround with spaces
    # Example: helo------------world -> helo --- world
    line = re.sub(r"([^\w\s])\1{3,}", r" \1\1\1 ", line)

    # Handle excessive pipes (e.g. |text|text|text)
    # Keep at most 2 pipes per line, replace the rest with spaces
    if "|" in line:
        parts = line.split("|")
        if len(parts) > 3:
            line = parts[0] + " | " + parts[1] + " | " + " ".join(parts[2:])
        else:
            line = line.replace("|", " | ")

    # Normalize unicode whitespace (non-breaking spaces, etc.) to regular space
    line = unicodedata.normalize("NFKC", line)

    # Collapse multiple spaces to single
    line = re.sub(r"[ \t]{2,}", " ", line)

    return line.strip()


def clean_document(doc: str):
    """Clean an entire document. Returns None if too short after cleaning."""
    lines = doc.split("\n")
    cleaned = []

    for line in lines:
        line = clean_line(line)

        # Skip empty lines (will re-add paragraph breaks)
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        # Skip junk lines (all punctuation/symbols, no letters)
        if JUNK_LINE_RE.match(line):
            continue

        # Skip very long "words" that are likely junk (paths, encoded data)
        words = line.split()
        filtered_words = [w for w in words if len(w) <= 50]

        # If we lost more than half the words, skip the line
        if len(filtered_words) < len(words) * 0.5:
            continue

        cleaned.append(" ".join(filtered_words))

    text = "\n".join(cleaned).strip()

    # Skip documents that are too short after cleaning
    if len(text) < 100:
        return None

    return text


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found")
        print("Run download_fineweb.py first.")
        sys.exit(1)

    input_size = os.path.getsize(INPUT_PATH)
    print(f"Cleaning FineWeb raw text")
    print(f"  Input:  {INPUT_PATH} ({input_size / (1024**3):.2f} GB)")
    print(f"  Output: {OUTPUT_PATH}")
    print()

    start = time.time()
    doc_count = 0
    written = 0
    total_input_bytes = 0
    total_output_bytes = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as inp, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as out:

        current_doc = []

        for line in inp:
            total_input_bytes += len(line.encode("utf-8"))

            # Documents are separated by double newlines
            if line.strip() == "" and current_doc and current_doc[-1] == "":
                # End of document — process it
                doc_text = "\n".join(current_doc)
                doc_count += 1

                cleaned = clean_document(doc_text)
                if cleaned:
                    out.write(cleaned)
                    out.write("\n\n")
                    total_output_bytes += len(cleaned) + 2
                    written += 1

                current_doc = []

                if doc_count % 100_000 == 0:
                    elapsed = time.time() - start
                    pct = total_input_bytes / input_size * 100
                    in_gb = total_input_bytes / (1024**3)
                    out_gb = total_output_bytes / (1024**3)
                    print(
                        f"  {doc_count:>10,} docs ({pct:.1f}%) | "
                        f"in: {in_gb:.2f} GB | out: {out_gb:.2f} GB | "
                        f"kept: {written:,} ({written/doc_count*100:.0f}%) | "
                        f"{elapsed:.0f}s",
                        flush=True,
                    )
            else:
                current_doc.append(line.rstrip())

        # Handle last document
        if current_doc:
            doc_text = "\n".join(current_doc)
            doc_count += 1
            cleaned = clean_document(doc_text)
            if cleaned:
                out.write(cleaned)
                out.write("\n\n")
                written += 1

    elapsed = time.time() - start
    out_size = os.path.getsize(OUTPUT_PATH)

    print()
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  Documents: {doc_count:,} total → {written:,} kept ({written/max(doc_count,1)*100:.0f}%)")
    print(f"  Size: {input_size/(1024**3):.2f} GB → {out_size/(1024**3):.2f} GB")
    print(f"  Saved to: {OUTPUT_PATH}")
    print()
    print("Next step: run build_freqs on the cleaned text")
    print(f"  ./build/build_freqs word_freqs.ddfreq datasets/fineweb_clean.txt")


if __name__ == "__main__":
    main()
