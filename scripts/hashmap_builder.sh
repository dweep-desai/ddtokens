#!/bin/bash
# hashmap_builder.sh — One-click word frequency hashmap builder (macOS/Linux)
#
# What this script does:
#   1. Builds the C++ project (if not already built)
#   2. Ingests all local text files in datasets/
#   3. Streams the FineWeb dataset from HuggingFace
#   4. Saves the combined word frequency hashmap to word_freqs.ddfreq
#
# Usage:
#   chmod +x scripts/hashmap_builder.sh
#   ./scripts/hashmap_builder.sh

set -e

# Resolve project root (script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="$PROJECT_ROOT/build"
DATASETS_DIR="$PROJECT_ROOT/datasets"
OUTPUT_FILE="$PROJECT_ROOT/word_freqs.ddfreq"
BUILD_FREQS="$BUILD_DIR/build_freqs"

echo "=========================================="
echo "  ddtokens — Hashmap Builder"
echo "=========================================="
echo ""

# --- Step 1: Build the C++ project ---
echo "[1/3] Building C++ project..."
cmake -B "$BUILD_DIR" -S "$PROJECT_ROOT" > /dev/null 2>&1
cmake --build "$BUILD_DIR" --target build_freqs > /dev/null 2>&1
echo "  ✓ build_freqs compiled"
echo ""

# --- Step 2: Collect all local dataset files ---
LOCAL_FILES=()
if [ -d "$DATASETS_DIR" ]; then
    for f in "$DATASETS_DIR"/*.txt; do
        [ -f "$f" ] && LOCAL_FILES+=("$f")
    done
fi

if [ ${#LOCAL_FILES[@]} -gt 0 ]; then
    echo "[2/3] Found ${#LOCAL_FILES[@]} local dataset file(s):"
    for f in "${LOCAL_FILES[@]}"; do
        SIZE=$(du -h "$f" | cut -f1)
        echo "  • $(basename "$f") ($SIZE)"
    done
else
    echo "[2/3] No local .txt files found in datasets/"
fi
echo ""

# --- Step 3: Stream FineWeb + ingest local files ---
echo "[3/3] Building hashmap (local files + FineWeb stream)..."
echo "  Output: $OUTPUT_FILE"
echo ""

# Pipe FineWeb through stdin, pass local files as args, "-" reads the pipe.
python3 "$SCRIPT_DIR/stream_fineweb.py" \
    | "$BUILD_FREQS" "$OUTPUT_FILE" "${LOCAL_FILES[@]}" -

echo ""
echo "=========================================="
echo "  ✓ Hashmap saved to: $OUTPUT_FILE"
echo ""
echo "  Next step — run BPE training:"
echo "  ./build/train_bpe <num_merges> $OUTPUT_FILE"
echo "=========================================="
