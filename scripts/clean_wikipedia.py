"""
reclean_wikipedia.py — Applies additional cleaning to already extracted Wikipedia text

What this file does:
  Reads the existing `wikipedia_clean.txt`, applies new cleaning rules
  (removing leading bullets/dashes/stars, handling excessive pipes),
  and writes the cleaned content to a temporary file before replacing
  the original.

Why this file exists:
  The original raw XML is no longer available, so we must clean the
  already extracted text directly to fix excessive pipes and bullets.
"""

import os
import re
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "wikipedia_clean.txt")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "wikipedia_clean_temp.txt")

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    print(f"Recleaning Wikipedia text...")
    print(f"  Input:  {INPUT_PATH}")
    print(f"  Output: {OUTPUT_PATH}")

    start = time.time()
    total_lines = 0
    
    # Pre-compile regex for performance
    # Remove all special characters (punctuation, bullets, etc.) and spaces at the start of a line
    START_SPECIAL_RE = re.compile(r"^[\W]+")
    # Collapse 4+ repeated special characters to 3 and surround with spaces
    REPEAT_RE = re.compile(r"([^\w\s])\1{3,}")

    with open(INPUT_PATH, "r", encoding="utf-8") as inp, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as out:
         
        for line in inp:
            total_lines += 1
            
            # Remove leading special characters
            line = START_SPECIAL_RE.sub("", line)
            
            # Collapse 4+ repeated special characters to 3 and surround with spaces
            line = REPEAT_RE.sub(r" \1\1\1 ", line)
            
            # Handle excessive pipes
            if "|" in line:
                parts = line.split("|")
                if len(parts) > 3:
                    # Keep first two pipes, space pad the rest
                    line = parts[0] + " | " + parts[1] + " | " + " ".join(parts[2:])
                else:
                    line = line.replace("|", " | ")
                    
            out.write(line)
            
            if total_lines % 10_000_000 == 0:
                elapsed = time.time() - start
                print(f"  processed {total_lines:,} lines | {elapsed:.0f}s", flush=True)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.0f}s. Replacing old file...")
    
    # Replace original file with the newly cleaned file
    os.replace(OUTPUT_PATH, INPUT_PATH)
    print("Recleaning finished successfully.")

if __name__ == "__main__":
    main()
