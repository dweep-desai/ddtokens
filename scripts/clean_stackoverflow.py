import xml.etree.ElementTree as ET
import html
import re
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "Posts.xml")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "stackoverflow_clean.txt")

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_COLLAPSE_RE = re.compile(r"\n{3,}")

def strip_html(raw_html):
    """Decode HTML entities, strip tags, collapse excessive whitespace."""
    text = html.unescape(raw_html)
    text = HTML_TAG_RE.sub("", text)
    text = WHITESPACE_COLLAPSE_RE.sub("\n\n", text)
    return text.strip()

def main():
    start = time.time()
    count = 0
    written = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        # iterparse + clear pattern keeps memory flat regardless of file size
        context = ET.iterparse(INPUT_PATH, events=("end",))

        for event, elem in context:
            if elem.tag != "row":
                continue

            count += 1
            parts = []

            title = elem.get("Title")
            if title:
                parts.append(strip_html(title))

            body = elem.get("Body")
            if body:
                parts.append(strip_html(body))

            if parts:
                text = "\n".join(parts)
                # Skip posts with almost no content
                if len(text) > 20:
                    out.write(text + "\n\n")
                    written += 1

            # Free parsed element from memory immediately
            elem.clear()

            if count % 500_000 == 0:
                elapsed = time.time() - start
                print(f"  processed {count:,} rows | wrote {written:,} | {elapsed:.0f}s", flush=True)

    elapsed = time.time() - start
    print(f"Done. {count:,} rows processed, {written:,} written to {OUTPUT_PATH} in {elapsed:.0f}s")

if __name__ == "__main__":
    main()
