import xml.etree.ElementTree as ET
import re
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "enwiki-latest-pages-articles-multistream.xml")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "wikipedia_clean.txt")

# MediaWiki XML namespace
NS = "{http://www.mediawiki.org/xml/export-0.11/}"

# Precompiled regexes for wiki markup stripping
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL)
REF_SELF_RE = re.compile(r"<ref[^/]*/\s*>")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
FILE_RE = re.compile(r"\[\[(File|Image|Category):[^\]]*\]\]", re.IGNORECASE)
PIPED_LINK_RE = re.compile(r"\[\[[^|\]]*\|([^\]]*)\]\]")
PLAIN_LINK_RE = re.compile(r"\[\[([^\]]*)\]\]")
EXT_LINK_LABEL_RE = re.compile(r"\[https?://\S+\s+([^\]]+)\]")
EXT_LINK_BARE_RE = re.compile(r"\[https?://\S+\]")
HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+$", re.MULTILINE)
BOLD_ITALIC_RE = re.compile(r"'{2,5}")
WHITESPACE_COLLAPSE_RE = re.compile(r"\n{3,}")

def strip_wiki_markup(text):
    """Best-effort conversion of wikitext to plain text."""
    # Remove references
    text = REF_RE.sub("", text)
    text = REF_SELF_RE.sub("", text)

    # Remove HTML comments
    text = COMMENT_RE.sub("", text)

    # Remove tables
    text = TABLE_RE.sub("", text)

    # Iteratively remove nested templates (handles up to ~5 levels deep)
    for _ in range(5):
        new_text = TEMPLATE_RE.sub("", text)
        if new_text == text:
            break
        text = new_text

    # Remove file/image/category links
    text = FILE_RE.sub("", text)

    # Convert piped links [[target|display]] -> display
    text = PIPED_LINK_RE.sub(r"\1", text)

    # Convert plain links [[target]] -> target
    text = PLAIN_LINK_RE.sub(r"\1", text)

    # External links [http://... label] -> label
    text = EXT_LINK_LABEL_RE.sub(r"\1", text)
    text = EXT_LINK_BARE_RE.sub("", text)

    # Strip remaining HTML tags
    text = HTML_TAG_RE.sub("", text)

    # Headings: == Foo == -> Foo
    text = HEADING_RE.sub(r"\1", text)

    # Bold/italic markers
    text = BOLD_ITALIC_RE.sub("", text)

    # Collapse excessive newlines
    text = WHITESPACE_COLLAPSE_RE.sub("\n\n", text)

    return text.strip()

# Sections that are boilerplate, not useful prose
SKIP_SECTIONS = {
    "see also", "references", "external links", "further reading",
    "notes", "bibliography", "sources", "citations"
}

def should_skip_page(title, ns, text):
    """Filter out non-article pages and stubs."""
    if ns != "0":
        return True
    if text is None:
        return True
    # Redirect pages
    if text.strip().upper().startswith("#REDIRECT"):
        return True
    # Disambiguation pages
    if "{{disambiguation}}" in text.lower():
        return True
    # Very short articles are noise
    if len(text) < 200:
        return True
    return False

def main():
    start = time.time()
    page_count = 0
    written = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        context = ET.iterparse(INPUT_PATH, events=("end",))

        title = None
        ns = None

        for event, elem in context:
            tag = elem.tag

            if tag == f"{NS}title":
                title = elem.text or ""
            elif tag == f"{NS}ns":
                ns = elem.text or ""
            elif tag == f"{NS}text":
                raw = elem.text or ""

                if not should_skip_page(title, ns, raw):
                    clean = strip_wiki_markup(raw)
                    if len(clean) > 100:
                        out.write(f"{title}\n{clean}\n\n")
                        written += 1
            elif tag == f"{NS}page":
                page_count += 1
                title = None
                ns = None
                # Free entire page subtree from memory
                elem.clear()

                if page_count % 100_000 == 0:
                    elapsed = time.time() - start
                    print(f"  processed {page_count:,} pages | wrote {written:,} | {elapsed:.0f}s", flush=True)

    elapsed = time.time() - start
    print(f"Done. {page_count:,} pages processed, {written:,} written to {OUTPUT_PATH} in {elapsed:.0f}s")

if __name__ == "__main__":
    main()
