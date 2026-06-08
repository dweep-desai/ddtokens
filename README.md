# ddtokens

creating a tokenizer (byte level bpe) and possibly trying to replicate cl100k_base's massive token dictionary. (Note : did not include code in dataset due to massive unique word hashmap entries blow up - this project only deals with general text.)

## Papers

- Neural Machine Translation of Rare Words with Subword Units (2016)
  - Paper: https://arxiv.org/abs/1508.07909

- Language Models are Unsupervised Multitask Learners (GPT-2)
  - Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

- Training Multilingual Pre-trained Language Model with Byte-level Subwords
  - Paper: https://arxiv.org/abs/2101.09469

## Datasets

- wikipedia - https://dumps.wikimedia.org/enwiki/latest/
- stack overflow - https://archive.org/download/stackexchange (Did not use due to RAM limitations - while building the hashmap, it had started to reach 25gb of RAM storage on a 16 gb machine- The hashmap builder was then starting to run exponentially slower due to the extensive use of swap files)
- fineweb - https://huggingface.co/datasets/HuggingFaceFW/fineweb (sample-10BT)

**Uncompressed and uncleaned data:**
- wikipedia : 115 GB
- fineweb-sample-10BT : 43 GB

**Cleaned data:**
- wikipedia : 18 GB
- fineweb : 42.75 GB

## Strategy

### Data Collection

We train on three diverse, large-scale text corpora to build a general-purpose vocabulary:
- **Stack Overflow** (~104 GB XML dump) — covers programming languages, technical jargon, code snippets, and developer Q&A (Can be used for the dataset but not used in my final training data due to physical system limitations)
- **English Wikipedia** (~116 GB XML dump) — covers formal prose, scientific terminology, historical text, and multilingual proper nouns
- **FineWeb sample-10BT** (sample of 43 GB (10 Billion tokens) from the 800 GB (10 Trillion tokens) dataset) — covers general web text, informal writing, and conversational language

### Data Cleaning

Raw dumps contain XML tags, HTML entities, MediaWiki markup, and metadata that would pollute the vocabulary. Each dataset goes through a dedicated cleaning pipeline (Python scripts using streaming XML parsers) that strips all markup and outputs plain UTF-8 text files. Only actual human-written content is retained.

### Dataset Cleaning Strategy

To prevent unique word inflation and hashmap memory bloat (which can easily exceed RAM limits on raw web data), the text goes through aggressive filtering before frequency building:
- Leading special characters: All punctuation, bullets, dashes, or symbols at the start of a line are stripped.
- Junk sequences: Hexadecimal blobs, base64 data, URLs, and email addresses are entirely removed.
- Repeating characters: Any special character repeated 4 or more times (e.g., long dashes) is collapsed to exactly 3 repetitions and padded with spaces to separate it from surrounding words.
- Pipe limits: Files with excessive pipe characters (like broken markdown tables) have all but the first two pipes replaced with spaces, preserving the pipe token without creating massive concatenated words.
- Normalization: Unicode whitespace is normalized to standard ASCII spaces, and multiple spaces are collapsed into one.
- Frequency Pruning: After building the initial word frequency hashmap, words that appear exactly once across the massive corpus are aggressively pruned. This removes millions of garbage entries (typos, unique hashes, parsing errors) from the long tail of Zipf's law, freeing up over 10GB of RAM for the BPE merge loops without degrading tokenization quality.

### Training Architecture

The tokenizer is trained using a two-phase approach designed to handle 200+ GB of text on a single machine without requiring all data in memory:

**Phase 1 — Word Frequency Extraction (Single Pass):**
Each cleaned text file is streamed line-by-line through the C++ engine exactly once. The engine splits text on whitespace boundaries (preserving leading spaces as part of word tokens, matching GPT-2/tiktoken conventions) and accumulates a global hashmap of `word → frequency`. After this pass, the original text files are no longer needed and can be deleted.

**Phase 2 — BPE Merge Loop (In-Memory):**
Each unique word in the hashmap is decomposed into its individual bytes (the 256 base tokens). The algorithm then iteratively:
1. Counts all adjacent token pairs across every word, weighted by that word's corpus frequency
2. Finds the single most frequent pair
3. Merges that pair into a new token everywhere it appears
4. Records the merge rule

This repeats for a target number of merges (29,744 merges → 30,000 total vocab size). Since this project only covers English general text (no code or multilingual data), a 30k vocabulary provides an optimal balance between compression efficiency and compute cost — similar to Meta's LLaMA 2 (32k vocab). The key insight is that human language is highly repetitive — billions of tokens compress down to millions of unique words, which fit comfortably in RAM.

### Optimised Approach

I originally started training the BPE merges using a naive approach where the CPU recalculates the frequencies of all adjacent byte pairs by scanning every word in the vocabulary on every single merge step. However, with millions of unique words and 30,000 target merges, I soon realized this was going to take days (over 4 trillion operations!). So, I implemented this new optimised approach using a Priority Queue (Max-Heap) and a Reverse Index.

**How it works:**
Instead of scanning all words, the algorithm counts pairs exactly once at the beginning and places them into a Max-Heap to instantly find the most frequent pair. A Reverse Index maps every byte pair to the exact IDs of the words that contain it. When a merge happens, the algorithm only updates the specific words flagged by the Reverse Index, taking the time complexity from days down to a few minutes (under 1 billion operations).

**Example in a 15-word space:**
Imagine a tiny dataset of 15 words where 5 are `c a b`, 3 are `d a b`, and 7 are `c a d`. 
1. **Initial Scan:** The pair `(a, b)` appears 8 times. `(c, a)` appears 12 times. The Reverse Index remembers which words have which pairs.
2. **The Merge:** The heap pops the most frequent pair: `(c, a)`. 
3. **The Update:** The Reverse Index instantly points us only to the 12 words containing `(c, a)`. We merge `(c, a)` into `ca` for those words. 
4. **Delta Math:** Because `a` is no longer standing alone, overlapping pairs like `(a, b)` and `(a, d)` are mathematically reduced in frequency, and new pairs like `(ca, b)` and `(ca, d)` are added to the heap. The other 3 words (`d a b`) are entirely ignored, saving massive amounts of compute time!

### Output

The trained tokenizer produces two files:
- **merges.ddtok** — Ordered list of merge rules (the sequence matters for deterministic tokenization)
- **vocab.ddtok** — Complete token-to-ID mapping (256 base bytes + all merged tokens)

These files fully define the tokenizer and can be loaded to encode any arbitrary text into token IDs.

*NOTE* - this is the current strategy and might change as the codebase progresses
