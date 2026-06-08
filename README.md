# ddtokens

creating a tokenizer (byte level bpe) and possibly trying to replicate cl100k_base's massive token dictionary using an optimised fast heap-based approach (introduced in 2023) to run BPE algorithm training significantly faster (about 100x) than the naive loop-through approach.

avoid the detailed read - check out merges.ddtok and vocab.ddtok to see the generated vocabulary and merging rules

## Table of Contents
- [Papers](#papers)
- [Datasets](#datasets)
- [Strategy](#strategy)
  - [Data Collection](#data-collection)
  - [Data Cleaning](#data-cleaning)
  - [Dataset Cleaning Strategy](#dataset-cleaning-strategy)
  - [Training Architecture](#training-architecture)
  - [Optimised Approach](#optimised-approach)
    - [Critical Problems Addressed](#critical-problems-addressed)
    - [The "Vanishing Pair" Bug (The Biggest Heap Pitfall)](#the-vanishing-pair-bug-the-biggest-heap-pitfall)
  - [Output](#output)

## Papers


* **Neural Machine Translation of Rare Words with Subword Units (2016)** – Introduced BPE to NLP for subword tokenization, enabling models to handle rare and unseen words without a fixed word-level vocabulary.

  * Paper: [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)

* **Language Models are Unsupervised Multitask Learners (GPT-2, 2019)** – Introduced Byte-Level BPE (BBPE), applying BPE directly to UTF-8 bytes instead of characters, eliminating unknown tokens and enabling robust tokenization of arbitrary text.

  * Paper: [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

* **Training Multilingual Pre-trained Language Model with Byte-level Subwords (2021)** – Demonstrates the effectiveness of Byte-Level BPE for multilingual language models, showing improved vocabulary sharing across languages while maintaining strong downstream performance.

  * Paper: [https://arxiv.org/abs/2101.09469](https://arxiv.org/abs/2101.09469)

* **A Formal Perspective on Byte-Pair Encoding (2023)** – Rather than treating BPE as a heuristic, the paper formalizes it as a combinatorial optimization problem and derives theoretical guarantees for its behavior. Improves runtime complexity from **O(NM)** to **O(N log M)**, where **N** is sequence length and **M** is the number of merge operations.

  * Paper: [https://arxiv.org/pdf/2306.16837](https://arxiv.org/pdf/2306.16837)



 

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

#### Critical Problems Addressed
Implementing the $O(N \log N)$ heap approach introduces several edge cases that this codebase explicitly solves:
- **Stale Entries in the Heap:** Since you cannot efficiently remove pairs mid-heap when their frequencies decrease, the codebase uses **Lazy Deletion**. When a pair is popped, the algorithm verifies its heap frequency against a ground-truth `current_pair_counts` hashmap. If they don't match, it is discarded as stale.
- **Word Frequency Weighting:** Pair counts are not incremented by `1`. They are strictly weighted by the parent word's frequency in the massive corpus (e.g., `pair_count += word_freq`).
- **Neighbor Pair Delta Math:** When merging `A` and `B` into `AB`, the left and right neighbor boundaries are dynamically updated. `(X, A)` and `(B, Y)` are decremented, while `(X, AB)` and `(AB, Y)` are incremented and pushed to the heap.
- **Word Representation:** Words are not stored as strings (which require costly string splits). They are stored as a `std::vector<std::string>` to allow extremely fast $O(L)$ linear array rebuilds when tokens merge.
- **End-of-Word Tokens:** Instead of appending artificial `</w>` tokens to word boundaries, this algorithm strictly follows the GPT-2 / tiktoken byte-level standard. Whitespace is preserved as a leading byte (`" hello"` vs `"hello"`), acting as a natural boundary.
- **O(N) Heap Initialization:** The initial Max-Heap is constructed by passing an entire `std::vector` of pairs directly to the `std::priority_queue` constructor. Under the hood, this uses `std::make_heap` to build the structure in pure $O(N)$ linear time, which is strictly over 2x faster than inserting elements one-by-one in $O(N \log N)$ time.

#### The "Vanishing Pair" Bug (The Biggest Heap Pitfall)
When implementing Lazy Deletion, there is a massive logical trap that initially broke the statistical accuracy of this algorithm.

**What it was doing:**
When a pair merges (e.g., merging `h` and `e`), it naturally destroys overlapping pairs (like `e` and `r` in the word `h e r e`). Because `(e, r)` is destroyed, its true frequency mathematically decreases. 
The algorithm correctly identified that the old, higher frequency for `(e, r)` in the priority queue was now "stale". When that stale entry reached the top of the heap, the Lazy Deletion check successfully threw it in the garbage. However, the algorithm **failed to push the new, lower frequency back into the priority queue.** 
As a result, the pair `(e, r)` completely vanished from the heap and was lost forever, causing major frequency drift and incorrect merge choices later down the line.

**What it should do (The Fix):**
Every single time a neighbor pair's frequency is decremented due to an overlap destruction, that new lower frequency **must** be explicitly pushed back into the heap.

**Example:**
Imagine the pair `(e, r)` has a massive frequency of 485 million.
1. The algorithm merges `(h, e)` into `he`.
2. In words like `h e r e`, the `e` is consumed, destroying the `(e, r)` pair.
3. We decrement the true count of `(e, r)` down to 460 million.
4. The heap still contains the old `(485M, (e, r))` entry.
5. **The Fix:** We immediately push `(460M, (e, r))` into the heap.
6. Later, the heap pops the 485M entry. Lazy Deletion sees that 485M != 460M, and throws the 485M entry away.
7. Eventually, the heap pops the 460M entry. Lazy Deletion sees that 460M == 460M, and successfully merges it!

### Output

The trained tokenizer produces two files:
- **merges.ddtok** — Ordered list of merge rules (the sequence matters for deterministic tokenization)
- **vocab.ddtok** — Complete token-to-ID mapping (256 base bytes + all merged tokens)

These files fully define the tokenizer and can be loaded to encode any arbitrary text into token IDs.

*NOTE* - this is the current strategy and might change as the codebase progresses
