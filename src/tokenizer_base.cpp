/*
 * tokenizer.cpp — Core BPE engine implementation
 *
 * What this file does:
 *   Implements the full byte-level BPE training pipeline in two phases:
 *
 *   Phase 1 (build_word_frequencies):
 *     Streams cleaned text files line-by-line and builds a hashmap of
 *     word → corpus frequency. Whitespace is treated as a leading byte
 *     of the subsequent word (" hello" vs "hello") to match GPT-2/tiktoken
 *     conventions. This is the only phase that touches disk.
 *
 *   Phase 2 (train):
 *     Decomposes each unique word into its raw bytes, then iteratively
 *     finds the most frequent adjacent byte pair across all words (weighted
 *     by word frequency), merges that pair, and records the merge rule.
 *     Runs entirely in-memory — no file access after phase 1.
 *
 * Where it gets its data:
 *   Cleaned text files produced by scripts/clean_stackoverflow.py and
 *   scripts/clean_wikipedia.py, passed in as istream references from main.cpp.
 *
 * Who consumes its output:
 *   save() writes two files: merges.ddtok (ordered merge rules) and
 *   vocab.ddtok (full token-to-id mapping). These files define the
 *   learned tokenizer and can be loaded later for encoding/decoding text.
 *
 * Why this file exists:
 *   Keeps the BPE algorithm isolated from the CLI logic in main.cpp.
 *   The tokenizer class is self-contained and testable independently.
 */

#include "../include/tokenizer_base.h"
#include <cctype>
#include <iomanip>
#include <sstream>

using namespace std;

// --- First Pass: Streaming Word Frequency Builder ---

void BPETokenizerBase::build_word_frequencies(istream& input) {
    string line;
    size_t lines_read = 0;

    while (getline(input, line)) {
        string word;
        word.reserve(64);

        for (char c : line) {
            if (isspace(c)) {
                if (!word.empty()) {
                    word_freqs[word]++;
                    word.clear();
                }
                // Leading space becomes part of the next word.
                // This is how GPT-2/tiktoken handle word boundaries —
                // " hello" and "hello" are distinct vocabulary entries.
                word += c;
            } else {
                word += c;
            }
        }

        if (!word.empty()) {
            word_freqs[word]++;
        }

        // getline strips \n, but it's a real byte in the corpus
        word_freqs["\n"]++;
        lines_read++;

        if (lines_read % 5000000 == 0) {
            cout << "  ingested " << lines_read / 1000000 << "M lines | "
                 << word_freqs.size() << " unique words\n";
        }
    }

    cout << "  finished: " << lines_read << " lines, "
         << word_freqs.size() << " unique words\n";
}



// --- Output ---

// Escape a raw token string into a printable representation.
// Nonprintable bytes become \xHH, backslash becomes \\.
static string escape_token(const string& token) {
    ostringstream oss;
    for (unsigned char c : token) {
        if (c == '\\') {
            oss << "\\\\";
        } else if (c == '\n') {
            oss << "\\n";
        } else if (c == '\t') {
            oss << "\\t";
        } else if (c >= 0x20 && c < 0x7F) {
            oss << static_cast<char>(c);
        } else {
            oss << "\\x" << hex << setfill('0') << setw(2) << (int)c;
        }
    }
    return oss.str();
}

void BPETokenizerBase::save(const string& merges_path, const string& vocab_path) const {
    // Write merge rules (ordered — order matters for deterministic tokenization)
    {
        ofstream f(merges_path);
        f << "# ddtokens merge rules\n";
        f << "# total merges: " << merges.size() << "\n";
        for (const auto& [a, b] : merges) {
            f << escape_token(a) << "\t" << escape_token(b) << "\n";
        }
        cout << "Merge rules written to " << merges_path << "\n";
    }

    // Write full vocabulary: 256 base bytes + all merged tokens
    {
        ofstream f(vocab_path);
        size_t id = 0;

        // Base vocabulary: all 256 single-byte tokens
        for (int b = 0; b < 256; b++) {
            string token(1, static_cast<char>(b));
            f << id++ << "\t" << escape_token(token) << "\n";
        }

        // Merged tokens in order of creation
        for (const auto& [a, b] : merges) {
            f << id++ << "\t" << escape_token(a + b) << "\n";
        }

        cout << "Vocabulary (" << id << " tokens) written to " << vocab_path << "\n";
    }
}

void BPETokenizerBase::print_stats() const {
    cout << "Word frequencies: " << word_freqs.size() << " unique words\n";
    cout << "Merges learned: " << merges.size() << "\n";
    cout << "Current vocab size: " << BASE_VOCAB_SIZE + merges.size() << "\n";
}

// --- Word Frequency Serialization ---

void BPETokenizerBase::save_word_freqs(const string& path) const {
    ofstream f(path, ios::binary);
    if (!f.is_open()) {
        cerr << "Error: could not open " << path << " for writing\n";
        return;
    }

    uint64_t count = word_freqs.size();
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& [word, freq] : word_freqs) {
        uint64_t len = word.size();
        uint64_t freq_val = freq;
        f.write(reinterpret_cast<const char*>(&len), sizeof(len));
        f.write(word.data(), len);
        f.write(reinterpret_cast<const char*>(&freq_val), sizeof(freq_val));
    }

    cout << "Word frequencies (" << count << " entries) saved to " << path << "\n";
}

void BPETokenizerBase::load_word_freqs(const string& path) {
    ifstream f(path, ios::binary);
    if (!f.is_open()) {
        cerr << "Error: could not open " << path << " for reading\n";
        return;
    }

    word_freqs.clear();

    uint64_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));

    word_freqs.reserve(count);

    for (uint64_t i = 0; i < count; i++) {
        uint64_t len = 0;
        f.read(reinterpret_cast<char*>(&len), sizeof(len));

        string word(len, '\0');
        f.read(word.data(), len);

        uint64_t freq = 0;
        f.read(reinterpret_cast<char*>(&freq), sizeof(freq));

        word_freqs[std::move(word)] = freq;
    }

    cout << "Loaded " << word_freqs.size() << " word frequencies from " << path << "\n";
}

void BPETokenizerBase::prune_word_freqs(size_t min_freq) {
    size_t original_size = word_freqs.size();
    for (auto it = word_freqs.begin(); it != word_freqs.end(); ) {
        if (it->second < min_freq) {
            it = word_freqs.erase(it);
        } else {
            ++it;
        }
    }
    cout << "Pruned " << (original_size - word_freqs.size()) << " words.\n";
    cout << "Kept " << word_freqs.size() << " words out of " << original_size << ".\n";
}
