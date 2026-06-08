/*
 * tokenizer.h — BPETokenizer class declaration
 *
 * What this file does:
 *   Declares the interface and data structures for a byte-level BPE tokenizer.
 *   This includes the word frequency map, per-word token splits, merge history,
 *   and the custom hash needed to key on token pairs in an unordered_map.
 *
 * Where it gets its data:
 *   Nothing directly — this is a header. The implementation lives in tokenizer.cpp.
 *   Data flows in from cleaned text files (stackoverflow_clean.txt, wikipedia_clean.txt,
 *   or any plain text) via the build_word_frequencies() method.
 *
 * Who consumes its output:
 *   main.cpp includes this header to instantiate and drive the tokenizer.
 *   tokenizer.cpp includes this header to implement the declared methods.
 *
 * Why this file exists:
 *   Separating declarations from implementation lets main.cpp know *what* the
 *   tokenizer can do without pulling in *how* it does it. Standard C++ practice
 *   for keeping compile times sane and interfaces clean.
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <utility>

using std::string;
using std::vector;
using std::pair;
using std::unordered_map;
using std::istream;

using TokenPair = pair<string, string>;

// Hash for TokenPair so we can use it as an unordered_map key.
// Uses Fibonacci hashing to reduce collision clustering.
struct PairHash {
    size_t operator()(const TokenPair& p) const {
        size_t h1 = std::hash<string>{}(p.first);
        size_t h2 = std::hash<string>{}(p.second);
        return h1 ^ (h2 * 2654435761ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

using PairCounts = unordered_map<TokenPair, size_t, PairHash>;

class BPETokenizer {
public:
    // 256 raw byte values form the base vocabulary before any merges
    static constexpr size_t BASE_VOCAB_SIZE = 256;

    // Ingest text from a stream and accumulate word frequencies.
    // Safe to call multiple times across different files — counts accumulate.
    void build_word_frequencies(istream& input);

    // Run `num_merges` BPE iterations on the accumulated word data.
    // After this, merges + BASE_VOCAB_SIZE = total vocab size.
    void train(size_t num_merges);

    // Persist merge rules and final vocabulary to disk.
    void save(const string& merges_path, const string& vocab_path) const;

    void print_stats() const;

    // Serialize word frequency hashmap to a binary .ddfreq file.
    // Format: [count][len1][word1][freq1][len2][word2][freq2]...
    void save_word_freqs(const string& path) const;

    // Load word frequency hashmap from a binary .ddfreq file.
    // Replaces any existing word_freqs data.
    void load_word_freqs(const string& path);

    // Prune words with frequency less than min_freq
    void prune_word_freqs(size_t min_freq);

private:
    // word -> corpus frequency (survives entire lifetime, never mutated after ingestion)
    unordered_map<string, size_t> word_freqs;

    // word -> current token decomposition
    // Starts as individual bytes, tokens grow as merges are applied.
    unordered_map<string, vector<string>> word_splits;

    // Ordered merge history — this IS the learned tokenizer
    vector<TokenPair> merges;

    // Walk every word's token list, count adjacent pairs weighted by word frequency
    PairCounts count_pairs() const;

    // Replace all (a, b) adjacencies with (a+b) across every word split
    void apply_merge(const string& a, const string& b);
};
