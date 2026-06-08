#include "../include/tokenizer_naive.h"
#include <iostream>

using namespace std;

// --- BPE Training (runs entirely in-memory) ---

void BPETokenizerNaive::train(size_t num_merges) {
    // Initialize word_splits: decompose each word into individual bytes.
    // This only happens once — subsequent merges mutate these vectors in place.
    if (word_splits.empty()) {
        word_splits.reserve(word_freqs.size());
        for (const auto& [word, freq] : word_freqs) {
            vector<string> bytes;
            bytes.reserve(word.size());
            for (unsigned char c : word) {
                bytes.emplace_back(1, static_cast<char>(c));
            }
            word_splits[word] = std::move(bytes);
        }
    }

    cout << "Starting Naive BPE: " << word_freqs.size() << " unique words, "
         << num_merges << " target merges\n";

    for (size_t step = 0; step < num_merges; step++) {
        PairCounts pair_counts = count_pairs();

        if (pair_counts.empty()) {
            cout << "No mergeable pairs remain at step " << step << "\n";
            break;
        }

        // Linear scan for max — fine for typical vocab sizes.
        // A priority queue would help if pair_counts is huge,
        // but the bottleneck is count_pairs(), not this scan.
        TokenPair best;
        size_t best_count = 0;
        for (const auto& [pair, count] : pair_counts) {
            if (count > best_count) {
                best_count = count;
                best = pair;
            }
        }

        merges.push_back(best);
        apply_merge(best.first, best.second);

        // Log every merge
        cout << "  merge " << (step + 1) << "/" << num_merges
             << ": \"" << best.first << "\" + \"" << best.second
             << "\" (count=" << best_count << ")\n";
    }

    cout << "Training complete. Final vocab size: "
         << BASE_VOCAB_SIZE + merges.size() << "\n";
}

PairCounts BPETokenizerNaive::count_pairs() const {
    PairCounts counts;

    for (const auto& [word, split] : word_splits) {
        if (split.size() < 2) continue;

        size_t freq = word_freqs.at(word);
        for (size_t i = 0; i + 1 < split.size(); i++) {
            counts[{split[i], split[i + 1]}] += freq;
        }
    }

    return counts;
}

void BPETokenizerNaive::apply_merge(const string& a, const string& b) {
    string merged = a + b;

    for (auto& [word, split] : word_splits) {
        if (split.size() < 2) continue;

        vector<string> new_split;
        new_split.reserve(split.size());

        size_t i = 0;
        while (i < split.size()) {
            if (i + 1 < split.size() && split[i] == a && split[i + 1] == b) {
                new_split.push_back(merged);
                i += 2;
            } else {
                new_split.push_back(std::move(split[i]));
                i++;
            }
        }

        split = std::move(new_split);
    }
}
