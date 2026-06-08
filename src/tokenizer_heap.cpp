/*
 * tokenizer_heap.cpp — Optimized Heap-Based BPE engine implementation
 *
 * What this file does:
 *   Implements a significantly faster version of BPE training using a Priority Queue
 *   (Max-Heap) and a Reverse Index. Instead of scanning all words on every merge
 *   (O(V * N)), this approach calculates pair frequencies once, pushes them to a heap,
 *   and updates only the exact words affected by a merge via the reverse index.
 *   This brings the time complexity down from days to minutes.
 *
 * Where it gets its data:
 *   Operates entirely in-memory using the word_freqs loaded into the BPETokenizer object.
 *
 * Who consumes its output:
 *   train_bpe_heap.cpp calls train() and then saves the resulting merges and vocab.
 */
#include "../include/tokenizer_heap.h"
#include <queue>
#include <chrono>

using namespace std;

struct WordData {
    size_t freq;
    vector<string> split;
};

void BPETokenizerHeap::train(size_t num_merges) {
    cout << "Starting heap BPE training...\n";
    auto start_time = chrono::high_resolution_clock::now();

    vector<WordData> word_data;
    word_data.reserve(word_freqs.size());

    // 1. Initialize WordData from word_freqs
    // We also keep track of keys so we can update word_splits at the end.
    vector<string> word_keys;
    word_keys.reserve(word_freqs.size());

    for (const auto& [word, freq] : word_freqs) {
        word_keys.push_back(word);
        vector<string> bytes;
        bytes.reserve(word.size());
        for (unsigned char c : word) {
            bytes.emplace_back(1, static_cast<char>(c));
        }
        word_data.push_back({freq, std::move(bytes)});
    }

    // 2. Initialize Reverse Index and Pair Counts
    PairCounts current_pair_counts;
    unordered_map<TokenPair, vector<size_t>, PairHash> pair_locs;

    for (size_t w_idx = 0; w_idx < word_data.size(); ++w_idx) {
        const auto& split = word_data[w_idx].split;
        size_t freq = word_data[w_idx].freq;
        if (split.size() < 2) continue;

        for (size_t i = 0; i + 1 < split.size(); ++i) {
            TokenPair p = {split[i], split[i+1]};
            current_pair_counts[p] += freq;
            if (pair_locs[p].empty() || pair_locs[p].back() != w_idx) {
                pair_locs[p].push_back(w_idx);
            }
        }
    }

    // 3. Initialize Priority Queue
    using HeapItem = pair<size_t, TokenPair>;
    auto cmp = [](const HeapItem& a, const HeapItem& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second > b.second; // Deterministic tie-breaker
    };
    vector<HeapItem> initial_heap;
    initial_heap.reserve(current_pair_counts.size());
    for (const auto& [p, count] : current_pair_counts) {
        initial_heap.push_back({count, p});
    }
    
    // Pass the vector directly to the priority_queue constructor.
    // This internally calls std::make_heap in O(P) linear time, which is 
    // significantly faster than pushing items one-by-one in O(P log P) time.
    priority_queue<HeapItem, vector<HeapItem>, decltype(cmp)> pq(cmp, std::move(initial_heap));

    cout << "Initialization complete. Target merges: " << num_merges << "\n";

    // 4. Fast Merge Loop
    for (size_t step = 0; step < num_merges; ++step) {
        TokenPair best;
        size_t best_count = 0;

        // Lazy pop: throw away stale elements
        while (!pq.empty()) {
            auto top = pq.top();
            pq.pop();
            if (top.first == current_pair_counts[top.second]) {
                best = top.second;
                best_count = top.first;
                break;
            }
        }

        if (best_count == 0) {
            cout << "No mergeable pairs remain at step " << step << "\n";
            break;
        }

        merges.push_back(best);
        string merged = best.first + best.second;

        // Apply merge
        vector<size_t>& affected_words = pair_locs[best];
        for (size_t w_idx : affected_words) {
            auto& split = word_data[w_idx].split;
            size_t freq = word_data[w_idx].freq;

            vector<string> new_split;
            new_split.reserve(split.size());

            size_t i = 0;
            bool changed = false;
            while (i < split.size()) {
                if (i + 1 < split.size() && split[i] == best.first && split[i + 1] == best.second) {
                    changed = true;
                    
                    // Destroy pair before the merged tokens
                    if (new_split.size() > 0) {
                        TokenPair prev_pair = {new_split.back(), split[i]};
                        current_pair_counts[prev_pair] -= freq;
                        pq.push({current_pair_counts[prev_pair], prev_pair});
                        
                        TokenPair new_prev = {new_split.back(), merged};
                        current_pair_counts[new_prev] += freq;
                        pq.push({current_pair_counts[new_prev], new_prev});
                        if (pair_locs[new_prev].empty() || pair_locs[new_prev].back() != w_idx) {
                            pair_locs[new_prev].push_back(w_idx);
                        }
                    }
                    
                    // Destroy pair after the merged tokens
                    if (i + 2 < split.size()) {
                        TokenPair next_pair = {split[i + 1], split[i + 2]};
                        current_pair_counts[next_pair] -= freq;
                        pq.push({current_pair_counts[next_pair], next_pair});
                        
                        TokenPair new_next = {merged, split[i + 2]};
                        current_pair_counts[new_next] += freq;
                        pq.push({current_pair_counts[new_next], new_next});
                        if (pair_locs[new_next].empty() || pair_locs[new_next].back() != w_idx) {
                            pair_locs[new_next].push_back(w_idx);
                        }
                    }
                    
                    new_split.push_back(merged);
                    i += 2;
                } else {
                    new_split.push_back(split[i]);
                    i++;
                }
            }
            if (changed) {
                split = std::move(new_split);
            }
        }

        // Free memory for the old pair
        pair_locs[best].clear();
        pair_locs[best].shrink_to_fit();
        
        // Zero out the best pair so it can never accidentally be popped again
        current_pair_counts[best] = 0;

        // Log every merge
        cout << "  fast merge " << (step + 1) << "/" << num_merges
             << ": \"" << best.first << "\" + \"" << best.second
             << "\" (count=" << best_count << ")\n";
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end_time - start_time;

    cout << "Fast training complete. Final vocab size: " << BASE_VOCAB_SIZE + merges.size() << "\n";
    cout << "Time elapsed: " << diff.count() << " seconds\n";

    // 5. Transfer to word_splits so save() functions normally
    word_splits.clear();
    for (size_t w_idx = 0; w_idx < word_data.size(); ++w_idx) {
        word_splits[word_keys[w_idx]] = std::move(word_data[w_idx].split);
    }
}
