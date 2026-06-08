/*
 * train_bpe_heap.cpp — CLI tool for fast BPE training
 *
 * What this file does:
 *   This is the entry point for running the optimized heap-based BPE training.
 *   It takes the target number of merges and the path to the pruned .ddfreq file
 *   as command-line arguments, runs the fast O(N log N) training algorithm,
 *   and saves the resulting merges.ddtok and vocab.ddtok files.
 */
#include <iostream>
#include <string>
#include <cstdlib>
#include "../include/tokenizer.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: train_bpe_heap <num_merges> <word_freqs.ddfreq>\n"
             << "  num_merges: how many BPE merges to learn (e.g. 30000)\n"
             << "  word_freqs.ddfreq: binary frequency map from build_freqs\n\n"
             << "Example:\n"
             << "  ./train_bpe_heap 30000 word_freqs.ddfreq\n";
        return 1;
    }

    size_t num_merges = std::stoull(argv[1]);
    string freqs_path = argv[2];

    BPETokenizer tokenizer;
    
    // Phase 1: Load pruned hashmap
    tokenizer.load_word_freqs(freqs_path);
    
    // Phase 2: Train BPE rules using fast algorithm
    tokenizer.train_heap(num_merges);
    
    // Phase 3: Save to disk
    tokenizer.save("merges.ddtok", "vocab.ddtok");

    return 0;
}
