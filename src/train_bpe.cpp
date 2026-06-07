/*
 * train_bpe.cpp — Phase 2: Load word frequencies and run BPE training
 *
 * What this file does:
 *   Loads a pre-built word frequency hashmap from a .ddfreq file,
 *   runs the specified number of BPE merge iterations, and writes
 *   the learned merge rules and vocabulary to disk.
 *
 * Where it gets its data:
 *   A .ddfreq binary file produced by build_freqs. No access to the
 *   original text files is needed.
 *
 * Who consumes its output:
 *   Produces merges.ddtok (ordered merge rules) and vocab.ddtok
 *   (full token-to-id mapping). These define the complete tokenizer.
 *
 * Why this file exists:
 *   Decouples BPE training from corpus ingestion. Once the hashmap is
 *   built, you can experiment with different merge counts instantly
 *   without re-streaming the raw text.
 *
 * Usage:
 *   ./train_bpe <num_merges> <word_freqs.ddfreq>
 *   ./train_bpe 100000 word_freqs.ddfreq
 */

#include <iostream>
#include <cstdlib>
#include "../include/tokenizer.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: train_bpe <num_merges> <word_freqs.ddfreq>\n"
             << "  num_merges: how many BPE merges to learn (e.g. 100000)\n"
             << "  word_freqs.ddfreq: binary frequency map from build_freqs\n\n"
             << "Example:\n"
             << "  ./train_bpe 100000 word_freqs.ddfreq\n";
        return 1;
    }

    size_t num_merges = stoull(argv[1]);
    string freqs_path = argv[2];

    BPETokenizer tokenizer;

    // Load the pre-built hashmap — no raw text files needed.
    tokenizer.load_word_freqs(freqs_path);
    tokenizer.print_stats();

    // BPE training runs entirely in-memory.
    tokenizer.train(num_merges);

    // Persist the learned tokenizer.
    tokenizer.save("merges.ddtok", "vocab.ddtok");

    return 0;
}
