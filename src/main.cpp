/*
 * main.cpp — CLI entry point for ddtokens BPE trainer
 *
 * What this file does:
 *   Orchestrates the full tokenizer training pipeline:
 *   1. Accepts command-line args: number of merges + one or more text file paths
 *   2. Streams each file through BPETokenizer::build_word_frequencies() (first pass)
 *   3. Triggers in-memory BPE training via BPETokenizer::train()
 *   4. Saves the learned merge rules and vocabulary to disk
 *
 * Where it gets its data:
 *   Cleaned plain-text files in datasets/ (produced by the Python cleaning scripts).
 *   Files are read once during the first pass, then never accessed again.
 *
 * Who consumes its output:
 *   Produces merges.ddtok and vocab.ddtok in the working directory.
 *   These define the complete tokenizer and will eventually be used by an
 *   encoder/decoder module to convert arbitrary text into token IDs.
 *
 * Why this file exists:
 *   Separates CLI concerns (arg parsing, file I/O, error handling) from the
 *   BPE algorithm itself. Keeps tokenizer.cpp reusable without CLI coupling.
 *
 * Usage:
 *   ./tokenizer <num_merges> <file1> [file2] ...
 *   ./tokenizer 100000 datasets/stackoverflow_clean.txt datasets/wikipedia_clean.txt
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "../include/tokenizer.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: tokenizer <num_merges> <file1> [file2] ...\n"
             << "  num_merges: how many BPE merges to learn (e.g. 100000)\n"
             << "  file1, file2, ...: cleaned text files to train on\n\n"
             << "Example:\n"
             << "  ./tokenizer 100000 datasets/stackoverflow_clean.txt datasets/wikipedia_clean.txt\n";
        return 1;
    }

    size_t num_merges = stoull(argv[1]);

    BPETokenizer tokenizer;

    // First pass — stream each file once to build word frequency table.
    // After this loop, the raw text files are never touched again.
    for (int i = 2; i < argc; i++) {
        cout << "Reading: " << argv[i] << "\n";

        ifstream file(argv[i]);
        if (!file.is_open()) {
            cerr << "Error: could not open " << argv[i] << "\n";
            return 1;
        }

        tokenizer.build_word_frequencies(file);
    }

    tokenizer.print_stats();

    // All file data is now compressed into the frequency map.
    // BPE runs entirely in-memory from here.
    tokenizer.train(num_merges);

    // Persist the learned tokenizer
    tokenizer.save("merges.ddtok", "vocab.ddtok");

    return 0;
}
