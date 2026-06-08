/*
 * build_freqs.cpp — Phase 1: Build and serialize word frequency hashmap
 *
 * What this file does:
 *   Streams one or more text sources and builds a word → frequency hashmap,
 *   then serializes it to a binary .ddfreq file on disk.
 *   Supports both local files and stdin (use "-" to read from a pipe).
 *
 * Where it gets its data:
 *   - Local cleaned text files in datasets/
 *   - Piped stdin from a streaming script (e.g. streaming FineWeb from HuggingFace)
 *   Sources can be mixed in a single run — frequencies accumulate across all inputs.
 *
 * Who consumes its output:
 *   Produces a .ddfreq file that train_bpe consumes to run BPE merges
 *   without needing access to the original text sources.
 *
 * Why this file exists:
 *   Decouples the expensive I/O-bound first pass from the CPU-bound BPE training.
 *   Once the hashmap is built and saved, you can iterate on merge counts without
 *   re-reading or re-streaming the corpus.
 *
 * Usage:
 *   ./build_freqs <output.ddfreq> <file1 | -> [file2 | -] ...
 *
 *   # Local files only
 *   ./build_freqs word_freqs.ddfreq datasets/stackoverflow_clean.txt datasets/wikipedia_clean.txt
 *
 *   # Stream from stdin (e.g. FineWeb via Python)
 *   python scripts/stream_fineweb.py | ./build_freqs word_freqs.ddfreq -
 *
 *   # Mix local files and stdin
 *   python scripts/stream_fineweb.py | ./build_freqs word_freqs.ddfreq datasets/wikipedia_clean.txt -
 */

#include <iostream>
#include <fstream>
#include "../include/tokenizer_base.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: build_freqs <output.ddfreq> <file1 | -> [file2 | -] ...\n"
             << "  output.ddfreq: path to write the binary word frequency map\n"
             << "  file1, file2, ...: cleaned text files to ingest\n"
             << "  -: read from stdin (for piped/streamed input)\n\n"
             << "Examples:\n"
             << "  ./build_freqs word_freqs.ddfreq datasets/stackoverflow_clean.txt\n"
             << "  python stream_fineweb.py | ./build_freqs word_freqs.ddfreq -\n";
        return 1;
    }

    string output_path = argv[1];
    BPETokenizerBase tokenizer;

    // Process each source in order — frequencies accumulate across all inputs.
    for (int i = 2; i < argc; i++) {
        string source = argv[i];

        if (source == "-") {
            // Read from stdin — typically piped from a streaming script.
            cout << "Reading: stdin (streaming)\n";

            if (cin.eof()) {
                cerr << "Warning: stdin is empty or already consumed\n";
                continue;
            }

            tokenizer.build_word_frequencies(cin);
        } else {
            // Read from a local file.
            cout << "Reading: " << source << "\n";

            ifstream file(source);
            if (!file.is_open()) {
                cerr << "Error: could not open " << source << "\n";
                return 1;
            }

            tokenizer.build_word_frequencies(file);
        }
    }

    tokenizer.print_stats();

    // Persist the hashmap — raw text sources are never needed again.
    tokenizer.save_word_freqs(output_path);

    return 0;
}

