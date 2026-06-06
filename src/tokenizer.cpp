#include "../include/tokenizer.h"
#include <cctype>

using namespace std;

void BPETokenizer::train_from_stream(istream& in_stream) {
    string line;
    
    while (getline(in_stream, line)) {
        string current_word;
        // Pre-allocate to minimize reallocations
        current_word.reserve(line.size());
        
        for (char c : line) {
            if (isspace(c)) {
                if (!current_word.empty()) {
                    word_counts[current_word]++;
                    current_word.clear();
                }
                // Prepend space to the subsequent word token, 
                // aligning with standard byte-level BPE behavior.
                current_word += c;
            } else {
                current_word += c;
            }
        }
        
        if (!current_word.empty()) {
            word_counts[current_word]++;
        }
        
        // Account for dropped newline from getline
        word_counts["\n"]++;
    }
}

const unordered_map<string, size_t>& BPETokenizer::get_word_counts() const {
    return word_counts;
}
