#pragma once

#include <string>
#include <unordered_map>
#include <iostream>

using std::string;
using std::unordered_map;
using std::istream;

class BPETokenizer {
public:
    unordered_map<string, size_t> word_counts;

    void train_from_stream(istream& in_stream);
    const unordered_map<string, size_t>& get_word_counts() const;
};
