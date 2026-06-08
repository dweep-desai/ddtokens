#include <iostream>
#include <string>
#include "../include/tokenizer_base.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: prune_hashmap <input.ddfreq> <output.ddfreq> <min_freq>\n";
        return 1;
    }

    string input_path = argv[1];
    string output_path = argv[2];
    size_t min_freq = stoull(argv[3]);

    BPETokenizerBase tokenizer;
    cout << "Loading hashmap from " << input_path << "...\n";
    tokenizer.load_word_freqs(input_path);
    
    cout << "Pruning words with frequency < " << min_freq << "...\n";
    tokenizer.prune_word_freqs(min_freq);
    
    cout << "Saving pruned hashmap to " << output_path << "...\n";
    tokenizer.save_word_freqs(output_path);

    return 0;
}
