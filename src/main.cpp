#include <iostream>
#include <unordered_map>
#include "../include/tokenizer.h"

using namespace std;

int main() {
    BPETokenizer tokenizer;
    
    // Read from standard input (e.g., piped from the Python download script)
    tokenizer.train_from_stream(cin);
    
    const auto& counts = tokenizer.get_word_counts();
    
    cout << "Unique tokens: " << counts.size() << "\n\n";
    
    for (const auto& [word, freq] : counts) {
        string display_word = word;
        size_t pos = 0;
        
        // Escape newlines for stdout visualization
        while ((pos = display_word.find('\n', pos)) != string::npos) {
            display_word.replace(pos, 1, "\\n");
            pos += 2;
        }
        
        cout << "[\"" << display_word << "\"] : " << freq << "\n";
    }
    
    return 0;
}
