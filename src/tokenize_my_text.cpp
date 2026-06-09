#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include "../include/tokenizer_base.h"

using namespace std;

string unescape_token(const string& s) {
    string out;
    for (size_t i = 0; i < s.length(); i++) {
        if (s[i] == '\\' && i + 1 < s.length()) {
            if (s[i+1] == '\\') { out += '\\'; i++; }
            else if (s[i+1] == 'n') { out += '\n'; i++; }
            else if (s[i+1] == 't') { out += '\t'; i++; }
            else if (s[i+1] == 'x' && i + 3 < s.length()) {
                string hex_str = s.substr(i+2, 2);
                out += (char)std::stoi(hex_str, nullptr, 16);
                i += 3;
            } else {
                out += s[i];
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

int main() {
    // 1. Load merges
    unordered_map<TokenPair, int, PairHash> merge_ranks;
    ifstream fm("merges.ddtok");
    if (!fm.is_open()) {
        cerr << "Error: Could not open merges.ddtok. Please run this command from the project root." << endl;
        return 1;
    }
    string line;
    int rank = 0;
    while (getline(fm, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t tab = line.find('\t');
        if (tab != string::npos) {
            string a = unescape_token(line.substr(0, tab));
            string b = unescape_token(line.substr(tab + 1));
            merge_ranks[{a, b}] = rank++;
        }
    }
    cout << "Loaded " << merge_ranks.size() << " merges." << endl;

    // 2. Load vocab
    unordered_map<string, int> vocab;
    ifstream fv("vocab.ddtok");
    if (!fv.is_open()) {
        cerr << "Error: Could not open vocab.ddtok. Please run this command from the project root." << endl;
        return 1;
    }
    while (getline(fv, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t tab = line.find('\t');
        if (tab != string::npos) {
            int id = std::stoi(line.substr(0, tab));
            string token = unescape_token(line.substr(tab + 1));
            vocab[token] = id;
        }
    }
    cout << "Loaded " << vocab.size() << " vocab entries." << endl;

    // 3. Interactive prompt
    cout << "\nEnter text to tokenize (or 'quit' to exit):" << endl;
    while (true) {
        cout << "> ";
        string text;
        if (!getline(cin, text)) break;
        if (text == "quit") break;

        // Split text into words
        vector<string> words;
        string word;
        for (char c : text) {
            if (isspace(c)) {
                if (!word.empty()) {
                    words.push_back(word);
                    word.clear();
                }
                word += c;
            } else {
                word += c;
            }
        }
        if (!word.empty()) {
            words.push_back(word);
        }

        // BPE Encode
        vector<int> token_ids;
        for (const string& w : words) {
            vector<string> tokens;
            for (char c : w) {
                tokens.push_back(string(1, c));
            }

            while (tokens.size() > 1) {
                int best_rank = 1e9;
                pair<string, string> best_pair;
                
                for (size_t i = 0; i < tokens.size() - 1; i++) {
                    pair<string, string> p = {tokens[i], tokens[i+1]};
                    auto it = merge_ranks.find(p);
                    if (it != merge_ranks.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_pair = p;
                    }
                }
                
                if (best_rank == 1e9) {
                    break; // No more merges
                }
                
                vector<string> new_tokens;
                for (size_t i = 0; i < tokens.size(); i++) {
                    if (i < tokens.size() - 1 && tokens[i] == best_pair.first && tokens[i+1] == best_pair.second) {
                        new_tokens.push_back(tokens[i] + tokens[i+1]);
                        i++; // skip next
                    } else {
                        new_tokens.push_back(tokens[i]);
                    }
                }
                tokens = std::move(new_tokens);
            }

            for (const string& t : tokens) {
                auto it = vocab.find(t);
                if (it != vocab.end()) {
                    token_ids.push_back(it->second);
                } else {
                    // Fallback, though base vocab covers all bytes
                    cerr << "Warning: Token not found in vocab: " << t << endl;
                }
            }
        }

        // Print output
        cout << "Token IDs: [";
        for (size_t i = 0; i < token_ids.size(); i++) {
            cout << token_ids[i] << (i + 1 == token_ids.size() ? "" : ", ");
        }
        cout << "]" << endl;
    }

    return 0;
}
