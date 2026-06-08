#pragma once
#include "tokenizer_base.h"

class BPETokenizerNaive : public BPETokenizerBase {
public:
    void train(size_t num_merges) override;

private:
    PairCounts count_pairs() const;
    void apply_merge(const string& a, const string& b);
};
