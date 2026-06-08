#pragma once
#include "tokenizer_base.h"

class BPETokenizerHeap : public BPETokenizerBase {
public:
    void train(size_t num_merges) override;
};
