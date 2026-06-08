# CMakeLists.txt — Build configuration for ddtokens
#
# What this file does:
#   Tells CMake how to compile the project into three executables:
#   - tokenizer:   Full pipeline (ingest + train) in one shot
#   - build_freqs: Phase 1 — stream text files, serialize word frequency hashmap
#   - train_bpe:   Phase 2 — load hashmap, run BPE merges, output vocabulary
#
# Why this file exists:
#   Standard C++ build config. Alternatively you can compile directly with:
#   clang++ -std=c++17 -O2 -Iinclude src/main.cpp src/tokenizer.cpp -o tokenizer

cmake_minimum_required(VERSION 3.10)
project(ddtokens)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory so we can find tokenizer.h
include_directories(include)

# Full pipeline (backward compat)
add_executable(tokenizer src/main.cpp src/tokenizer_base.cpp src/tokenizer_naive.cpp)

# Phase 1: build word frequency hashmap from text files
add_executable(build_freqs src/build_freqs.cpp src/tokenizer_base.cpp)

# Phase 2: load hashmap, run BPE training, produce vocabulary
add_executable(train_bpe src/train_bpe.cpp src/tokenizer_base.cpp src/tokenizer_naive.cpp)

# Phase 2 Fast: heap-based optimization for BPE training
add_executable(train_bpe_heap src/train_bpe_heap.cpp src/tokenizer_base.cpp src/tokenizer_heap.cpp)

# Utility: prune hashmap to save memory
add_executable(prune_hashmap src/prune_hashmap.cpp src/tokenizer_base.cpp)
