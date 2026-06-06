# CMakeLists.txt — Build configuration for ddtokens
#
# What this file does:
#   Tells CMake how to compile the project into a single executable called "tokenizer".
#   Links main.cpp and tokenizer.cpp together, points the compiler at include/ for headers.
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

# Create the executable program named "tokenizer"
add_executable(tokenizer src/main.cpp src/tokenizer.cpp)
