cmake_minimum_required(VERSION 3.10)
project(ddtokens)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory so we can find tokenizer.h
include_directories(include)

# Create the executable program named "tokenizer"
add_executable(tokenizer src/main.cpp src/tokenizer.cpp)
