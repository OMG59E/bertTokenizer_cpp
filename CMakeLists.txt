cmake_minimum_required(VERSION 3.10)

project(bertTokenizer_cpp C CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(3rdparty/onnxruntime/include)
include_directories(3rdparty/onnxruntime-extensions/include)

link_directories(3rdparty/onnxruntime/lib)

list(APPEND LINK_LIBS onnxruntime dl)

add_library(tokenizer SHARED tokenizer.cpp)
target_link_libraries(tokenizer ${LINK_LIBS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} tokenizer)