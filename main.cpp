//
// Created by xingwg on 24-4-10.
//
#include <string>
#include <iostream>
#include "bert_tokenizer.h"


int main() {
    const std::string custom_op_library_path = "../3rdparty/onnxruntime-extensions/lib/libortextensions.so.0.11.0";
    BertTokenizer tokenizer(custom_op_library_path);

    std::vector<Tensor> outputs;
    tokenizer.encode("falldown", outputs);
    std::string text;
    tokenizer.decode(outputs[0].buf, text);
    std::cout << text << std::endl;

    tokenizer.encode("nice", outputs);
    tokenizer.decode(outputs[0].buf, text);
    std::cout << text << std::endl;

    return 0;
}