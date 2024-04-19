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
    tokenizer.encode("falldown", outputs, 16);
    printf("input_ids dim: %d, len: %ld\n", outputs[0].ndim, outputs[0].size());
    printf("token_type_ids dim: %d, len: %ld\n", outputs[1].ndim, outputs[1].size());
    printf("attention_mask dim: %d, len: %ld\n", outputs[2].ndim, outputs[2].size());

    std::string text;
    tokenizer.decode(outputs[0].buf, text);
    printf("text: %s\n", text.c_str());

    tokenizer.encode("nice", outputs);
    tokenizer.decode(outputs[0].buf, text);
    printf("text: %s\n", text.c_str());

    return 0;
}