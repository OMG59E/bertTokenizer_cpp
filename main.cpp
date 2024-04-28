//
// Created by xingwg on 24-4-10.
//
#include <string>
#include <iostream>
#include "bert_tokenizer.h"

void print_tensor(Tensor& t) {
    printf("%s dim: %d, len: %ld\n", t.name.c_str(), t.ndim, t.size());

    printf("shape: ");
    for (int i=0; i<t.ndim; ++i) {
        printf("%d ", t.dims[i]);
    }
    printf("\n");

    printf("value: ");
    for (int i=0; i<t.size(); ++i) {
        printf("%ld ", t.buf[i]);
    }
    printf("\n");
}


int main() {
    const std::string custom_op_library_path = "../3rdparty/onnxruntime-extensions/lib/libortextensions.so.0.11.0";
    BertTokenizer tokenizer(custom_op_library_path);
    std::string prompt = "falldown . background .";
    std::vector<Tensor> outputs;
    tokenizer.encode(prompt, outputs, 16);
    print_tensor(outputs[0]);
    print_tensor(outputs[1]);
    print_tensor(outputs[2]);
    print_tensor(outputs[3]);

    std::cout << "sub0: " << prompt.substr(0, 0) << std::endl;
    std::cout << "sub1: " << prompt.substr(0, 4) << std::endl;
    std::cout << "sub2: " << prompt.substr(4, 4) << std::endl;
    std::cout << "sub3: " << prompt.substr(9, 1) << std::endl;
    std::cout << "sub4: " << prompt.substr(11, 10) << std::endl;
    std::cout << "sub5: " << prompt.substr(22, 1) << std::endl;
    std::cout << "sub5: " << prompt.substr(0, 0) << std::endl;

    std::string text;
    tokenizer.decode(outputs[0].buf, text);
    printf("text: %s\n", text.c_str());

    tokenizer.encode("nice", outputs);
    tokenizer.decode(outputs[0].buf, text);
    printf("text: %s\n", text.c_str());

    return 0;
}