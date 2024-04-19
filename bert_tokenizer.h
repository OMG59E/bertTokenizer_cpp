//
// Created by xingwg on 24-4-11.
//

#ifndef CPP_TOKENIZER_H
#define CPP_TOKENIZER_H

#include <vector>
#include <string>


#define SAFE_FREE(ptr) \
do {                   \
    if (ptr) {         \
        delete ptr;    \
        ptr = nullptr; \
    }                  \
} while(0)


struct Tensor {
    std::string name;
    int32_t ndim{0};
    int32_t dims[8]{};
    std::vector<int64_t> buf;

    int64_t size() {
        int64_t total = 1;
        for (int n=0; n<ndim; ++n) {
            total *= dims[n];
        }
        return total;
    }
};

typedef std::vector<int64_t> input_ids_t;

/**
 *
 */
class BertTokenizer {
public:
    /**
     *
     * @param custom_op_library_path
     */
    explicit BertTokenizer(const std::string& custom_op_library_path);
    ~BertTokenizer();

    /**
     *
     * @param text
     * @param outputs
     * @param max_length 最大长度，默认0 表示禁用padding, >0 表示启用padding
     * @return
     */
    int encode(const std::string& text, std::vector<Tensor>& outputs, int32_t max_length = 0);

    /**
     *
     * @param input_ids
     * @param text
     * @return
     */
    int decode(const input_ids_t& input_ids, std::string& text);

private:
    void* handle_{nullptr};
};

#endif //CPP_TOKENIZER_H
