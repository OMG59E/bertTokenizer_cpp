//
// Created by xingwg on 24-4-11.
//

#include <dlfcn.h>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_cpp_api_legacy.hpp"

#include "bert_tokenizer.h"
#include "bert_tokenizer_encode.h"
#include "bert_tokenizer_decode.h"


static std::string base64Decode(const char *Data, int DataByte) {
    const char DecodeTable[] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            62, // '+'
            0, 0, 0,
            63, // '/'
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'
            0, 0, 0, 0, 0, 0, 0,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'
            0, 0, 0, 0, 0, 0,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'
    };

    std::string strDecode;
    int nValue;
    int i = 0;
    while (i < DataByte) {
        if (*Data != '\r' && *Data != '\n') {
            nValue = DecodeTable[*Data++] << 18;
            nValue += DecodeTable[*Data++] << 12;
            strDecode += (nValue & 0x00FF0000) >> 16;
            if (*Data != '=') {
                nValue += DecodeTable[*Data++] << 6;
                strDecode += (nValue & 0x0000FF00) >> 8;
                if (*Data != '=') {
                    nValue += DecodeTable[*Data++];
                    strDecode += nValue & 0x000000FF;
                }
            }
            i += 4;
        } else {
            Data++;
            i++;
        }
    }
    return strDecode;
}


class BertTokenizerImpl {
public:
    explicit BertTokenizerImpl(const std::string& custom_op_library_path)
        : api_{Ort::GetApi()} {
        env_ = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "tokenizer");
        Ort::ThrowOnError(api_.RegisterCustomOpsLibrary(
                static_cast<OrtSessionOptions*>(session_options_),
                custom_op_library_path.c_str(), &custom_so_handle_));

        std::string enc_model_str = base64Decode(tokenizer_encode_str.data(), tokenizer_encode_str.size());
        std::string dec_model_str = base64Decode(tokenizer_decode_str.data(), tokenizer_decode_str.size());
        m_enc_ = new Ort::Session(*env_, enc_model_str.data(), enc_model_str.size(), session_options_);
        m_dec_ = new Ort::Session(*env_, dec_model_str.data(), dec_model_str.size(), session_options_);
    }

    ~BertTokenizerImpl() {
        SAFE_FREE(m_enc_);
        SAFE_FREE(m_dec_);
        SAFE_FREE(env_);
        if (custom_so_handle_)
            static_cast<void>(::dlclose(custom_so_handle_));
    }

    int encode(const std::string& text, std::vector<Tensor>& outputs, int32_t max_length = 0) {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<int64_t> enc_input_dims = {1}; // Assuming input shape [1]
        std::vector<Ort::Value> enc_inputs;
        Ort::Value& value = enc_inputs.emplace_back(
                Ort::Value::CreateTensor(
                        allocator, enc_input_dims.data(), enc_input_dims.size(),
                        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING));

        outputs.clear();
        value.FillStringTensorElement(text.c_str(), 0);
        auto ort_outputs = m_enc_->Run(Ort::RunOptions{nullptr},
                                       enc_input_names_.data(), enc_inputs.data(), enc_inputs.size(),
                                       enc_output_names_.data(), enc_output_names_.size());

        outputs.resize(3);
        for (int k=0; k<outputs.size(); ++k) {
            auto type_info = ort_outputs[k].GetTensorTypeAndShapeInfo();
            // ONNXTensorElementDataType output_type = type_info.GetElementType();
            std::vector<int64_t> dimension = type_info.GetShape();
            outputs[k].name = enc_output_names_[k];
            outputs[k].ndim = dimension.size();
            for (int n=0; n < dimension.size(); ++n)
                outputs[k].dims[n] = dimension[n];
            auto *data = ort_outputs[k].GetTensorMutableData<int64_t>();
            outputs[k].buf.assign(data, data + outputs[k].size());
        }

        if (max_length > 0) {
            int32_t input_ids_len = outputs[0].ndim;
            if (max_length < input_ids_len) {
                printf("max_length[%d] < input_ids_len[%d]\n", max_length, input_ids_len);
                return -1;
            }
            
            int32_t padding_len = max_length - input_ids_len;
            outputs[0].dims[0] = max_length;
            outputs[1].dims[0] = max_length;
            outputs[2].dims[0] = max_length;
            for (int i=0; i<padding_len; ++i) {
                outputs[0].buf.emplace_back(0);
                outputs[1].buf.emplace_back(0);
                outputs[2].buf.emplace_back(0);
            }
        }
        return 0;
    }

    int decode(const input_ids_t& input_ids, std::string& text) {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<int64_t> ids_dims = {static_cast<long>(input_ids.size())};
        std::vector<Ort::Value> dec_inputs;
        dec_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, const_cast<int64_t*>(
                        input_ids.data()), input_ids.size(), ids_dims.data(), ids_dims.size()));
        std::vector<int64_t> position_dims = {0,2};
        std::vector<int64_t> position = {};
        dec_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, const_cast<int64_t*>(
                        position.data()), position.size(), position_dims.data(), position_dims.size()));

        auto ort_outputs = m_dec_->Run(Ort::RunOptions{nullptr},
                                       dec_input_names_.data(), dec_inputs.data(), dec_inputs.size(),
                                       dec_output_names_.data(), dec_output_names_.size());
        std::vector<std::string> o_str;
        Ort::Value& value = ort_outputs[0];
        GetTensorMutableDataString(value, o_str);
        text = o_str[0];
        return 0;
    }

private:
    static void GetTensorMutableDataString(const OrtValue* value, std::vector<std::string>& output) {
        OrtTensorDimensions dimensions(OrtW::CustomOpApi(Ort::GetApi()), value);
        size_t len = static_cast<size_t>(dimensions.Size());
        size_t data_len;
        OrtW::ThrowOnError(Ort::GetApi(), Ort::GetApi().GetStringTensorDataLength(value, &data_len));
        output.resize(len);
        std::vector<char> result(data_len + len + 1, '\0');
        std::vector<size_t> offsets(len);
        OrtW::ThrowOnError(Ort::GetApi(), Ort::GetApi().GetStringTensorContent(value, (void*)result.data(), data_len, offsets.data(), offsets.size()));
        output.resize(len);
        for (int64_t i = (int64_t)len - 1; i >= 0; --i) {
            if (i < static_cast<int64_t>(len) - 1) {
                result[offsets[i + (int64_t)1]] = '\0';
            }
            output[i] = result.data() + offsets[i];
        }
    }

private:
    std::vector<const char*> enc_input_names_ = {"text"};
    std::vector<const char*> enc_output_names_ = {"input_ids", "token_type_ids", "attention_mask", "offset_mapping"};

    std::vector<const char*> dec_input_names_ = {"ids", "position"};
    std::vector<const char*> dec_output_names_ = {"str"};

    void* custom_so_handle_{nullptr};
    Ort::Env* env_{nullptr};
    const OrtApi& api_;
    Ort::SessionOptions session_options_;
    Ort::Session *m_enc_{nullptr}, *m_dec_{nullptr};
};


BertTokenizer::BertTokenizer(const std::string &custom_op_library_path) {
    auto* p = new BertTokenizerImpl(custom_op_library_path);
    handle_ = p;
}

BertTokenizer::~BertTokenizer() {
    auto* p = static_cast<BertTokenizerImpl*>(handle_);
    SAFE_FREE(p);
}

int BertTokenizer::encode(const std::string& text, std::vector<Tensor>& outputs, int32_t max_length) {
    auto* p = static_cast<BertTokenizerImpl*>(handle_);
    return p->encode(text, outputs, max_length);
}

int BertTokenizer::decode(const input_ids_t &input_ids, std::string &text) {
    auto* p = static_cast<BertTokenizerImpl*>(handle_);
    return p->decode(input_ids, text);
}
