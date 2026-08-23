#pragma once

#include <cmath>
#include <vector>

#include "Cublas.h"
#include "TransformerKernel.cuh"

namespace px {

template<>
inline PositionalEncoding<Device::CUDA>::PositionalEncoding(Model<Device::CUDA>& model, YAML::Node layerDef)
        : Layer<Device::CUDA>(model, layerDef)
{
    PX_CHECK(this->channels() > 0 && this->height() > 0 && this->width() > 0,
             "positional encoding requires positive dimensions");
    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->outHeight() * this->outWidth() * this->outChannels());
    std::vector<float> encoding(this->outputs(), 0.0f);
    const auto rowChannels = std::max(1, this->channels() / 2);
    for (auto c = 0; c < this->channels(); ++c) {
        const auto column = c >= rowChannels;
        const auto local = column ? c - rowChannels : c;
        const auto axis = column ? this->channels() - rowChannels : rowChannels;
        const auto divisor = std::pow(10000.0f, 2.0f * static_cast<float>(local / 2) / std::max(1, axis));
        for (auto y = 0; y < this->height(); ++y) for (auto x = 0; x < this->width(); ++x) {
            const auto position = static_cast<float>(column ? x : y);
            const auto value = position / divisor;
            encoding[x + this->width() * (c + this->channels() * y)] = local % 2 == 0 ? std::sin(value) : std::cos(value);
        }
    }
    encoding_ = V(this->outputs(), 0.0f);
    encoding_.copyHost(encoding.data(), encoding.size());
    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);
}

template<>
inline void PositionalEncoding<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);
    positionalEncodingGpu(this->batch() * this->outputs(), input.data(), encoding_.data(), this->output_.data());
}

template<>
inline void PositionalEncoding<Device::CUDA>::backward(const V& input, V* grad)
{
    Layer<Device::CUDA>::backward(input, grad);
    if (grad != nullptr) {
        auto one = 1.0f;
        PX_CHECK_CUBLAS(cublasSaxpy(this->cublasContext(), this->delta_.size(), &one, this->delta_.data(), 1,
                                    grad->data(), 1));
    }
}

template<>
inline std::ostream& PositionalEncoding<Device::CUDA>::print(std::ostream& os)
{
    Layer<Device::CUDA>::print(os, "positional-encoding", { this->height(), this->width(), this->channels() },
                               { this->outHeight(), this->outWidth(), this->outChannels() });
    return os;
}

} // namespace px
