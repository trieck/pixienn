/********************************************************************************
* Copyright 2026 Thomas A. Rieck, All Rights Reserved
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
********************************************************************************/

#pragma once

#include <cmath>
#include <cblas.h>

#include "Layer.h"

namespace px {

/**
 * Adds fixed two-dimensional sinusoidal position information to each token.
 *
 * PixieNN tensors use [batch, channels, height, width]. The first half of the
 * channels encode the row and the remaining channels encode the column. This
 * layer preserves the tensor shape and has an identity backward path.
 */
template<Device D = Device::CPU>
class PositionalEncoding : public Layer<D>
{
public:
    using V = Layer<D>::V;

    PositionalEncoding(Model<D>& model, YAML::Node layerDef);

    void forward(const V& input) override;
    void backward(const V& input, V* grad) override;

    std::ostream& print(std::ostream& os) override;

private:
    V encoding_;
};

using CpuPositionalEncoding = PositionalEncoding<>;

template<>
inline PositionalEncoding<>::PositionalEncoding(Model<>& model, YAML::Node layerDef)
        : Layer<>(model, layerDef)
{
    PX_CHECK(this->channels() > 0, "positional encoding requires positive channels");
    PX_CHECK(this->height() > 0 && this->width() > 0,
             "positional encoding requires positive height and width");

    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->outHeight() * this->outWidth() * this->outChannels());
    encoding_ = V(this->outputs(), 0.0f);

    const auto rowChannels = std::max(1, this->channels() / 2);
    for (auto c = 0; c < this->channels(); ++c) {
        const auto column = c >= rowChannels;
        const auto localChannel = column ? c - rowChannels : c;
        const auto axisChannels = column ? this->channels() - rowChannels : rowChannels;
        const auto divisor = std::pow(10000.0f,
                                      2.0f * static_cast<float>(localChannel / 2)
                                              / static_cast<float>(std::max(1, axisChannels)));
        for (auto y = 0; y < this->height(); ++y) {
            for (auto x = 0; x < this->width(); ++x) {
                const auto position = static_cast<float>(column ? x : y);
                const auto value = position / divisor;
                const auto index = x + this->width() * (c + this->channels() * y);
                encoding_[index] = localChannel % 2 == 0 ? std::sin(value) : std::cos(value);
            }
        }
    }

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);
}

template<>
inline void PositionalEncoding<>::forward(const V& input)
{
    Layer<>::forward(input);
    for (auto b = 0; b < this->batch(); ++b) {
        const auto offset = b * this->outputs();
        for (auto i = 0; i < this->outputs(); ++i) {
            this->output_[offset + i] = input[offset + i] + encoding_[i];
        }
    }
}

template<>
inline void PositionalEncoding<>::backward(const V& input, V* grad)
{
    Layer<>::backward(input, grad);
    if (grad != nullptr) {
        cblas_saxpy(this->batch() * this->outputs(), 1.0f,
                    this->delta_.data(), 1, grad->data(), 1);
    }
}

template<>
inline std::ostream& PositionalEncoding<>::print(std::ostream& os)
{
    Layer<>::print(os, "positional-encoding",
                   { this->height(), this->width(), this->channels() },
                   { this->outHeight(), this->outWidth(), this->outChannels() });
    return os;
}

} // namespace px

#ifdef USE_CUDA
#include "cuda/PositionalEncoding.h"
#endif
