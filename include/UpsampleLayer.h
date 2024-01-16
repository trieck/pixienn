/********************************************************************************
* Copyright 2020-2023 Thomas A. Rieck, All Rights Reserved
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

#include "Layer.h"

namespace px {

template<Device D = Device::CPU>
class UpsampleExtras
{
};

template<Device D = Device::CPU>
class UpsampleLayer : public Layer<D>, public UpsampleExtras<D>
{
public:
    using V = typename Layer<D>::V;

    UpsampleLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;
    void update() override;

    std::ostream& print(std::ostream& os) override;

private:
    void setup();
    void upsample(const float* in, float* out, float* acc, bool forward);

    int stride_;
    float scale_;
};

template<Device D>
UpsampleLayer<D>::UpsampleLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    stride_ = this->template property<int>("stride", 2);
    scale_ = this->template property<float>("scale", 1.0f);

    this->setOutChannels(this->channels());
    this->setOutHeight(this->height() * stride_);
    this->setOutWidth(this->width() * stride_);
    this->setOutputs(this->outHeight() * this->outWidth() * this->outChannels());

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);

    setup();
}

template<Device D>
void UpsampleLayer<D>::setup()
{

}

template<Device D>
std::ostream& UpsampleLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "upsample", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

template<Device D>
void UpsampleLayer<D>::forward(const V& input)
{
    upsample(input.data(), this->output_.data(), nullptr, true);
}

template<Device D>
void UpsampleLayer<D>::backward(const V& input)
{
    upsample(nullptr, this->delta_.data(), this->netDelta()->data(), false);
}

template<Device D>
void UpsampleLayer<D>::update()
{
}

template<Device D>
void UpsampleLayer<D>::upsample(const float* in, float* out, float* acc, bool forward)
{
    auto c = this->channels();
    auto h = this->height();
    auto w = this->width();

    for (auto b = 0; b < this->batch(); ++b) {
        for (auto k = 0; k < c; ++k) {
            for (auto j = 0; j < h * stride_; ++j) {
                for (auto i = 0; i < w * stride_; ++i) {
                    auto inIndex = b * w * h * c + k * w * h + (j / stride_) * w + i / stride_;
                    auto outIndex =
                            b * w * h * c * stride_ * stride_ + k * w * h * stride_ * stride_ + j * w * stride_ + i;

                    if (forward) {
                        out[outIndex] = scale_ * in[inIndex];
                    } else {
                        acc[inIndex] += scale_ * out[outIndex];
                    }
                }
            }
        }
    }
}

using CpuUpsample = UpsampleLayer<>;
using CudaUpsample = UpsampleLayer<Device::CUDA>;

}   // px

#ifdef USE_CUDA

#include "cuda/UpsampleLayer.h"

#endif // USE_CUDA
