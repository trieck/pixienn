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

template<Device D>
class MPExtras
{
protected:
    PxCpuVectorT<int> indexes_;
};


template<Device D = Device::CPU>
class MaxPoolLayer : public Layer<D>, public MPExtras<D>
{
public:
    using V = typename Layer<D>::V;

    MaxPoolLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;

    std::ostream& print(std::ostream& os) override;

private:
    void setup();
    int kernel_ = 0;
    int stride_ = 0;
    int padding_ = 0;
};

template<Device D>
MaxPoolLayer<D>::MaxPoolLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    kernel_ = this->template property<int>("kernel", 1);
    stride_ = this->template property<int>("stride", 1);
    padding_ = this->template property<int>("padding", std::max<int>(0, kernel_ - 1));

    this->setOutChannels(this->channels());
    this->setOutHeight((this->height() + padding_ - kernel_) / stride_ + 1);
    this->setOutWidth((this->width() + padding_ - kernel_) / stride_ + 1);
    this->setOutputs(this->outChannels() * this->outHeight() * this->outWidth());
    auto outputSize = this->batch() * this->outputs();

    this->output_ = V(outputSize, 0.0f);
    this->delta_ = V(outputSize, 0.0f);

    setup();
}

template<Device D>
void MaxPoolLayer<D>::setup()
{
    this->indexes_ = PxCpuVectorT<int>(this->batch() * this->outputs(), 0);
}

template<Device D>
std::ostream& MaxPoolLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "maxpool", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() },
                    std::nullopt, std::array<int, 3>{ this->kernel_, this->kernel_, this->stride_ });
    return os;
}

template<Device D>
void MaxPoolLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    auto wOffset = -padding_ / 2;
    auto hOffset = -padding_ / 2;

    auto ih = this->height();
    auto iw = this->width();
    auto oh = this->outHeight();
    auto ow = this->outWidth();
    auto c = this->channels();

    const auto min = std::numeric_limits<float>::lowest();
    const auto* pin = input.data();
    auto* pout = this->output_.data();
    auto* pidx = this->indexes_.data();

    for (auto b = 0; b < this->batch(); ++b) {
        for (auto k = 0; k < c; ++k) {
            for (auto i = 0; i < oh; ++i) {
                for (auto j = 0; j < ow; ++j) {
                    auto outIndex = j + ow * (i + oh * (k + c * b));
                    auto max = min;
                    auto maxIndex = -1;

                    for (auto n = 0; n < kernel_; ++n) {
                        for (auto m = 0; m < kernel_; ++m) {
                            auto curH = hOffset + i * stride_ + n;
                            auto curW = wOffset + j * stride_ + m;
                            auto index = curW + iw * (curH + ih * (k + c * b));
                            auto valid = (curH >= 0 && curH < ih && curW >= 0 && curW < iw);
                            auto val = (valid != 0) ? pin[index] : min;
                            maxIndex = (val > max) ? index : maxIndex;
                            max = (val > max) ? val : max;
                        }
                    }

                    pout[outIndex] = max;
                    pidx[outIndex] = maxIndex;
                }
            }
        }
    }
}

template<Device D>
void MaxPoolLayer<D>::backward(const V& input)
{
    Layer<D>::backward(input);

    auto size = this->batch() * this->outputs();

    auto* pindexes = this->indexes_.data();
    auto* netDelta = this->netDelta()->data();
    const auto* pDelta = this->delta_.data();

    for (auto i = 0; i < size; ++i) {
        auto index = pindexes[i];
        netDelta[index] += pDelta[i];
    }
}

using CpuMaxPool = MaxPoolLayer<>;
using CudaMaxPool = MaxPoolLayer<Device::CUDA>;

} // px

#ifdef USE_CUDA

#include "cuda/MaxPoolLayer.h"

#endif
