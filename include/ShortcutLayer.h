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
class ShortcutLayer : public Layer<D>
{
public:
    using V = typename Layer<D>::V;

    ShortcutLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;

    std::ostream& print(std::ostream& os) override;

private:
    void shorcut(int w1, int h1, int c1, const V& add, int w2, int h2, int c2, V& out);

    Activations<D>::Ptr activation_;
    Layer<D>::Ptr from_;
    float alpha_, beta_;
};

template<Device D>
ShortcutLayer<D>::ShortcutLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    auto activation = this->template property<std::string>("activation", "linear");
    activation_ = Activations<D>::get(activation);

    alpha_ = this->template property<float>("alpha", 1.0f);
    beta_ = this->template property<float>("beta", 1.0f);

    auto index = this->template property<int>("from");
    if (index < 0) {   // relative backwards
        index = this->index() + index;
    }

    PX_CHECK(index < model.layerSize(), "Layer index out of range.");
    PX_CHECK(index < this->index(), "Layer index ahead of current layer.");

    from_ = this->model().layerAt(index);

    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->outHeight() * this->outWidth() * this->outChannels());

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);
}

template<Device D>
std::ostream& ShortcutLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "shortcut", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

template<Device D>
void ShortcutLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    this->output_.copy(input);

    shorcut(this->width(), this->height(), this->channels(), from_->output(),
            this->outWidth(), this->outHeight(), this->outChannels(), this->output_);

    activation_->apply(this->output_);
}

template<Device D>
void ShortcutLayer<D>::backward(const V& input)
{
    Layer<D>::backward(input);

    activation_->gradient(this->output_, this->delta_);

    cblas_saxpy(this->batch() * this->outputs(), alpha_, this->delta_.data(), 1, this->model().delta()->data(), 1);

    shorcut(this->outWidth(), this->outHeight(), this->outChannels(), this->delta_,
            this->width(), this->height(), this->channels(), from_->delta());
}

template<Device D>
void ShortcutLayer<D>::shorcut(int w1, int h1, int c1, const V& add, int w2, int h2, int c2, V& out)
{
    const auto* padd = add.data();
    auto* pout = out.data();

    auto stride = std::max(1, w1 / w2);
    auto sample = std::max(1, w2 / w1);

    PX_CHECK(stride == h1 / h2, "Stride must be equal in x and y direction.");
    PX_CHECK(sample == h2 / h1, "Sample must be equal in x and y direction.");

    auto minw = std::min(w1, w2);
    auto minh = std::min(h1, h2);
    auto minc = std::min(c1, c2);

    for (auto b = 0; b < this->batch(); ++b) {
        for (auto k = 0; k < minc; ++k) {
            for (auto j = 0; j < minh; ++j) {
                for (auto i = 0; i < minw; ++i) {
                    auto outIndex = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                    auto addIndex = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
                    pout[outIndex] = alpha_ * pout[outIndex] + beta_ * padd[addIndex];
                }
            }
        }
    }
}

using CpuShortcut = ShortcutLayer<>;
using CudaShortcut = ShortcutLayer<Device::CUDA>;

} // px

#ifdef USE_CUDA

#include "cuda/ShortcutLayer.h"

#endif // USE_CUDA

