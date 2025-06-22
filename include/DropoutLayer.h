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
#include "Utility.h"

namespace px {

template<Device D>
class DropoutExtras
{
    using V = typename Layer<D>::V;

protected:
    V randoms_;
    float scale_;
};

template<Device D = Device::CPU>
class DropoutLayer : public Layer<D>, public DropoutExtras<D>
{
public:
    using V = typename Layer<D>::V;
    DropoutLayer(Model<D>& model, YAML::Node layerDef);

    void forward(const V& input) override;
    void backward(const V& input, V* grad) override;

    std::ostream& print(std::ostream& os) override;

private:
    void setup();
    float probability_;
};

template<Device D>
DropoutLayer<D>::DropoutLayer(Model<D>& model, YAML::Node layerDef) : Layer<D>(model, layerDef)
{
    probability_ = this->template property<float>("probability", 0.5f);

    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->outHeight() * this->outWidth() * this->outChannels());

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);

    setup();
}

template<Device D>
void DropoutLayer<D>::setup()
{
    this->randoms_ = V(this->batch() * this->outputs(), 0.0f);
    this->scale_ = 1.0f / (1.0f - this->probability_ + 1e-6f);
}

template<Device D>
std::ostream& DropoutLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "dropout", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

template<Device D>
void DropoutLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    this->output_.copy(input);

    if (!this->model().training()) {
        return;
    }

    this->randoms_ = random<V>(this->batch() * this->outputs(), 0.0f, 1.0f);

    for (auto i = 0; i < this->batch() * this->outputs(); ++i) {
        auto r = this->randoms_[i];
        if (r < probability_) {
            this->output_[i] = 0.0f;
        } else {
            this->output_[i] *= this->scale_;
        }
    }
}

template<Device D>
void DropoutLayer<D>::backward(const V& input, V* grad)
{
    Layer<D>::backward(input, grad);

    if (!this->model().training()) {
        return;
    }

    for (auto i = 0; i < this->batch() * this->outputs(); ++i) {
        auto r = this->randoms_[i];
        if (r < probability_) {
            this->delta_[i] = 0.0f;
        } else {
            this->delta_[i] *= this->scale_;
        }
    }
}

using CpuDropout = DropoutLayer<>;
using CudaDropout = DropoutLayer<Device::CUDA>;

}   // px


#ifdef USE_CUDA

#include "cuda/DropoutLayer.h"

#endif  // USE_CUDA
