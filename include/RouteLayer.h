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
class RouteLayer : public Layer<D>
{
public:
    using V = typename Layer<D>::V;

    RouteLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;
    void update() override;

    std::ostream& print(std::ostream& os) override;

private:
    std::vector<typename Layer<D>::Ptr> layers_;
};

template<Device D>
RouteLayer<D>::RouteLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    auto layers = this->template property<std::vector<int>>("layers");

    layers_.resize(layers.size());

    auto i = 0, outputs = 0, outWidth = 0, outHeight = 0, outChannels = 0;
    for (auto index: layers) {
        if (index < 0) {   // relative backwards
            index = this->index() + index;
        }

        PX_CHECK(index < model.layerSize(), "Layer index out of range.");
        PX_CHECK(index < this->index(), "Layer index ahead of current layer.");

        const auto& layer = model.layerAt(index);

        outputs += layer->outputs();

        if (outWidth == 0 && outHeight == 0 && outChannels == 0) {
            outWidth = layer->outWidth();
            outHeight = layer->outHeight();
            outChannels = layer->outChannels();
        } else if (outWidth == layer->outWidth() && outHeight == layer->outHeight()) {
            outChannels += layer->outChannels();
        } else {
            outWidth = outHeight = outChannels;
        }

        layers_[i++] = layer;
    }

    this->setOutChannels(outChannels);
    this->setOutHeight(outHeight);
    this->setOutWidth(outWidth);
    this->setOutputs(outputs);

    this->output_ = V(this->batch() * outputs, 0.0f);
    this->delta_ = V(this->batch() * outputs, 0.0f);
}

template<Device D>
std::ostream& RouteLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "route", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

template<Device D>
void RouteLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    auto offset = 0;

    auto* output = this->output_.data();

    for (const auto& layer: layers_) {
        const auto* pin = layer->output().data();
        auto inputSize = layer->outputs();

        for (auto i = 0; i < this->batch(); ++i) {
            const auto* in = pin + i * inputSize;
            auto* out = output + offset + i * this->outputs();

            cblas_scopy(inputSize, in, 1, out, 1);
        }

        offset += inputSize;
    }
}

template<Device D>
void RouteLayer<D>::backward(const V& input)
{
}

template<Device D>
void RouteLayer<D>::update()
{
}

using CpuRoute = RouteLayer<>;
using CudaRoute = RouteLayer<Device::CUDA>;

} // px

#ifdef USE_CUDA

#include "cuda/RouteLayer.h"

#endif // USE_CUDA


