/********************************************************************************
* Copyright 2020 Thomas A. Rieck, All Rights Reserved
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

#include "Model.h"
#include "RouteLayer.h"
#include <cblas.h>
#include <xtensor/xtensor.hpp>

namespace px {

using namespace xt;

RouteLayer::RouteLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
    auto layers = property<std::vector<int>>("layers");

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

    setOutChannels(outChannels);
    setOutHeight(outHeight);
    setOutWidth(outWidth);
    setOutputs(outputs);

    output_ = empty<float>({ batch(), outChannels, outHeight, outWidth });
}

std::ostream& RouteLayer::print(std::ostream& os)
{
    Layer::print(os, "route", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void RouteLayer::forward(const xt::xarray<float>& /*input*/)
{
    auto offset = 0;

    auto* output = output_.data();

    for (auto layer: layers_) {
        auto* input = layer->output().data();
        auto inputSize = layer->outputs();

        for (auto i = 0; i < batch(); ++i) {
            cblas_scopy(inputSize, input + i * inputSize, 1, output + offset + i * outputs(), 1);
        }

        offset += inputSize;
    }
}

} // px
