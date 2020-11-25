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

#include "BatchNormLayer.h"
#include "Utility.h"

namespace px {

using namespace xt;

BatchNormLayer::BatchNormLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
    biases_ = zeros<float>({ channels() });
    scales_ = ones<float>({ channels() });
    rollingMean_ = zeros<float>({ channels() });
    rollingVar_ = zeros<float>({ channels() });

    setOutChannels(channels());
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(outHeight() * outWidth() * outChannels());

    output_ = empty<float>({ batch(), outChannels(), outHeight(), outWidth() });
}

std::ostream& BatchNormLayer::print(std::ostream& os)
{
    Layer::print(os, "batchnorm", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void BatchNormLayer::forward(const xt::xarray<float>& input)
{
    output_ = input;

    auto b = batch();
    auto c = outChannels();
    auto size = outHeight() * outWidth();

    normalize_cpu(output_.data(), rollingMean_.data(), rollingVar_.data(), b, c, size);

    scale_bias(output_.data(), scales_.data(), b, c, size);
    add_bias(output_.data(), biases_.data(), b, c, size);
}

std::streamoff BatchNormLayer::loadDarknetWeights(std::istream& is)
{
    auto start = is.tellg();

    is.read((char*) biases_.data(), sizeof(float) * biases_.size());
    is.read((char*) scales_.data(), sizeof(float) * scales_.size());
    is.read((char*) rollingMean_.data(), sizeof(float) * rollingMean_.size());
    is.read((char*) rollingVar_.data(), sizeof(float) * rollingVar_.size());
    PX_CHECK(is.good(), "Could not read batch_normalize parameters");

    return is.tellg() - start;
}

}   // px

