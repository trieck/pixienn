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

#include "ConnLayer.h"
#include "Utility.h"
#include "xtensor/xrandom.hpp"
#include <cblas.h>

namespace px {

using namespace xt;

ConnLayer::ConnLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
    activation_ = property<std::string>("activation", "logistic");
    activationFnc_ = Activation::get(activation_);

    auto batchNormalize = property<bool>("batch_normalize", false);

    setChannels(inputs());
    setHeight(1);
    setWidth(1);

    setOutputs(property<int>("output", 1));
    setOutHeight(1);
    setOutWidth(1);
    setOutChannels(outputs());

    if (batchNormalize) {
        auto def = layerDef;
        def["type"] = "batchnorm";
        def["channels"] = outChannels();
        def["height"] = outHeight();
        def["width"] = outWidth();
        batchNormalize_ = Layer::create(model, def);
    } else {
        biases_ = zeros<float>({ outputs() });
    }

    weights_ = random::rand<float>({ inputs(), outputs() });
    output_ = zeros<float>({ batch() * outputs() });
}

std::ostream& ConnLayer::print(std::ostream& os)
{
    os << std::setfill('.');

    os << std::setw(60) << std::left << "connected"
       << std::setw(20) << std::left << inputs()
       << std::setw(20) << std::left
       << outputs()
       << std::endl;

    return os;
}

std::streamoff ConnLayer::loadDarknetWeights(std::istream& is)
{
    auto start = is.tellg();

    is.read((char*) biases_.data(), biases_.size() * sizeof(float));
    PX_CHECK(is.good(), "Could not read biases");

    is.read((char*) weights_.data(), sizeof(float) * weights_.size());
    PX_CHECK(is.good(), "Could not read weights");

    if (batchNormalize_) {
        is.read((char*) &scales_, sizeof(float));
        is.read((char*) &rollingMean_, sizeof(float));
        is.read((char*) &rollingVar_, sizeof(float));
        PX_CHECK(is.good(), "Could not read batch_normalize parameters");
    }

    return is.tellg() - start;
}

void ConnLayer::forward(const xt::xarray<float>& input)
{
    output_.fill(0);

    auto m = batch();
    auto n = outputs();
    auto k = inputs();
    auto* a = input.data();
    auto* b = weights_.data();
    auto* c = output_.data();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a, k, b, k, 1.0f, c, n);

    if (batchNormalize_) {
        batchNormalize_->forward(output_);
        output_ = batchNormalize_->output();
    } else {
        add_bias(c, biases_.data(), m, outChannels(), outHeight() * outWidth());
    }

    activationFnc_->apply(output_);
}

} // px
