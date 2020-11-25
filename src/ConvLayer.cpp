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

#include "Activation.h"
#include "ConvLayer.h"
#include "Utility.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xrandom.hpp"
#include <cblas.h>

namespace px {

using namespace xt;

ConvLayer::ConvLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
    activation_ = property<std::string>("activation", "logistic");
    activationFnc_ = Activation::get(activation_);

    auto batchNormalize = property<bool>("batch_normalize", false);
    dilation_ = property<int>("dilation", 0);
    filters_ = property<int>("filters", 1);
    kernel_ = property<int>("kernel", 1);
    auto pad = property<bool>("pad", 0);
    padding_ = pad ? kernel_ / 2 : 0;
    stride_ = property<int>("stride", 1);
    groups_ = std::max(1, property<int>("groups", 1));

    setOutChannels(filters_);
    setOutHeight((height() + 2 * padding_ - kernel_) / stride_ + 1);
    setOutWidth((width() + 2 * padding_ - kernel_) / stride_ + 1);
    setOutputs(outHeight() * outWidth() * outChannels());

    if (batchNormalize) {
        auto def = layerDef;
        def["type"] = "batchnorm";
        def["channels"] = outChannels();
        def["height"] = outHeight();
        def["width"] = outWidth();
        batchNormalize_ = Layer::create(model, def);
    } else {
        biases_ = zeros<float>({ filters_ });
    }

    weights_ = random::rand<float>({ filters_, channels() / groups_, kernel_, kernel_ });
    column_ = empty<float>({ kernel_ * kernel_ * channels() / groups_, outHeight() * outWidth() });
    output_ = empty<float>({ batch(), outChannels(), outHeight(), outWidth() });
}

std::ostream& ConvLayer::print(std::ostream& os)
{
    os << std::setfill('.');

    os << std::setw(20) << std::left << "conv"
       << std::setw(20) << std::left << filters_
       << std::setw(20) << std::left
       << std::string(std::to_string(kernel_) + " x " + std::to_string(kernel_) + " / " + std::to_string(stride_))
       << std::setw(20) << std::left
       << std::string(std::to_string(height()) + " x " + std::to_string(width()) + " x " + std::to_string(channels()))
       << std::setw(20) << std::left
       << std::string(
               std::to_string(outHeight()) + " x " + std::to_string(outWidth()) + " x " + std::to_string(outChannels()))
       << std::endl;

    return os;
}

std::streamoff ConvLayer::loadDarknetWeights(std::istream& is)
{
    auto start = is.tellg();

    if (batchNormalize_) {
        batchNormalize_->loadDarknetWeights(is);
    } else {
        is.read((char*) biases_.data(), biases_.size() * sizeof(float));
        PX_CHECK(is.good(), "Could not read biases");
    }

    is.read((char*) weights_.data(), sizeof(float) * weights_.size());
    PX_CHECK(is.good(), "Could not read weights");

    return is.tellg() - start;
}

void ConvLayer::forward(const xt::xarray<float>& input)
{
    int m = filters_ / groups_;
    int n = outWidth() * outHeight();
    int k = kernel_ * kernel_ * channels() / groups_;

    int nweights = weights_.size();
    const auto* pweights = weights_.data();

    const auto* pin = input.data();
    auto* pout = output_.data();

    for (auto i = 0; i < batch(); ++i) {
        for (auto j = 0; j < groups_; ++j) {
            const auto* im = pin + (i * groups_ + j) * channels() / groups_ * height() * width();
            const auto* a = pweights + j * nweights / groups_;
            const auto* b = kernel_ == 1 ? im : column_.data();
            auto* c = pout + (i * groups_ + j) * n * m;

            if (kernel_ != 1) {
                im2col_cpu(im, channels() / groups_, height(), width(), kernel_, stride_, padding_, column_.data());
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
        }
    }

    if (batchNormalize_) {
        batchNormalize_->forward(output_);
        output_ = batchNormalize_->output();
    } else {
        add_bias(output_.data(), biases_.data(), batch(), outChannels(), outHeight() * outWidth());
    }

    activationFnc_->apply(output_);
}

}   // px
