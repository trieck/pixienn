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

#include "ConvLayer.h"
#include "ImageToCol.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xrandom.hpp"
#include <cblas.h>

using namespace px;
using namespace xt;


ConvLayer::ConvLayer(const YAML::Node& layerDef) : Layer(layerDef)
{
    activation_ = property<std::string>("activation");
    batchNormalize_ = property<bool>("batch_normalize", false);
    dilation_ = property<int>("dilation", 0);
    filters_ = property<int>("filters", 1);
    kernel_ = property<int>("kernel", 1);
    pad_ = property<int>("pad", 0);
    stride_ = property<int>("stride", 1);
    groups_ = std::max(1, property<int>("groups", 1));

    weights_ = random::rand<float>({ filters_, channels() / groups_, kernel_, kernel_ });
    biases_ = zeros<float>({ filters_ });

    if (batchNormalize_) {
        scales_ = zeros<float>({ filters_ });
        rollingMean_ = zeros<float>({ filters_ });
        rollingVar_ = zeros<float>({ filters_ });
    }

    setOutChannels(filters_);
    setOutHeight((height() + 2 * pad_ - kernel_) / stride_ + 1);
    setOutWidth((width() + 2 * pad_ - kernel_) / stride_ + 1);

    setOutputs(outHeight() * outWidth() * outChannels());

    output_ = zeros<float>({ batch(), outChannels(), outHeight(), outWidth() });
}

ConvLayer::~ConvLayer()
{

}

std::ostream& ConvLayer::print(std::ostream& os)
{
    os << std::setfill('.');

    os << std::setw(20) << std::left << "conv"
       << std::setw(20) << std::left << filters_
       << std::setw(20) << std::left
       << std::string(std::to_string(kernel_) + " x " + std::to_string(kernel_) + " / " + std::to_string(stride_))
       << std::setw(20) << std::left
       << std::string(std::to_string(channels()) + " x " + std::to_string(height()) + " x " + std::to_string(width()))
       << std::setw(20) << std::left
       << std::string(
               std::to_string(outChannels()) + " x " + std::to_string(outHeight()) + " x " + std::to_string(outWidth()))
       << std::endl;

    return os;
}

void ConvLayer::loadDarknetWeights(std::istream& is)
{
    is.read((char*) biases_.data(), filters_ * sizeof(float));
    PX_CHECK(is.good(), "Could not read biases");

    if (batchNormalize_) {
        is.read((char*) scales_.data(), sizeof(float) * scales_.size());
        is.read((char*) rollingMean_.data(), sizeof(float) * rollingMean_.size());
        is.read((char*) rollingVar_.data(), sizeof(float) * rollingVar_.size());
        PX_CHECK(is.good(), "Could not read batch_normalize parameters");
    }

    is.read((char*) weights_.data(), sizeof(float) * weights_.size());
    PX_CHECK(is.good(), "Could not read weights");
}

xt::xarray<float> ConvLayer::forward(const xt::xarray<float>& input)
{
    int m = filters_ / groups_;
    int n = outWidth() * outHeight();
    int k = kernel_ * kernel_ * channels() / groups_;

    int nweights = weights_.size();
    const float* pweights = weights_.data();

    const float* pin = input.data();
    float* pout = output_.data();

    xt::xtensor<float, 2> B = empty<float>({ k, outHeight() * outWidth() });

    for (auto i = 0; i < batch(); ++i) {
        for (auto j = 0; j < groups_; ++j) {
            const float* a = pweights + j * nweights / groups_;
            const float* im = pin + (i * groups_ + j) * channels() / groups_ * height() * width();
            float* c = pout + (i * groups_ + j) * n * m;

            const float* b = B.data();
            if (kernel_ == 1) {
                b = im;
            } else {
                im2col_cpu(im, channels() / groups_, height(), width(), kernel_, stride_, pad_, B.data());
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
        }
    }

    return output_;
}
