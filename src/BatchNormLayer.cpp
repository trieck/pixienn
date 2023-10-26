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
    setOutChannels(channels());
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(outHeight() * outWidth() * outChannels() * batch());

#ifdef USE_CUDA
    biases_ = PxDevVector<float>(channels(), 0.f);
    scales_ = PxDevVector<float>(channels(), 1.f);
    rollingMean_ = PxDevVector<float>(channels(), 0.f);
    rollingVar_ = PxDevVector<float>(channels(), 0.f);
    output_ = PxDevVector<float>(batch() * outChannels() * outHeight() * outWidth(), 0.f);
    x_ = PxDevVector<float>(batch() * outputs());

    cudnnSetTensor4dDescriptor(dstTens_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch(), outChannels(), outHeight(),
                               outWidth());
    cudnnSetTensor4dDescriptor(normTens_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outChannels(), 1, 1);
#else
    biases_ = zeros<float>({ channels() });
    scales_ = ones<float>({ channels() });
    rollingMean_ = zeros<float>({ channels() });
    rollingVar_ = zeros<float>({ channels() });
#endif
}

std::ostream& BatchNormLayer::print(std::ostream& os)
{
    Layer::print(os, "batchnorm", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void BatchNormLayer::forward(const PxDevVector<float>& input)
{
    output_.deviceCopy(input);

#ifdef USE_CUDA
    float alpha = 1;
    float beta = 0;

    CudnnContext context;

    auto status = cudnnBatchNormalizationForwardInference(context,
                                                          CUDNN_BATCHNORM_SPATIAL,
                                                          &alpha,
                                                          &beta,
                                                          dstTens_,
                                                          x_.get(),
                                                          dstTens_,
                                                          output_.get(),
                                                          normTens_,
                                                          scales_.get(),
                                                          biases_.get(),
                                                          rollingMean_.get(),
                                                          rollingVar_.get(),
                                                          0.00001);
    PX_CHECK_CUDNN(status);
#else
    auto b = batch();
    auto c = outChannels();
    auto size = outHeight() * outWidth();

    normalize_cpu(output_.data(), rollingMean_.data(), rollingVar_.data(), b, c, size);

    scale_bias(output_.data(), scales_.data(), b, c, size);
    add_bias(output_.data(), biases_.data(), b, c, size);
#endif // USE_CUDA
}

std::streamoff BatchNormLayer::loadDarknetWeights(std::istream& is)
{
    auto start = is.tellg();

#if USE_CUDA
    std::vector<float> biases(biases_.size());
    std::vector<float> scales(scales_.size());
    std::vector<float> rollingMean(rollingMean_.size());
    std::vector<float> rollingVar(rollingVar_.size());

    is.read((char*) biases.data(), sizeof(float) * biases_.size());
    is.read((char*) scales.data(), sizeof(float) * scales_.size());
    is.read((char*) rollingMean.data(), sizeof(float) * rollingMean_.size());
    is.read((char*) rollingVar.data(), sizeof(float) * rollingVar_.size());

    biases_.hostCopy(biases);
    scales_.hostCopy(scales);
    rollingMean_.hostCopy(rollingMean);
    rollingVar_.hostCopy(rollingVar);

#else
    is.read((char*) biases_.get(), sizeof(float) * biases_.size());
    is.read((char*) scales_.get(), sizeof(float) * scales_.size());
    is.read((char*) rollingMean_.get(), sizeof(float) * rollingMean_.size());
    is.read((char*) rollingVar_.get(), sizeof(float) * rollingVar_.size());
#endif

    PX_CHECK(is.good(), "Could not read batch_normalize parameters");

    return is.tellg() - start;
}

}   // px

