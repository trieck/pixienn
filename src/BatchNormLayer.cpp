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
}

void BatchNormLayer::setup()
{
    setOutChannels(channels());
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(outHeight() * outWidth() * outChannels() * batch());

    biases_ = zeros<float>({ channels() });
    scales_ = ones<float>({ channels() });
    rollingMean_ = zeros<float>({ channels() });
    rollingVar_ = zeros<float>({ channels() });
    output_ = empty<float>({ batch(), outChannels(), outHeight(), outWidth() });

#ifdef USE_CUDA
    setupGpu();
#endif
}

std::ostream& BatchNormLayer::print(std::ostream& os)
{
    Layer::print(os, "batchnorm", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void BatchNormLayer::forward(const xarray<float>& input)
{
    output_ = input;

    auto b = batch();
    auto c = outChannels();
    auto size = outHeight() * outWidth();

    normalize_cpu(output_.data(), rollingMean_.data(), rollingVar_.data(), b, c, size);

    scale_bias(output_.data(), scales_.data(), b, c, size);
    add_bias(output_.data(), biases_.data(), b, c, size);
}

#ifdef USE_CUDA

void BatchNormLayer::setupGpu()
{
    if (useGpu()) {
        biasesGpu_ = PxDevVector<float>(channels(), 0.f);
        scalesGpu_ = PxDevVector<float>(channels(), 1.f);
        rollingMeanGpu_ = PxDevVector<float>(channels(), 0.f);
        rollingVarGpu_ = PxDevVector<float>(channels(), 0.f);
        outputGpu_ = PxDevVector<float>(batch() * outputs(), 0.f);
        xGpu_ = PxDevVector<float>(batch() * outputs());

        dstTens_ = std::make_unique<CudnnTensorDesc>();
        normTens_ = std::make_unique<CudnnTensorDesc>();

        cudnnSetTensor4dDescriptor(*dstTens_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch(), outChannels(), outHeight(),
                                   outWidth());
        cudnnSetTensor4dDescriptor(*normTens_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outChannels(), 1, 1);
    }
}

void BatchNormLayer::forwardGpu(const PxDevVector<float>& input)
{
    outputGpu_.fromDevice(input);
    xGpu_.fromDevice(outputGpu_);

    float alpha = 1;
    float beta = 0;

    CudnnContext context;

    auto status = cudnnBatchNormalizationForwardInference(context,
                                                          CUDNN_BATCHNORM_SPATIAL,
                                                          &alpha,
                                                          &beta,
                                                          *dstTens_,
                                                          xGpu_.get(),
                                                          *dstTens_,
                                                          outputGpu_.get(),
                                                          *normTens_,
                                                          scalesGpu_.get(),
                                                          biasesGpu_.get(),
                                                          rollingMeanGpu_.get(),
                                                          rollingVarGpu_.get(),
                                                          0.00001);
    PX_CHECK_CUDNN(status);
}

#endif // USE_CUDA

std::streamoff BatchNormLayer::loadDarknetWeights(std::istream& is)
{
    auto start = is.tellg();

    is.read((char*) biases_.data(), int(sizeof(float) * biases_.size()));
    is.read((char*) scales_.data(), int(sizeof(float) * scales_.size()));
    is.read((char*) rollingMean_.data(), int(sizeof(float) * rollingMean_.size()));
    is.read((char*) rollingVar_.data(), int(sizeof(float) * rollingVar_.size()));

#if USE_CUDA
    if (useGpu()) {
        biasesGpu_.fromHost(biases_);
        scalesGpu_.fromHost(scales_);
        rollingMeanGpu_.fromHost(rollingMean_);
        rollingVarGpu_.fromHost(rollingVar_);
    }
#endif

    PX_CHECK(is.good(), "Could not read batch_normalize parameters");

    return is.tellg() - start;
}

}   // px

