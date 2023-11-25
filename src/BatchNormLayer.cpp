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

#include "BatchNormLayer.h"

namespace px {

BatchNormLayer::BatchNormLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
}

void BatchNormLayer::setup()
{
    setOutChannels(channels());
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(outHeight() * outWidth() * outChannels() * batch());

    biases_ = PxCpuTensor<1>({ (size_t) channels() }, 0.f);
    scales_ = PxCpuTensor<1>({ (size_t) channels() }, 1.f);
    rollingMean_ = PxCpuTensor<1>({ (size_t) channels() }, 0.f);
    rollingVar_ = PxCpuTensor<1>({ (size_t) channels() }, 0.f);

    output_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth());

#ifdef USE_CUDA
    setupGpu();
#endif
}

std::ostream& BatchNormLayer::print(std::ostream& os)
{
    Layer::print(os, "batchnorm", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void BatchNormLayer::forward(const PxCpuVector& input)
{
    auto ctxt = makeContext(input);
    batchNormForward(ctxt);
}

#ifdef USE_CUDA

void BatchNormLayer::setupGpu()
{
    if (useGpu()) {
        biasesGpu_ = PxCudaTensor<1>({ (size_t) channels() }, 0.f);
        scalesGpu_ = PxCudaTensor<1>({ (size_t) channels() }, 1.f);
        rollingMeanGpu_ = PxCudaTensor<1>({ (size_t) channels() }, 0.f);
        rollingVarGpu_ = PxCudaTensor<1>({ (size_t) channels() }, 0.f);
        outputGpu_ = PxCudaVector((size_t) batch() * outputs(), 0.f);
        xGpu_ = PxCudaTensor<1>({ (size_t) batch() * outputs() });

        dstTens_ = std::make_unique<CudnnTensorDesc>();
        normTens_ = std::make_unique<CudnnTensorDesc>();

        cudnnSetTensor4dDescriptor(*dstTens_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch(), outChannels(), outHeight(),
                                   outWidth());
        cudnnSetTensor4dDescriptor(*normTens_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outChannels(), 1, 1);
    }
}

void BatchNormLayer::forwardGpu(const PxCudaVector& input)
{
    float alpha = 1;
    float beta = 0;

    const auto& context = cudnnContext();
    auto status = cudnnBatchNormalizationForwardInference(context,
                                                          CUDNN_BATCHNORM_SPATIAL,
                                                          &alpha,
                                                          &beta,
                                                          *dstTens_,
                                                          input.data(),
                                                          *dstTens_,
                                                          outputGpu_.data(),
                                                          *normTens_,
                                                          scalesGpu_.data(),
                                                          biasesGpu_.data(),
                                                          rollingMeanGpu_.data(),
                                                          rollingVarGpu_.data(),
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
        biasesGpu_.copy(biases_);
        scalesGpu_.copy(scales_);
        rollingMeanGpu_.copy(rollingMean_);
        rollingVarGpu_.copy(rollingVar_);
    }
#endif

    PX_CHECK(is.good(), "Could not read batch_normalize parameters");

    return is.tellg() - start;
}

BNContext BatchNormLayer::makeContext(const PxCpuVector& input)
{
    BNContext ctxt;

    ctxt.input = &input;
    ctxt.output = &output_;
    ctxt.biases = &biases_;
    ctxt.scales = &scales_;
    ctxt.rollingMean = &rollingMean_;
    ctxt.rollingVar = &rollingVar_;

    ctxt.batch = batch();
    ctxt.channels = outChannels();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();

    return ctxt;
}

}   // px

