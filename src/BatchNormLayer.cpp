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

BatchNormLayer::BatchNormLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
}

void BatchNormLayer::setup()
{
    setOutChannels(channels());
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(outHeight() * outWidth() * outChannels());

    biases_ = PxCpuTensor<1>({ (size_t) channels() }, 0.0f);
    biasUpdates_ = PxCpuTensor<1>({ (size_t) channels() }, 0.0f);
    scales_ = PxCpuTensor<1>({ (size_t) channels() }, 1.f);
    scaleUpdates_ = PxCpuTensor<1>({ (size_t) channels() }, 0.f);
    mean_ = PxCpuTensor<1>({ (size_t) channels() }, 0.f);
    meanDelta_ = PxCpuTensor<1>({ (size_t) channels() }, 0.f);
    var_ = PxCpuTensor<1>({ (size_t) channels() }, 0.f);
    varDelta_ = PxCpuTensor<1>({ (size_t) channels() }, 0.f);
    rollingMean_ = PxCpuTensor<1>({ (size_t) channels() }, 0.f);
    rollingVar_ = PxCpuTensor<1>({ (size_t) channels() }, 0.f);

#ifdef USE_CUDA
    if (useGpu()) {
        setupGpu();
    } else {
        output_ = PxCpuVector(batch() * outputs(), 0.0f);
        delta_ = PxCpuVector(batch() * outputs(), 0.0f);
        x_ = PxCpuVector(batch() * outputs(), 0.0f);
        xNorm_ = PxCpuVector(batch() * outputs(), 0.0f);
    }
#else
    output_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth(), 0.0f);
    delta_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth(), 0.0f);
    x_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth(), 0.0f);
    xNorm_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth(), 0.0f);
#endif
}

std::ostream& BatchNormLayer::print(std::ostream& os)
{
    Layer::print(os, "batchnorm", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void BatchNormLayer::forward(const PxCpuVector& input)
{
    Layer::forward(input);

    auto ctxt = makeContext(input);
    batchNormForward(ctxt);
}

void BatchNormLayer::backward(const PxCpuVector& input)
{
    auto ctxt = makeContext(input);
    batchNormBackward(ctxt);
}

#ifdef USE_CUDA

void BatchNormLayer::setupGpu()
{
    biasesGpu_ = PxCudaTensor<1>({ (size_t) channels() }, 0.f);
    biasUpdatesGpu_ = PxCudaTensor<1>({ (size_t) channels() }, 0.f);
    scalesGpu_ = PxCudaTensor<1>({ (size_t) channels() }, 1.f);
    scaleUpdatesGpu_ = PxCudaTensor<1>({ (size_t) channels() }, 0.f);
    rollingMeanGpu_ = PxCudaTensor<1>({ (size_t) channels() }, 0.f);
    rollingVarGpu_ = PxCudaTensor<1>({ (size_t) channels() }, 0.f);
    outputGpu_ = PxCudaVector((size_t) batch() * outputs(), 0.f);
    dstTens_ = std::make_unique<CudnnTensorDesc>();
    normTens_ = std::make_unique<CudnnTensorDesc>();

    cudnnSetTensor4dDescriptor(*dstTens_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch(), outChannels(), outHeight(),
                               outWidth());
    cudnnSetTensor4dDescriptor(*normTens_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outChannels(), 1, 1);
}

void BatchNormLayer::forwardGpu(const PxCudaVector& input)
{
    auto ctxt = makeContext(input);
    batchNormForwardGpu(ctxt);
}

#endif // USE_CUDA

std::streamoff BatchNormLayer::loadWeights(std::istream& is)
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
    ctxt.x = &x_;
    ctxt.xNorm = &xNorm_;
    ctxt.biases = &biases_;
    ctxt.biasUpdates = &biasUpdates_;
    ctxt.delta = &delta_;
    ctxt.meanDelta = &meanDelta_;
    ctxt.scales = &scales_;
    ctxt.scaleUpdates = &scaleUpdates_;
    ctxt.mean = &mean_;
    ctxt.var = &var_;
    ctxt.varDelta = &varDelta_;
    ctxt.rollingMean = &rollingMean_;
    ctxt.rollingVar = &rollingVar_;

    ctxt.batch = batch();
    ctxt.channels = outChannels();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();

    ctxt.training = training();

    return ctxt;
}

#ifdef USE_CUDA

BNContext BatchNormLayer::makeContext(const PxCudaVector& input)
{
    BNContext ctxt;

    ctxt.inputGpu = &input;
    ctxt.outputGpu = &outputGpu_;
    ctxt.xGpu = &xGpu_;
    ctxt.xNormGpu = &xNormGpu_;
    ctxt.biasesGpu = &biasesGpu_;
    ctxt.biasUpdatesGpu = &biasUpdatesGpu_;
    ctxt.deltaGpu = &deltaGpu_;
    ctxt.meanDeltaGpu = &meanDeltaGpu_;
    ctxt.scalesGpu = &scalesGpu_;
    ctxt.scalesUpdatesGpu = &scaleUpdatesGpu_;
    ctxt.rollingMeanGpu = &rollingMeanGpu_;
    ctxt.rollingVarGpu = &rollingVarGpu_;
    ctxt.cudnnContext = &cudnnContext();
    ctxt.dstTens = dstTens_.get();
    ctxt.normTens = normTens_.get();

    ctxt.batch = batch();
    ctxt.channels = outChannels();
    ctxt.outHeight = outHeight();
    ctxt.outWidth = outWidth();

    ctxt.training = training();

    return ctxt;
}

#endif  // USE_CUDA

}   // px

