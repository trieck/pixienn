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

#include "ConnLayer.h"
#include "CpuUtil.h"

#if USE_CUDA

#include "BiasKernels.cuh"

#endif

namespace px {

ConnLayer::ConnLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef), scales_(0),
                                                                 rollingMean_(0), rollingVar_(0)
{
}

void ConnLayer::setup()
{
    auto activation = property<std::string>("activation", "logistic");
    activationFnc_ = Activation::get(activation);

    auto batchNormalize = property<bool>("batch_normalize", false);

    setChannels(inputs());
    setHeight(1);
    setWidth(1);

    setOutputs(property<int>("output", 1));
    setOutHeight(1);
    setOutWidth(1);
    setOutChannels(outputs());

    if (batchNormalize) {
        auto def = layerDef();
        def["type"] = "batchnorm";
        def["channels"] = outChannels();
        def["height"] = outHeight();
        def["width"] = outWidth();
        batchNormalize_ = Layer::create(model(), def);
    } else {
        biases_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);
        biasUpdates_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);
    }

    weights_ = random<decltype(weights_)>({ (size_t) inputs(), (size_t) outputs() });

#ifdef USE_CUDA
    if (useGpu()) {
        setupGpu();
    } else {
        output_ = PxCpuVector(batch() * outputs());
        delta_ = PxCpuVector(batch() * outputs());
    }
#else
    output_ = PxCpuVector(batch() * outputs());
    delta_ = PxCpuVector(batch() * outputs());
#endif
}

#ifdef USE_CUDA // USE_CUDA

void ConnLayer::setupGpu()
{
    if (!batchNormalize_) {
        biasesGpu_ = PxCudaTensor<1>({ (size_t) outputs() }, 0.f);
    }

    weightsGpu_ = random<decltype(weightsGpu_)>({ (size_t) inputs(), (size_t) outputs() });
    outputGpu_ = PxCudaVector(batch() * outputs(), 0.f);
}

#endif // USE_CUDA

std::ostream& ConnLayer::print(std::ostream& os)
{
    Layer::print(os, "connected", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

std::streamoff ConnLayer::loadWeights(std::istream& is)
{
    auto start = is.tellg();

    is.read((char*) biases_.data(), biases_.size() * sizeof(float));
    PX_CHECK(is.good(), "Could not read biases");

    is.read((char*) weights_.data(), sizeof(float) * weights_.size());
    PX_CHECK(is.good(), "Could not read weights");

#ifdef USE_CUDA
    if (useGpu()) {
        biasesGpu_.copy(biases_);
        weightsGpu_.copy(weights_);
    }
#endif

    if (batchNormalize_) {
        is.read((char*) &scales_, sizeof(float));
        is.read((char*) &rollingMean_, sizeof(float));
        is.read((char*) &rollingVar_, sizeof(float));
        PX_CHECK(is.good(), "Could not read batch_normalize parameters");
    }

    return is.tellg() - start;
}

void ConnLayer::forward(const PxCpuVector& input)
{
    auto ctxt = makeContext(input);
    connectedForward(ctxt);

    if (batchNormalize_) {
        batchNormalize_->forward(output_);
        output_ = batchNormalize_->output();
    } else {
        addBias(output_.data(), biases_.data(), batch(), outChannels(), outHeight() * outWidth());
    }

    activationFnc_->apply(output_);
}

void ConnLayer::backward(const PxCpuVector& input)
{
    auto ctxt = makeContext(input);

    activationFnc_->gradient(output_, delta_);

    if (batchNormalize_) {
        batchNormalize_->backward(output_);
        output_ = batchNormalize_->output();
    } else {
        backwardBias(biasUpdates_.data(), delta_.data(), batch(), outChannels(), outHeight() * outWidth());
    }

    connectedBackward(ctxt);
}

ConnContext ConnLayer::makeContext(const PxCpuVector& input)
{
    ConnContext ctxt;
    ctxt.input = &input;
    ctxt.output = &output_;
    ctxt.weights = &weights_;
    ctxt.batch = batch();
    ctxt.inputs = inputs();
    ctxt.outputs = outputs();

    return ctxt;
}

#if USE_CUDA

void ConnLayer::forwardGpu(const PxCudaVector& input)
{
    auto ctxt = makeContext(input);
    connectedForwardGpu(ctxt);

    if (batchNormalize_) {
        batchNormalize_->forwardGpu(outputGpu_);
        outputGpu_ = batchNormalize_->outputGpu();
    } else {
        addBiasGpu(outputGpu_.data(), biasesGpu_.data(), batch(), outputs(), 1);
    }

    activationFnc_->applyGpu(outputGpu_);
}

ConnContext ConnLayer::makeContext(const PxCudaVector& input)
{
    ConnContext ctxt;
    ctxt.cublasContext = &cublasContext();
    ctxt.inputGpu = &input;
    ctxt.outputGpu = &outputGpu_;
    ctxt.weightsGpu = &weightsGpu_;
    ctxt.batch = batch();
    ctxt.inputs = inputs();
    ctxt.outputs = outputs();

    return ctxt;
}

#endif  // USE_CUDA

} // px
