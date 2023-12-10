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

#include <cblas.h>

#include "ConnLayer.h"
#include "CpuUtil.h"
#include "Model.h"

#if USE_CUDA

#include "BiasKernels.cuh"

#endif

namespace px {

ConnLayer::ConnLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
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
        scales_ = PxCpuTensor<1>({ (size_t) outputs() }, 1.0f);
        scaleUpdates_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);
        rollingMean_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);
        rollingVar_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);
    } else {
        biases_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);
        biasUpdates_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);
    }

    auto scale = std::sqrt(2.0f / inputs());
    weights_ = random<PxCpuTensor<2>>({ (size_t) inputs(), (size_t) outputs() }, -1.0f, 1.0f) * scale;
    weightUpdates_ = PxCpuTensor<2>({ (size_t) inputs(), (size_t) outputs() }, 0.0f);

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
        is.read((char*) scales_.data(), outputs() * sizeof(float));
        is.read((char*) rollingMean_.data(), outputs() * sizeof(float));
        is.read((char*) rollingVar_.data(), outputs() * sizeof(float));
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
        output_.copy(batchNormalize_->output());
    } else {
        addBias(output_.data(), biases_.data(), batch(), outputs(), 1);
    }

    activationFnc_->apply(output_);
}

void ConnLayer::backward(const PxCpuVector& input)
{
    constrain(outputs() * batch(), 1, delta_.data(), 1);

    activationFnc_->gradient(output_, delta_);

    if (batchNormalize_) {
        batchNormalize_->backward(output_);
        output_.copy(batchNormalize_->output());
    } else {
        backwardBias(biasUpdates_.data(), delta_.data(), batch(), outputs(), 1);
    }

    auto ctxt = makeContext(input);
    connectedBackward(ctxt);
}

void ConnLayer::update()
{
    const auto& net = model();
    auto learningRate = net.learningRate();
    auto momentum = net.momentum();
    auto decay = net.decay();

    cblas_saxpy(outputs(), learningRate / batch(), biasUpdates_.data(), 1, biases_.data(), 1);
    cblas_sscal(outputs(), momentum, biasUpdates_.data(), 1);

    if (batchNormalize_) {
        cblas_saxpy(outputs(), learningRate / batch(), scaleUpdates_.data(), 1, scales_.data(), 1);
        cblas_sscal(outputs(), momentum, scaleUpdates_.data(), 1);
    }

    auto size = inputs() * outputs();
    cblas_saxpy(size, -decay * batch(), weights_.data(), 1, weightUpdates_.data(), 1);
    cblas_saxpy(size, learningRate / batch(), weightUpdates_.data(), 1, weights_.data(), 1);
    cblas_sscal(size, momentum, weightUpdates_.data(), 1);
}

ConnContext ConnLayer::makeContext(const PxCpuVector& input)
{
    ConnContext ctxt;
    ctxt.input = &input;
    ctxt.output = &output_;
    ctxt.delta = &delta_;
    ctxt.netDelta = model().delta();
    ctxt.weights = &weights_;
    ctxt.weightUpdates = &weightUpdates_;
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
