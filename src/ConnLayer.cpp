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
    activationFnc_ = Activations::get(activation);

    batchNormalize_ = property<bool>("batch_normalize", false);

    setChannels(inputs());
    setHeight(1);
    setWidth(1);

    setOutputs(property<int>("output", 1));
    setOutHeight(1);
    setOutWidth(1);
    setOutChannels(outputs());

    if (batchNormalize_) {
        scales_ = PxCpuTensor<1>({ (size_t) outputs() }, 1.0f);
        scaleUpdates_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);
        rollingMean_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);
        rollingVar_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);
        x_ = PxCpuVector(batch() * outputs(), 0.0f);
        xNorm_ = PxCpuVector(batch() * outputs(), 0.0f);
        mean_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.f);
        meanDelta_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.f);
        var_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.f);
        varDelta_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.f);
    }

    biases_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);
    biasUpdates_ = PxCpuTensor<1>({ (size_t) outputs() }, 0.0f);

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
    output_ = PxCpuVector(batch() * outputs(), 0.0f);
    delta_ = PxCpuVector(batch() * outputs(), 0.0f);
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
        PX_CHECK(is.good(), "Could not read connected layer parameters");
    }

    return is.tellg() - start;
}

std::streamoff ConnLayer::saveWeights(std::ostream& os)
{
    auto start = os.tellp();

    os.write((char*) biases_.data(), biases_.size() * sizeof(float));
    PX_CHECK(os.good(), "Could not write biases");

    os.write((char*) weights_.data(), sizeof(float) * weights_.size());
    PX_CHECK(os.good(), "Could not write weights");

    if (batchNormalize_) {
        os.write((char*) scales_.data(), outputs() * sizeof(float));
        os.write((char*) rollingMean_.data(), outputs() * sizeof(float));
        os.write((char*) rollingVar_.data(), outputs() * sizeof(float));
        PX_CHECK(os.good(), "Could not write connected layer parameters");
    }

    return os.tellp() - start;
}

void ConnLayer::forward(const PxCpuVector& input)
{
    Layer::forward(input);

    auto ctxt = makeContext(input);
    connectedForward(ctxt);

    if (batchNormalize_) {
        auto bnContext = makeBNContext(output_);
        batchNormForward(bnContext);
    } else {
        addBias(output_.data(), biases_.data(), batch(), outputs(), 1);
    }

    activationFnc_->apply(output_);
}

void ConnLayer::backward(const PxCpuVector& input)
{
    Layer::backward(input);

    activationFnc_->gradient(output_, delta_);

    if (batchNormalize_) {
        auto bnContext = makeBNContext(output_);
        batchNormBackward(bnContext);
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

    // update biases
    cblas_saxpy(outputs(), learningRate / batch(), biasUpdates_.data(), 1, biases_.data(), 1);
    cblas_sscal(outputs(), momentum, biasUpdates_.data(), 1);

    // update scales (if batch normalized)
    if (batchNormalize_) {
        cblas_saxpy(outputs(), learningRate / batch(), scaleUpdates_.data(), 1, scales_.data(), 1);
        cblas_sscal(outputs(), momentum, scaleUpdates_.data(), 1);
    }

    // update weights with weight decay
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
BNContext ConnLayer::makeBNContext(const PxCpuVector& input)
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
