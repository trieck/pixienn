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

#pragma once

#include <cblas.h>

#include "Activation.h"
#include "Layer.h"

namespace px {
template<Device D>
class FCExtras
{
};


template<Device D = Device::CPU>
class ConnLayer : public Layer<D>, public FCExtras<D>
{
public:
    using V = typename Layer<D>::V;

    ConnLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;
    void update() override;

    std::streamoff loadWeights(std::istream& is) override;
    std::streamoff saveWeights(std::ostream& os) override;

    std::ostream& print(std::ostream& os) override;

private:
    void setup();

    V weights_, weightUpdates_, biases_, biasUpdates_;
    V scales_, scaleUpdates_, mean_, meanDelta_, var_, varDelta_;
    V rollingMean_, rollingVar_, x_, xNorm_;

    Activations<D>::Ptr activation_;
    bool batchNorm_;
};

template<Device D>
ConnLayer<D>::ConnLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    auto activation = this->template property<std::string>("activation", "logistic");
    activation_ = Activations<D>::get(activation);

    batchNorm_ = this->template property<bool>("batch_normalize", false);

    this->setChannels(this->inputs());
    this->setHeight(1);
    this->setWidth(1);

    this->setOutputs(this->template property<int>("output", 1));
    this->setOutHeight(1);
    this->setOutWidth(1);
    this->setOutChannels(this->outputs());

    if (batchNorm_) {
        this->scales_ = V(this->outputs(), 1.0f);
        this->scaleUpdates_ = V(this->outputs(), 0.0f);
        this->rollingMean_ = V(this->outputs(), 0.0f);
        this->rollingVar_ = V(this->outputs(), 0.0f);
        this->x_ = V(this->batch() * this->outputs(), 0.0f);
        this->xNorm_ = V(this->batch() * this->outputs(), 0.0f);
        this->mean_ = V(this->outputs(), 0.f);
        this->meanDelta_ = V(this->outputs(), 0.f);
        this->var_ = V(this->outputs(), 0.f);
    }

    this->biases_ = V(this->outputs(), 0.0f);
    this->biasUpdates_ = V(this->outputs(), 0.0f);

    auto scale = std::sqrt(2.0f / this->inputs());
    this->weights_ = random<V>(this->inputs() * this->outputs(), -1.0f * scale, 1.0f * scale);
    this->weightUpdates_ = V(this->inputs() * this->outputs(), 0.0f);

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);

    setup();
}

template<Device D>
void ConnLayer<D>::setup()
{
}

template<Device D>
std::streamoff ConnLayer<D>::loadWeights(std::istream& is)
{
    auto start = is.tellg();

    is.read((char*) biases_.data(), biases_.size() * sizeof(float));
    is.read((char*) weights_.data(), weights_.size() * sizeof(float));

    if (batchNorm_) {
        is.read((char*) scales_.data(), scales_.size() * sizeof(float));
        is.read((char*) rollingMean_.data(), rollingMean_.size() * sizeof(float));
        is.read((char*) rollingVar_.data(), rollingVar_.size() * sizeof(float));
    }

    PX_CHECK(is.good(), "Could not read weights");

    return is.tellg() - start;
}

template<Device D>
std::streamoff ConnLayer<D>::saveWeights(std::ostream& os)
{
    auto start = os.tellp();

    os.write((char*) biases_.data(), biases_.size() * sizeof(float));
    PX_CHECK(os.good(), "Could not write biases");

    os.write((char*) weights_.data(), sizeof(float) * weights_.size());
    PX_CHECK(os.good(), "Could not write weights");

    if (batchNorm_) {
        os.write((char*) scales_.data(), this->outputs() * sizeof(float));
        os.write((char*) rollingMean_.data(), this->outputs() * sizeof(float));
        os.write((char*) rollingVar_.data(), this->outputs() * sizeof(float));
        PX_CHECK(os.good(), "Could not write connected layer parameters");
    }

    return os.tellp() - start;
}

template<Device D>
std::ostream& ConnLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "connected", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

template<Device D>
void ConnLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    auto m = this->batch();
    auto n = this->outputs();
    auto k = this->inputs();
    auto* a = input.data();
    auto* b = this->weights_.data();
    auto* c = this->output_.data();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a, k, b, k, 1.0f, c, n);

    if (batchNorm_) {
        batchNormForward(this->training(), this->batch(), this->outChannels(), this->outHeight(), this->outWidth(),
                         this->output_, this->output_, mean_, var_, rollingMean_, rollingVar_, scales_, biases_,
                         x_, xNorm_);
    } else {
        addBias(this->output_.data(), this->biases_.data(), this->batch(), this->outputs(), 1);
    }

    activation_->apply(this->output_);
}


template<Device D>
void ConnLayer<D>::backward(const V& input)
{
    Layer<D>::backward(input);

    activation_->gradient(this->output_, this->delta_);

    if (batchNorm_) {
        batchNormBackward(this->batch(), this->outChannels(), this->outHeight(), this->outWidth(), this->delta_,
                          mean_, var_, meanDelta_, varDelta_, scales_, scaleUpdates_, biasUpdates_, x_, xNorm_);
    } else {
        backwardBias(this->biasUpdates_.data(), this->delta_.data(), this->batch(), this->outputs(), 1);
    }

    auto m = this->outputs();
    auto n = this->inputs();
    auto k = this->batch();
    auto* a = this->delta_.data();
    auto* b = input.data();
    auto* c = this->weightUpdates_.data();

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0f, a, m, b, n, 1.0f, c, n);

    m = this->batch();
    k = this->outputs();
    n = this->inputs();
    b = this->weights_.data();
    c = this->model().delta()->data();

    if (c) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
    }
}

template<Device D>
void ConnLayer<D>::update()
{
    const auto& net = this->model();
    auto learningRate = net.learningRate();
    auto momentum = net.momentum();
    auto decay = net.decay();

    // update biases
    cblas_saxpy(this->outputs(), learningRate / this->batch(), biasUpdates_.data(), 1, biases_.data(), 1);
    cblas_sscal(this->outputs(), momentum, biasUpdates_.data(), 1);

    // update scales (if batch normalized)
    if (batchNorm_) {
        cblas_saxpy(this->outputs(), learningRate / this->batch(), scaleUpdates_.data(), 1, scales_.data(), 1);
        cblas_sscal(this->outputs(), momentum, scaleUpdates_.data(), 1);
    }

    // update weights with weight decay
    auto size = this->inputs() * this->outputs();

    cblas_saxpy(size, -decay * this->batch(), weights_.data(), 1, weightUpdates_.data(), 1);
    cblas_saxpy(size, learningRate / this->batch(), weightUpdates_.data(), 1, weights_.data(), 1);
    cblas_sscal(size, momentum, weightUpdates_.data(), 1);
}

using CpuConn = ConnLayer<>;
using CudaConn = ConnLayer<Device::CUDA>;

} // px

#ifdef USE_CUDA

#include "cuda/ConnLayer.h"

#endif
