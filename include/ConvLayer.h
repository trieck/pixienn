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
#include "BatchNorm.h"
#include "Layer.h"

namespace px {

template<Device D>
class CVExtras
{
};

template<Device D = Device::CPU>
class ConvLayer : public Layer<D>, public CVExtras<D>
{
public:
    using V = typename Layer<D>::V;
    ConvLayer(Model<D>& model, YAML::Node layerDef);

    void forward(const V& input) override;
    void backward(const V& input, V* grad) override;
    void update() override;

    std::streamoff loadWeights(std::istream& is) override;
    virtual std::streamoff saveWeights(std::ostream& os) override;

    void copyWeights(const V& weights);
    void copyBiases(const V& biases);
    void copyMean(const V& mean);
    void copyVariance(const V& var);
    void copyScales(const V& scales);
    void copyRollingMean(const V& rollingMean);
    void copyRollingVariance(const V& rollingVar);

    std::ostream& print(std::ostream& os) override;

private:
    void setup();
    void scaleGradients() override;
    void clipGradients() override;

    Activations<D>::Ptr activation_;

    V weights_, weightUpdates_, biases_, biasUpdates_;
    V scales_, scaleUpdates_, mean_, meanDelta_, var_, varDelta_;
    V rollingMean_, rollingVar_, x_, xNorm_;
    V column_, preActivation_;

    int dilation_, filters_, kernel_, padding_, stride_, groups_;
    bool batchNorm_;
};

template<Device D>
ConvLayer<D>::ConvLayer(Model<D>& model, YAML::Node layerDef) : Layer<D>(model, layerDef)
{
    auto activation = this->template property<std::string>("activation", "logistic");
    activation_ = Activations<D>::get(activation);

    batchNorm_ = this->template property<bool>("batch_normalize", false);
    dilation_ = this->template property<int>("dilation", 1);
    filters_ = this->template property<int>("filters", 1);
    kernel_ = this->template property<int>("kernel", 1);
    auto pad = this->template property<bool>("pad", false);
    padding_ = pad ? kernel_ / 2 : 0;
    stride_ = this->template property<int>("stride", 1);
    groups_ = std::max(1, this->template property<int>("groups", 1));

    this->setOutChannels(filters_);
    this->setOutHeight((this->height() + 2 * padding_ - kernel_) / stride_ + 1);
    this->setOutWidth((this->width() + 2 * padding_ - kernel_) / stride_ + 1);
    this->setOutputs(this->outHeight() * this->outWidth() * this->outChannels());

    if (batchNorm_) {
        scales_ = V(filters_, 1.0f);
        scaleUpdates_ = V(filters_, 0.0f);
        rollingMean_ = V(filters_, 0.0f);
        rollingVar_ = V(filters_, 0.0f);
        x_ = V(this->batch() * this->outputs(), 0.0f);
        xNorm_ = V(this->batch() * this->outputs(), 0.0f);
        mean_ = V(filters_, 0.0f);
        meanDelta_ = V(filters_, 0.0f);
        var_ = V(filters_, 0.0f);
        varDelta_ = V(filters_, 0.0f);
    }

    biases_ = V(filters_, 0.0f);
    biasUpdates_ = V(filters_, 0.0f);

    auto scale = std::sqrt(1.0f / (kernel_ * kernel_ * this->channels() / groups_));

    weights_ = random<V>({ (size_t) filters_ * this->channels() / groups_ * kernel_ * kernel_ }, -scale, scale);
    weightUpdates_ = V(filters_ * this->channels() / groups_ * kernel_ * kernel_, 0.0f);

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);
    this->preActivation_ = V(this->batch() * this->outputs(), 0.0f);

    setup();
}

template<Device D>
void ConvLayer<D>::setup()
{
    column_ = V(this->channels() / groups_ * kernel_ * kernel_ * this->outHeight() * this->outWidth(), 0.0f);
}

template<Device D>
std::streamoff ConvLayer<D>::loadWeights(std::istream& is)
{
    auto start = is.tellg();

    if (batchNorm_) {
        is.read((char*) biases_.data(), biases_.size() * sizeof(float));
        is.read((char*) scales_.data(), scales_.size() * sizeof(float));
        is.read((char*) rollingMean_.data(), rollingMean_.size() * sizeof(float));
        is.read((char*) rollingVar_.data(), rollingVar_.size() * sizeof(float));
    } else {
        is.read((char*) biases_.data(), biases_.size() * sizeof(float));
    }

    is.read((char*) weights_.data(), weights_.size() * sizeof(float));

    PX_CHECK(is.good(), "Could not read weights");

    return is.tellg() - start;
}

template<Device D>
std::streamoff ConvLayer<D>::saveWeights(std::ostream& os)
{
    auto start = os.tellp();

    if (batchNorm_) {
        os.write((char*) biases_.data(), int(sizeof(float) * biases_.size()));
        os.write((char*) scales_.data(), int(sizeof(float) * scales_.size()));
        os.write((char*) rollingMean_.data(), int(sizeof(float) * rollingMean_.size()));
        os.write((char*) rollingVar_.data(), int(sizeof(float) * rollingVar_.size()));
    } else {
        os.write((char*) biases_.data(), int(biases_.size() * sizeof(float)));
        PX_CHECK(os.good(), "Could not write biases");
    }

    os.write((char*) weights_.data(), int(sizeof(float) * weights_.size()));
    PX_CHECK(os.good(), "Could not write weights");

    return os.tellp() - start;
}

template<Device D>
std::ostream& ConvLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "conv", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() },
                    filters_, std::array<int, 3>{ kernel_, kernel_, stride_ });

    return os;
}

template<Device D>
void ConvLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    auto m = filters_ / groups_;
    auto n = this->outWidth() * this->outHeight();
    auto k = kernel_ * kernel_ * this->channels() / groups_;

    auto nweights = weights_.size();
    auto* pweights = weights_.data();

    auto* pin = input.data();
    auto* pout = this->output_.data();

    auto alpha = 1.0f;
    auto beta = 0.0f;

    for (auto i = 0; i < this->batch(); ++i) {
        for (auto j = 0; j < groups_; ++j) {
            const auto* im = pin + (i * groups_ + j) * this->channels() / groups_ * this->height() * this->width();
            const auto* a = pweights + j * nweights / groups_;
            const auto* b = column_.data();
            auto* c = pout + (i * groups_ + j) * n * m;

            im2ColCpuExt(im, this->channels() / groups_, this->height(), this->width(), kernel_, kernel_,
                         padding_ * dilation_, padding_ * dilation_,
                         stride_, stride_, dilation_, dilation_, column_.data());

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);
        }
    }

    if (batchNorm_) {
        batchNormForward(this->training(), this->batch(), this->outChannels(), this->outHeight(), this->outWidth(),
                         this->output_, this->output_, mean_, var_, rollingMean_, rollingVar_, scales_, biases_,
                         x_, xNorm_);
    } else {
        addBias(this->output_.data(), biases_.data(), this->batch(), filters_, this->outHeight() * this->outWidth());
    }

    this->preActivation_.copy(this->output_);
    activation_->apply(this->output_);
}

template<Device D>
void ConvLayer<D>::backward(const V& input, V* grad)
{
    Layer<D>::backward(input, grad);

    activation_->gradient(this->preActivation_, this->delta_);

    if (batchNorm_) {
        batchNormBackward(this->batch(), this->outChannels(), this->outHeight(), this->outWidth(), this->delta_,
                          mean_, var_, meanDelta_, varDelta_, scales_, scaleUpdates_, biasUpdates_, x_, xNorm_);
    } else {
        backwardBias(biasUpdates_.data(), this->delta_.data(), this->batch(), filters_,
                     this->outHeight() * this->outWidth());
    }

    auto m = filters_ / groups_;
    auto n = kernel_ * kernel_ * this->channels() / groups_;
    auto k = this->outHeight() * this->outWidth();

    auto nweights = weights_.size();
    auto* pweights = weights_.data();
    auto* pweightUpdates = weightUpdates_.data();

    const auto* pin = input.data();
    auto* pdelta = this->delta_.data();
    auto* pout = this->output_.data();

    auto alpha = 1.0f;
    auto beta = 1.0f;

    for (auto i = 0; i < this->batch(); ++i) {
        for (auto j = 0; j < groups_; ++j) {
            const auto* im = pin + (i * groups_ + j) * (this->channels() / groups_ * this->height() * this->width());
            const auto* a = pdelta + (i * groups_ + j) * m * k;
            const auto* b = column_.data();
            auto* c = pweightUpdates + j * nweights / groups_;

            im2ColCpuExt(im, this->channels() / groups_, this->height(), this->width(), kernel_, kernel_,
                         padding_ * dilation_, padding_ * dilation_,
                         stride_, stride_, dilation_, dilation_, column_.data());

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, a, k, b, k, beta, c, n);

            if (grad) {
                auto* pgrad = grad->data();
                auto* imd = pgrad + (i * groups_ + j) * (this->channels() / groups_) * this->height()
                                    * this->width();

                a = pweights + j * nweights / groups_;
                b = pdelta + (i * groups_ + j) * m * k;
                c = column_.data();

                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k, m, alpha, a, n, b, k, 0.0f, c, k);

                col2ImCpuExt(c, this->channels() / groups_, this->height(), this->width(), kernel_, kernel_,
                             padding_ * dilation_, padding_ * dilation_, stride_, stride_, dilation_, dilation_, imd);
            }
        }
    }
}

template<Device D>
void ConvLayer<D>::update()
{
    const auto& net = this->model();
    auto learningRate = net.learningRate();
    auto momentum = net.momentum();
    auto decay = net.decay();

    Layer<D>::update();

    cblas_saxpy(weights_.size(), -decay * this->batch(), weights_.data(), 1, weightUpdates_.data(), 1);
    cblas_saxpy(weights_.size(), learningRate / this->batch(), weightUpdates_.data(), 1, weights_.data(), 1);
    cblas_sscal(weights_.size(), momentum, weightUpdates_.data(), 1);

    cblas_saxpy(filters_, learningRate / this->batch(), biasUpdates_.data(), 1, biases_.data(), 1);
    cblas_sscal(filters_, momentum, biasUpdates_.data(), 1);

    if (scales_.size()) {
        cblas_saxpy(filters_, learningRate / this->batch(), scaleUpdates_.data(), 1, scales_.data(), 1);
        cblas_sscal(filters_, momentum, scaleUpdates_.data(), 1);
    }
}

template<Device D>
void ConvLayer<D>::scaleGradients()
{
    Layer<D>::scaleGradients();

    this->scaleTensor(weightUpdates_);
    this->scaleTensor(biasUpdates_);
    this->scaleTensor(scaleUpdates_);
}

template<Device D>
void ConvLayer<D>::clipGradients()
{
    Layer<D>::clipGradients();

    constrain(weightUpdates_.size(), this->gradientClipValue_, weightUpdates_.data(), 1);
    constrain(biasUpdates_.size(), this->gradientClipValue_, biasUpdates_.data(), 1);
    constrain(scaleUpdates_.size(), this->gradientClipValue_, scaleUpdates_.data(), 1);
}

template<Device D>
inline void ConvLayer<D>::copyWeights(const V& weights)
{
    PX_CHECK(weights.size() == weights_.size(), "Invalid weights size");
    weights_.copy(weights);
}

template<Device D>
inline void ConvLayer<D>::copyBiases(const V& biases)
{
    PX_CHECK(biases.size() == biases_.size(), "Invalid biases size");
    biases_.copy(biases);
}

template<Device D>
inline void ConvLayer<D>::copyMean(const V& mean)
{
    PX_CHECK(mean.size() == mean_.size(), "Invalid mean size");
    mean_.copy(mean);
}

template<Device D>
inline void ConvLayer<D>::copyVariance(const V& var)
{
    PX_CHECK(var.size() == var_.size(), "Invalid variance size");
    var_.copy(var);
}

template<Device D>
inline void ConvLayer<D>::copyScales(const V& scales)
{
    PX_CHECK(scales.size() == scales_.size(), "Invalid scales size");
    scales_.copy(scales);
}

template<Device D>
inline void ConvLayer<D>::copyRollingMean(const V& mean)
{
    PX_CHECK(mean.size() == rollingMean_.size(), "Invalid mean size");
    rollingMean_.copy(mean);
}

template<Device D>
inline void ConvLayer<D>::copyRollingVariance(const V& var)
{
    PX_CHECK(var.size() == rollingVar_.size(), "Invalid variance size");
    rollingVar_.copy(var);
}

using CpuConv = ConvLayer<>;
using CudaConv = ConvLayer<Device::CUDA>;

}   // px


#ifdef USE_CUDA

#include "cuda/ConvLayer.h"

#endif  // USE_CUDA