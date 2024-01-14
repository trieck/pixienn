#pragma once

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
    void backward(const V& input) override;

    std::streamoff loadWeights(std::istream& is) override;
    std::ostream& print(std::ostream& os) override;

private:
    void setup();

    Activations<D>::Ptr activation_;

    V weights_, weightUpdates_, biases_, biasUpdates_;
    V scales_, scaleUpdates_, mean_, meanDelta_, var_, varDelta_;
    V rollingMean_, rollingVar_, x_, xNorm_;
    V column_;

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

    weights_ = random<V>({ (size_t) filters_ * this->channels() / groups_ * kernel_ * kernel_ },
                         -1.0f * scale, 1.0f * scale);
    weightUpdates_ = V(filters_ * this->channels() / groups_ * kernel_ * kernel_, 0.0f);

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);

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

    auto* pin = const_cast<float*>(input.data()); // FIXME: don't do this
    auto* pout = this->output_.data();

    auto alpha = 1.0f;
    auto beta = 1.0f;

    for (auto i = 0; i < this->batch(); ++i) {
        for (auto j = 0; j < groups_; ++j) {
            auto* im = pin + (i * groups_ + j) * this->channels() / groups_ * this->height() * this->width();
            const auto* a = pweights + j * nweights / groups_;
            auto* b = column_.data();
            auto* c = pout + (i * groups_ + j) * n * m;

            if (kernel_ == 1 && stride_ == 1 && dilation_ == 1) {
                b = im;
            }

            im2ColCpuExt(im, this->channels() / groups_, this->height(), this->width(), kernel_, kernel_,
                         padding_ * dilation_, padding_ * dilation_,
                         stride_, stride_, dilation_, dilation_, b);

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

    activation_->apply(this->output_);
}

template<Device D>
void ConvLayer<D>::backward(const V& input)
{
    Layer<D>::backward(input);

    activation_->gradient(this->output_, this->delta_);

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

    auto* pin = const_cast<float*>(input.data()); // FIXME: don't do this
    auto* pdelta = this->delta_.data();
    auto* pNetDelta = this->model().delta()->data();
    auto* pout = this->output_.data();

    auto alpha = 1.0f;
    auto beta = 1.0f;

    for (auto i = 0; i < this->batch(); ++i) {
        for (auto j = 0; j < groups_; ++j) {
            auto* im = pin + (i * groups_ + j) * this->channels() / groups_ * this->height() * this->width();
            const auto* a = pdelta + (i * groups_ + j) * m * k;
            auto* b = column_.data();
            auto* c = pweightUpdates + j * nweights / groups_;

            if (kernel_ == 1 && stride_ == 1 && dilation_ == 1) {
                b = im;
            }

            im2ColCpuExt(im, this->channels() / groups_, this->height(), this->width(), kernel_, kernel_,
                         padding_ * dilation_, padding_ * dilation_,
                         stride_, stride_, dilation_, dilation_, b);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, a, k, b, k, beta, c, n);

            if (pNetDelta) {
                a = pweights + j * nweights / groups_;
                b = pdelta + (i * groups_ + j) * m * k;
                c = column_.data();

                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k, m, alpha, a, n, b, k, beta, c, k);

                col2ImCpuExt(column_.data(), this->channels() / groups_, this->height(), this->width(), kernel_,
                             kernel_, padding_ * dilation_, padding_ * dilation_,
                             stride_, stride_, dilation_, dilation_, pNetDelta);
            }
        }
    }
}

using CpuConv = ConvLayer<>;
using CudaConv = ConvLayer<Device::CUDA>;

}   // px


#ifdef USE_CUDA

#include "cuda/ConvLayer.h"

#endif  // USE_CUDA