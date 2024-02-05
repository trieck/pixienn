#pragma once

#include "BatchNorm.h"
#include "CpuUtil.h"
#include "Layer.h"

namespace px {

template<Device D>
class BNExtras
{
};

template<Device D = Device::CPU>
class BatchNormLayer : public Layer<D>, public BNExtras<D>
{
public:
    using V = typename Layer<D>::V;
    BatchNormLayer(Model<D>& model, YAML::Node layerDef);

    void forward(const V& input) override;
    void backward(const V& input, V* grad) override;

    std::streamoff loadWeights(std::istream& is) override;
    std::streamoff saveWeights(std::ostream& os) override;

    std::ostream& print(std::ostream& os) override;

private:
    void setup();

    V biases_, biasUpdates_, scales_, scaleUpdates_, mean_, meanDelta_, var_, varDelta_;
    V rollingMean_, rollingVar_, x_, xNorm_;
};

template<Device D>
BatchNormLayer<D>::BatchNormLayer(Model<D>& model, YAML::Node layerDef) : Layer<D>(model, layerDef)
{
    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->outHeight() * this->outWidth() * this->outChannels());

    biases_ = V(this->channels(), 0.f);
    biasUpdates_ = V(this->channels(), 0.f);
    scales_ = V(this->channels(), 1.f);
    scaleUpdates_ = V(this->channels(), 0.f);
    mean_ = V(this->channels(), 0.f);
    meanDelta_ = V(this->channels(), 0.f);
    var_ = V(this->channels(), 0.f);
    varDelta_ = V(this->channels(), 0.f);
    rollingMean_ = V(this->channels(), 0.f);
    rollingVar_ = V(this->channels(), 0.f);
    x_ = V(this->batch() * this->outputs(), 0.f);
    xNorm_ = V(this->batch() * this->outputs(), 0.f);

    this->output_ = V(this->batch() * this->outputs(), 0.f);
    this->delta_ = V(this->batch() * this->outputs(), 0.f);

    setup();
}

template<Device D>
void BatchNormLayer<D>::setup()
{
}

template<Device D>
std::streamoff BatchNormLayer<D>::loadWeights(std::istream& is)
{
    auto start = is.tellg();

    V biases(this->biases_.size());
    V scales(this->scales_.size());
    V rollingMean(this->rollingMean_.size());
    V rollingVar(this->rollingVar_.size());

    is.read((char*) biases.data(), biases.size() * sizeof(float));
    is.read((char*) scales.data(), scales.size() * sizeof(float));
    is.read((char*) rollingMean.data(), rollingMean.size() * sizeof(float));
    is.read((char*) rollingVar.data(), rollingVar.size() * sizeof(float));

    PX_CHECK(is.good(), "Could not read batch normalize parameters");

    this->biases_.copy(biases);
    this->scales_.copy(scales);
    this->rollingMean_.copy(rollingMean);
    this->rollingVar_.copy(rollingVar);

    return is.tellg() - start;
}

template<Device D>
std::streamoff BatchNormLayer<D>::saveWeights(std::ostream& os)
{
    auto start = os.tellp();

    os.write((char*) biases_.data(), int(sizeof(float) * biases_.size()));
    os.write((char*) scales_.data(), int(sizeof(float) * scales_.size()));
    os.write((char*) rollingMean_.data(), int(sizeof(float) * rollingMean_.size()));
    os.write((char*) rollingVar_.data(), int(sizeof(float) * rollingVar_.size()));

    PX_CHECK(os.good(), "Could not write batch normalize parameters");

    return os.tellp() - start;
}

template<Device D>
std::ostream& BatchNormLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "batchnorm", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

template<Device D>
void BatchNormLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    batchNormForward(this->training(), this->batch(), this->outChannels(), this->outHeight(), this->outWidth(),
                     this->output_, this->output_, mean_, var_, rollingMean_, rollingVar_, scales_, biases_,
                     x_, xNorm_);
}

template<Device D>
void BatchNormLayer<D>::backward(const V& input, V* grad)
{
    Layer<D>::backward(input, grad);

    batchNormBackward(this->batch(), this->outChannels(), this->outHeight(), this->outWidth(), this->delta_,
                      mean_, var_, meanDelta_, varDelta_, scales_, scaleUpdates_, biasUpdates_, x_, xNorm_);

    if (grad != nullptr) {
        cblas_scopy(this->batch() * this->outputs(), this->delta_.data(), 1, grad->data(), 1);
    }
}

using CpuBatchNorm = BatchNormLayer<>;
using CudaBatchNorm = BatchNormLayer<Device::CUDA>;

}   // px


#ifdef USE_CUDA

#include "cuda/BatchNormLayer.h"

#endif  // USE_CUDA
