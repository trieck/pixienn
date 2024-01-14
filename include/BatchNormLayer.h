#pragma once

#include <cblas.h>

#include "CpuUtil.h"
#include "Layer.h"

#ifdef USE_CUDA

#include "Cudnn.h"

#endif  // USE_CUDA

namespace px {

template<Device D>
class BNExtras
{
};

template<>
class BNExtras<Device::CUDA>
{
protected:
    CudnnTensorDesc::Ptr dstTens_, normTens_;
};

template<Device D = Device::CPU>
class BatchNormLayer : public Layer<D>, public BNExtras<D>
{
public:
    using V = typename Layer<D>::V;
    BatchNormLayer(Model<D>& model, YAML::Node layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;

    std::streamoff loadWeights(std::istream& is) override;
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

template<>
inline void BatchNormLayer<Device::CUDA>::setup()
{
    this->dstTens_ = std::make_unique<CudnnTensorDesc>();
    this->normTens_ = std::make_unique<CudnnTensorDesc>();
}

template<Device D>
std::streamoff BatchNormLayer<D>::loadWeights(std::istream& is)
{
    auto start = is.tellg();

    is.read((char*) biases_.data(), biases_.size() * sizeof(float));
    is.read((char*) scales_.data(), scales_.size() * sizeof(float));
    is.read((char*) rollingMean_.data(), rollingMean_.size() * sizeof(float));
    is.read((char*) rollingVar_.data(), rollingVar_.size() * sizeof(float));

    PX_CHECK(is.good(), "Could not read weights");

    return is.tellg() - start;
}

template<>
inline std::streamoff BatchNormLayer<Device::CUDA>::loadWeights(std::istream& is)
{
    auto start = is.tellg();

    // TODO: implement CUDA weights loading

    return is.tellg() - start;
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

    if (input.data() != this->output_.data()) {
        this->output_.copy(input);
    }

    auto b = this->batch();
    auto c = this->channels();
    auto size = this->outHeight() * this->outWidth();
    auto outputs = c * size;

    if (this->training()) {
        cblas_scopy(b * outputs, this->output_.data(), 1, x_.data(), 1);

        meanCpu(x_.data(), b, c, size, mean_.data());
        varianceCpu(this->output_.data(), mean_.data(), b, c, size, var_.data());

        cblas_sscal(c, 0.99f, rollingMean_.data(), 1);
        cblas_saxpy(c, 0.01f, mean_.data(), 1, rollingMean_.data(), 1);
        cblas_sscal(c, 0.99f, rollingVar_.data(), 1);
        cblas_saxpy(c, 0.01f, var_.data(), 1, rollingVar_.data(), 1);

    } else {
        normalizeCpu(this->output_.data(), this->rollingMean_.data(), this->rollingVar_.data(), b, c, size);
    }

    scaleBias(this->output_.data(), this->scales_.data(), b, c, size);
    addBias(this->output_.data(), this->biases_.data(), b, c, size);
}

template<Device D>
void BatchNormLayer<D>::backward(const V& input)
{
    Layer<D>::backward(input);

    backwardBias(biasUpdates_.data(), this->delta_.data(), this->batch(), this->channels(),
                 this->outHeight() * this->outWidth());

    backwardScaleCpu(xNorm_.data(), this->delta_.data(), this->batch(), this->channels(),
                     this->outHeight() * this->outWidth(), scaleUpdates_.data());

    scaleBias(this->delta_.data(), this->scales_.data(), this->batch(), this->channels(),
              this->outHeight() * this->outWidth());

    meanDeltaCpu(this->delta_.data(), this->var_.data(), this->batch(), this->channels(),
                 this->outHeight() * this->outWidth(), meanDelta_.data());

    varianceDeltaCpu(x_.data(), this->delta_.data(), this->mean_.data(), this->var_.data(), this->batch(),
                     this->channels(), this->outHeight() * this->outWidth(), var_.data());

    normalizeDeltaCpu(this->x_.data(), this->mean_.data(), this->var_.data(), this->meanDelta_.data(),
                      this->varDelta_.data(), this->batch(), this->channels(), this->outHeight() * this->outWidth(),
                      this->delta_.data());
}

using CpuBatchNorm = BatchNormLayer<>;
using CudaBatchNorm = BatchNormLayer<Device::CUDA>;

}   // px


#ifdef USE_CUDA

#include "BatchNormCuda.h"

#endif  // USE_CUDA
