#pragma once

#include "Cudnn.h"

namespace px {

template<>
class BNExtras<Device::CUDA>
{
protected:
    CudnnTensorDesc::Ptr dstTens_, normTens_;
};

template<>
inline void BatchNormLayer<Device::CUDA>::setup()
{
    this->dstTens_ = std::make_unique<CudnnTensorDesc>();
    this->normTens_ = std::make_unique<CudnnTensorDesc>();
}

template<>
inline std::streamoff BatchNormLayer<Device::CUDA>::loadWeights(std::istream& is)
{
    auto start = is.tellg();

    is.read((char*) biases_.data(), int(sizeof(float) * biases_.size()));
    is.read((char*) scales_.data(), int(sizeof(float) * scales_.size()));
    is.read((char*) rollingMean_.data(), int(sizeof(float) * rollingMean_.size()));
    is.read((char*) rollingVar_.data(), int(sizeof(float) * rollingVar_.size()));

    PX_CHECK(is.good(), "Could not read batch normalize parameters");

    return is.tellg() - start;
}

template<>
inline void BatchNormLayer<Device::CUDA>::forward(const PxCudaVector& input)
{
    Layer<Device::CUDA>::forward(input);

    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

    auto exponentialAverageFactor = 0.01f;
    auto epsilon = 0.00001f;
    auto alpha = 1.0f;
    auto beta = 0.0f;

    if (training()) {
        status = cudnnBatchNormalizationForwardTraining(
                cudnnContext(),
                CUDNN_BATCHNORM_SPATIAL,
                &alpha,
                &beta,
                *dstTens_,
                x_.data(),
                *dstTens_,
                output_.data(),
                *normTens_,
                scales_.data(),
                biases_.data(),
                exponentialAverageFactor,
                rollingMean_.data(),
                rollingVar_.data(),
                epsilon,
                mean_.data(),
                var_.data());
    } else {
        status = cudnnBatchNormalizationForwardInference(
                cudnnContext(),
                CUDNN_BATCHNORM_SPATIAL,
                &alpha,
                &beta,
                *dstTens_,
                input.data(),
                *dstTens_,
                output_.data(),
                *normTens_,
                scales_.data(),
                biases_.data(),
                rollingMean_.data(),
                rollingVar_.data(),
                epsilon);
    }

    PX_CHECK_CUDNN(status);
}

template<>
inline void BatchNormLayer<Device::CUDA>::backward(const V& input, V* grad)
{
    Layer<Device::CUDA>::backward(input, grad);

    auto alpha = 1.0f;
    auto beta = 0.0f;
    auto epsilon = 0.00001f;

    auto status = cudnnBatchNormalizationBackward(
            cudnnContext(),
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            &alpha,
            &beta,
            *dstTens_,
            x_.data(),
            *dstTens_,
            delta_.data(),
            *dstTens_,
            xNorm_.data(),
            *normTens_,
            scales_.data(),
            scaleUpdates_.data(),
            biasUpdates_.data(),
            epsilon,
            mean_.data(),
            var_.data());

    PX_CHECK_CUDNN(status);

    delta_.copy(xNorm_);

    if (grad != nullptr) {
        grad->copy(delta_);
    }
}

}   // px