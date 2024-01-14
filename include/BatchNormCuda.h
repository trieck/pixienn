#pragma once

namespace px {

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
inline void BatchNormLayer<Device::CUDA>::backward(const px::PxCudaVectorT<float>& input)
{
    Layer<Device::CUDA>::backward(input);

    auto alpha = 1.0f;
    auto beta = 0.0f;
    auto epsilon = 0.00001f;

    auto status = cudnnBatchNormalizationBackward(
            cudnnContext(),
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            &alpha,
            &alpha,
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
}

}   // px