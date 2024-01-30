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

#include "Cublas.h"
#include "Cudnn.h"
#include "SoftmaxKernel.cuh"

namespace px {

template<>
class SMExtras<Device::CUDA>
{
protected:
    CudnnTensorDesc::Ptr xDesc_, yDesc_;
};

template<>
inline void SoftmaxLayer<Device::CUDA>::setup()
{
    xDesc_ = std::make_unique<CudnnTensorDesc>();
    auto status = cudnnSetTensor4dDescriptor(*xDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                             this->batch(), this->channels(), this->height(), this->width());
    PX_CHECK_CUDNN(status);

    yDesc_ = std::make_unique<CudnnTensorDesc>();
    status = cudnnSetTensor4dDescriptor(*yDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        this->batch(), this->outChannels(), this->outHeight(), this->outWidth());
    PX_CHECK_CUDNN(status);
}

template<>
inline void SoftmaxLayer<Device::CUDA>::computeLoss()
{
    PxCpuVector truth(this->batch() * this->outputs(), 0.0f);

    for (auto b = 0; b < this->batch(); ++b) {
        const auto& gts = this->groundTruth(b);
        auto* ptruth = truth.data() + b * this->outputs();

        for (const auto& gt: gts) {
            auto index = gt.classId;
            ptruth[index] = 1.0f;
        }
    }

    PxCudaVector truthGpu(&(*truth.begin()), &(*truth.end()));

    softmaxCrossEntropy(this->batch() * this->outputs(), output_.data(), truthGpu.data(), this->delta_.data(),
                        loss_.data());
}

template<>
inline void SoftmaxLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);

    auto alpha = 1.0f;
    auto beta = 0.0f;

    auto status = cudnnSoftmaxForward(this->cudnnContext(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                      &alpha, *xDesc_, input.data(), &beta, *yDesc_, output_.data());
    PX_CHECK_CUDNN(status);

    if (training()) {
        computeLoss();

        auto result = cublasSasum(this->cublasContext(), loss_.size(), loss_.data(), 1, &this->cost_);
        PX_CHECK_CUBLAS(result);
    }
}

template<>
inline void SoftmaxLayer<Device::CUDA>::backward(const V& input)
{
    Layer<Device::CUDA>::backward(input);

    auto alpha = 1.0f;
    auto beta = 0.0f;

    const auto& ctxt = this->cudnnContext();

    auto status = cudnnSoftmaxBackward(ctxt, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                       &alpha, *yDesc_, output_.data(), *yDesc_, this->delta_.data(), &beta,
                                       *xDesc_, this->netDelta()->data());

    PX_CHECK_CUDNN(status);
}

template<>
inline void SoftmaxLayer<Device::CUDA>::addDetects(Detections& detections, float threshold)
{
    auto predictions = output_.data();
    addDetects(detections, 0, 0, threshold, predictions);
}

template<>
inline void SoftmaxLayer<Device::CUDA>::addDetects(Detections& detections, int width, int height, float threshold)
{
    auto predictions = output_.data();
    addDetects(detections, width, height, threshold, predictions);
}

} // px
