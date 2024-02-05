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

#include "Cudnn.h"

namespace px {

template<>
class APExtras<Device::CUDA>
{
protected:
    CudnnPoolingDesc::Ptr poolDesc_;
    CudnnTensorDesc::Ptr xDesc_, yDesc_, dxDesc_, dyDesc_;
};

template<>
inline void AvgPoolLayer<Device::CUDA>::setup()
{
    poolDesc_ = std::make_unique<CudnnPoolingDesc>();

    auto status = cudnnSetPooling2dDescriptor(*poolDesc_, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                              CUDNN_NOT_PROPAGATE_NAN, this->height(), this->width(), 0, 0, 1, 1);
    PX_CHECK_CUDNN(status);

    xDesc_ = std::make_unique<CudnnTensorDesc>();
    status = cudnnSetTensor4dDescriptor(*xDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        this->batch(), this->channels(), this->height(), this->width());
    PX_CHECK_CUDNN(status);

    yDesc_ = std::make_unique<CudnnTensorDesc>();
    status = cudnnSetTensor4dDescriptor(*yDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        this->batch(), this->outChannels(), this->outHeight(), this->outWidth());
    PX_CHECK_CUDNN(status);

    dxDesc_ = std::make_unique<CudnnTensorDesc>();
    status = cudnnSetTensor4dDescriptor(*dxDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        this->batch(), this->channels(), this->height(), this->width());
    PX_CHECK_CUDNN(status);

    dyDesc_ = std::make_unique<CudnnTensorDesc>();
    status = cudnnSetTensor4dDescriptor(*dyDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        this->batch(), this->outChannels(), this->outHeight(), this->outWidth());
    PX_CHECK_CUDNN(status);
}

template<>
inline void AvgPoolLayer<Device::CUDA>::forward(const V& input)
{
    auto alpha = 1.0f;
    auto beta = 0.0f;

    Layer<Device::CUDA>::forward(input);

    auto status = cudnnPoolingForward(this->cudnnContext(), *poolDesc_, &alpha, *xDesc_, input.data(), &beta,
                                      *yDesc_, this->output_.data());
    PX_CHECK_CUDNN(status);
}

template<>
inline void AvgPoolLayer<Device::CUDA>::backward(const V& input, V* grad)
{
    if (grad == nullptr) {
        return;
    }

    auto alpha = 1.0f;
    auto beta = 0.0f;

    Layer<Device::CUDA>::backward(input, grad);

    auto status = cudnnPoolingBackward(this->cudnnContext(), *poolDesc_, &alpha, *yDesc_, this->output_.data(),
                                       *dyDesc_, delta_.data(), *xDesc_, input.data(), &beta,
                                       *dxDesc_, grad->data());
    PX_CHECK_CUDNN(status);
}

}   // px
