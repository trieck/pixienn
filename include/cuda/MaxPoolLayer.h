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
class MPExtras<Device::CUDA>
{
protected:
    CudnnPoolingDesc::Ptr poolDesc_;
    CudnnTensorDesc::Ptr xDesc_;
    CudnnTensorDesc::Ptr yDesc_;
};

template<>
inline void MaxPoolLayer<Device::CUDA>::setup()
{
    poolDesc_ = std::make_unique<CudnnPoolingDesc>();

    auto status = cudnnSetPooling2dDescriptor(*poolDesc_, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                                              kernel_, kernel_, padding_ / 2, padding_ / 2, stride_, stride_);
    PX_CHECK_CUDNN(status);

    xDesc_ = std::make_unique<CudnnTensorDesc>();
    status = cudnnSetTensor4dDescriptor(*xDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        this->batch(), this->channels(), this->height(), this->width());
    PX_CHECK_CUDNN(status);

    yDesc_ = std::make_unique<CudnnTensorDesc>();
    status = cudnnSetTensor4dDescriptor(*yDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        this->batch(), this->outChannels(), this->outHeight(), this->outWidth());
}

template<>
inline void MaxPoolLayer<Device::CUDA>::forward(const V& input)
{
    auto alpha = 1.0f;
    auto beta = 0.0f;

    auto status = cudnnPoolingForward(this->cudnnContext(), *poolDesc_, &alpha, *xDesc_, input.data(), &beta,
                                      *yDesc_, this->output_.data());
    PX_CHECK_CUDNN(status);
}

template<>
inline void MaxPoolLayer<Device::CUDA>::backward(const V& input)
{
    // TODO: implement
}

}   // px
