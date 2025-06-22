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

namespace px {

template<>
class DropoutExtras<Device::CUDA>
{
protected:
    using V = typename Layer<Device::CUDA>::V;

    CudnnDropoutDesc::Ptr dropoutDesc_;
    CudnnTensorDesc::Ptr inputDesc_, outputDesc_;

    V states_, reserves_;

    size_t stateSize_ = 0;
    size_t reserveSize_ = 0;
};

template<>
inline void DropoutLayer<Device::CUDA>::setup()
{
    auto dropoutProb = this->probability_;
    size_t seed = 1337;

    dropoutDesc_ = std::make_unique<CudnnDropoutDesc>();

    const auto& ctxt = this->cudnnContext();
    auto status = cudnnDropoutGetStatesSize(ctxt, &stateSize_);
    PX_CHECK_CUDNN(status);

    states_ = V(stateSize_);
    status = cudnnSetDropoutDescriptor(*dropoutDesc_, ctxt, dropoutProb, states_.data(), stateSize_, seed);
    PX_CHECK_CUDNN(status);

    inputDesc_ = std::make_unique<CudnnTensorDesc>();
    outputDesc_ = std::make_unique<CudnnTensorDesc>();

    status = cudnnSetTensor4dDescriptor(*inputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        this->batch(), this->channels(), this->height(), this->width());
    PX_CHECK_CUDNN(status);

    status = cudnnSetTensor4dDescriptor(*outputDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        this->batch(), this->channels(), this->height(), this->width());
    PX_CHECK_CUDNN(status);

    status = cudnnDropoutGetReserveSpaceSize(*inputDesc_, &reserveSize_);
    PX_CHECK_CUDNN(status);

    reserves_ = V(reserveSize_);
}

template<>
inline void DropoutLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);

    this->output_.copy(input);

    if (!this->model().training()) {
        return;
    }

    const auto& ctxt = this->cudnnContext();
    auto status = cudnnDropoutForward(
            ctxt,
            *dropoutDesc_,
            *inputDesc_, input.data(),
            *outputDesc_, this->output_.data(),
            reserves_.data(), reserveSize_);

    PX_CHECK_CUDNN(status);
}

template<>
inline void DropoutLayer<Device::CUDA>::backward(const V& input, V* grad)
{
    Layer<Device::CUDA>::backward(input, grad);

    if (!this->model().training()) {
        return;
    }

    const auto& ctxt = this->cudnnContext();

    auto status = cudnnDropoutBackward(
            ctxt,
            *dropoutDesc_,
            *outputDesc_, this->delta_.data(),
            *inputDesc_, grad->data(),
            reserves_.data(), reserveSize_);

    PX_CHECK_CUDNN(status);
}

}   // px
