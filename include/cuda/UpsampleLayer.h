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

#include "UpsampleKernels.cuh"

namespace px {

template<>
class UpsampleExtras<Device::CUDA>
{
};

template<>
inline void UpsampleLayer<Device::CUDA>::setup()
{
}

template<>
inline void UpsampleLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);

    upsampleGpu(input.data(), this->width(), this->height(), this->channels(), this->batch(), stride_,
                1, scale_, nullptr, this->output_.data());
}

template<>
inline void UpsampleLayer<Device::CUDA>::backward(const px::UpsampleLayer<px::Device::CUDA>::V& input)
{
    Layer<Device::CUDA>::backward(input);

    upsampleGpu(nullptr, this->width(), this->height(), this->channels(), this->batch(), stride_,
                1, scale_, this->netDelta()->data(), this->delta_.data());
}

}   // px
