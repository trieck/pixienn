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

#include "MaxPoolKernels.cuh"

namespace px {

template<>
class MPExtras<Device::CUDA>
{
protected:
    PxCudaVectorT<int> indexes_;
};

template<>
inline void MaxPoolLayer<Device::CUDA>::setup()
{
    indexes_ = PxCudaVectorT<int>(this->batch() * this->outputs(), -1);
}

template<>
inline void MaxPoolLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);
    maxPoolForwardGpu(input.data(), this->output_.data(), indexes_.data(), this->batch(), this->channels(),
                      this->height(), this->width(), this->outHeight(), this->outWidth(), kernel_, stride_, padding_);
}

template<>
inline void MaxPoolLayer<Device::CUDA>::backward(const V& input, V* grad)
{
    Layer<Device::CUDA>::backward(input, grad);

    if (grad == nullptr) {
        return;
    }

    maxPoolBackwardGpu(delta_.data(), indexes_.data(), grad->data(), this->batch() * this->outputs());
}

}   // px
