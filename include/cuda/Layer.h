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
#include "Cublas.h"

namespace px {

template<>
inline void Layer<Device::CUDA>::scaleGradients()
{
    const auto& ctxt = cublasContext();

    auto norm = 0.0f;
    auto status = cublasSnrm2(ctxt, delta_.size(), delta_.data(), 1, &norm);
    PX_CHECK_CUBLAS(status);

    if (norm > 0 && norm > gradientThreshold_) {
        auto scale = gradientThreshold_ / norm;
        status = cublasSscal(ctxt, delta_.size(), &scale, delta_.data(), 1);

        PX_CHECK_CUBLAS(status);
    }
}

template<>
inline void Layer<Device::CUDA>::clipGradients()
{
    constrainGpu(delta_.size(), gradientClipValue_, delta_.data());
}

}   // px
