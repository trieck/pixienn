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

namespace px {

template<>
inline void Model<Device::CUDA>::setup()
{
    this->cublasCtxt_ = std::make_unique<CublasContext>();
    this->cudnnCtxt_ = std::make_unique<CudnnContext>();
}

template<>
inline void Model<Device::CUDA>::forward(const ImageVec& image)
{
    V input(&(*image.data.begin()), &(*image.data.end()));

    forward(input);
}

template<>
inline float Model<Device::CUDA>::trainBatch()
{
    trainBatch_ = trainLoader_->next();

    const auto& imageData = trainBatch_.imageData();

    V input(&(*imageData.begin()), &(*imageData.end()));

    auto error = trainOnce(input);

    return error;
}

}   // px
