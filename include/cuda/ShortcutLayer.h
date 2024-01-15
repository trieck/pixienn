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

#include "ShortcutKernel.cuh"

namespace px {

template<>
inline void ShortcutLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);

    this->output_.copy(input);

    shortcutGpu(this->batch(), this->width(), this->height(), this->channels(), from_->output().data(),
                this->outWidth(), this->outHeight(), this->outChannels(), alpha_, beta_, this->output_.data());

    activation_->apply(this->output_);
}

}   // px
