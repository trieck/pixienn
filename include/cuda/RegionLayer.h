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
inline void RegionLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);

    PxCpuVector cpuInput(input.size());
    cpuInput.copyDevice(input.data(), input.size());

    PxCpuVector cpuOutput(this->output_.size());

    forwardCpu(cpuInput, cpuOutput);

    this->output_.copyHost(cpuOutput.data(), cpuOutput.size());
}

template<>
inline void RegionLayer<Device::CUDA>::addDetects(Detections& detects, int width, int height, float threshold)
{
    auto preds = this->output_.asVector();

    addDetects(detects, width, height, threshold, preds.data());
}

}   // px
