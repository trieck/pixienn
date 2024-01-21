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
class RegionExtras<Device::CUDA>
{
protected:
    PxCpuVector cpuOutput_;
    PxCpuVector cpuDelta_;
};

template<>
inline void RegionLayer<Device::CUDA>::setup()
{
    cpuOutput_ = PxCpuVector(this->output_.size());
    cpuDelta_ = PxCpuVector(this->delta_.size());

    poutput_ = &cpuOutput_;
    pdelta_ = &cpuDelta_;
};

template<>
inline void RegionLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);

    PxCpuVector cpuInput(input.size());
    cpuInput.copyDevice(input.data(), input.size());

    cpuOutput_.copyDevice(this->output_.data(), this->output_.size());
    cpuDelta_.copyDevice(this->delta_.data(), this->delta_.size());

    forwardCpu(cpuInput);

    this->output_.copyHost(cpuOutput_.data(), cpuOutput_.size());
    this->delta_.copyHost(cpuDelta_.data(), cpuDelta_.size());
}

template<>
inline void RegionLayer<Device::CUDA>::backward(const V& input)
{
    Layer<Device::CUDA>::backward(input);

    // FIXME: flatten on GPU

    auto alpha = 1.0f;

    const auto& ctxt = this->cublasContext();

    auto status = cublasSaxpy(ctxt, this->delta_.size(), &alpha, this->delta_.data(), 1, this->netDelta()->data(), 1);

    PX_CHECK_CUBLAS(status);
}

template<>
inline void RegionLayer<Device::CUDA>::addDetects(Detections& detects, int width, int height, float threshold)
{
    auto preds = this->output_.asVector();

    addDetects(detects, width, height, threshold, preds.data());
}

}   // px
