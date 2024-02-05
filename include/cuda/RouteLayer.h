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
inline void RouteLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);

    auto offset = 0;

    auto* output = this->output_.data();

    for (const auto& layer: layers_) {
        const auto* pin = layer->output().data();
        auto inputSize = layer->outputs();

        for (auto i = 0; i < batch(); ++i) {
            const auto* start = pin + i * inputSize;
            auto* out = output + offset + i * outputs();

            auto result = cudaMemcpy(out, start, inputSize * sizeof(float), cudaMemcpyDeviceToDevice);
            PX_CUDA_CHECK_ERR(result);
        }

        offset += inputSize;
    }
}

template<>
inline void RouteLayer<Device::CUDA>::backward(const V& input, V* grad)
{
    Layer<Device::CUDA>::backward(input, grad);

    auto alpha = 1.0f;
    auto offset = 0;
    auto* pdelta = this->delta_.data();

    const auto& ctxt = this->cublasContext();

    for (const auto& layer: layers_) {
        auto* ldelta = layer->delta().data();
        auto outputSize = layer->outputs();

        for (auto i = 0; i < this->batch(); ++i) {
            auto* in = pdelta + offset + i * this->outputs();
            auto* out = ldelta + i * outputSize;

            auto status = cublasSaxpy(ctxt, outputSize, &alpha, in, 1, out, 1);
            PX_CHECK_CUBLAS(status);
        }

        offset += outputSize;
    }
}

}   // px
