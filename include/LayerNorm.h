/********************************************************************************
* Copyright 2026 Thomas A. Rieck, All Rights Reserved
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

#include "Layer.h"

namespace px {

/**
 * Layer normalization over the channel dimension of each spatial location.
 *
 * PixieNN tensors are stored as [batch, channels, height, width]. For every
 * (batch, y, x) location, this layer normalizes the channels independently of
 * every other spatial location. CPU and CUDA implementations share the same
 * parameter layout.
 */
template<Device D = Device::CPU>
class LayerNorm : public Layer<D>
{
public:
    using V = Layer<D>::V;

    LayerNorm(Model<D>& model, YAML::Node layerDef);

    void forward(const V& input) override;
    void backward(const V& input, V* grad) override;
    void update() override;

    void copyScales(const V& scales);
    void copyBiases(const V& biases);

    std::streamoff loadWeights(std::istream& is) override;
    std::streamoff saveWeights(std::ostream& os) override;

    std::ostream& print(std::ostream& os) override;

private:
    void scaleGradients() override;
    void clipGradients() override;

    float epsilon_ = 1e-5f;
    V biases_, biasUpdates_, scales_, scaleUpdates_;
    V mean_, variance_, normalized_;
};

using CpuLayerNorm = LayerNorm<>;

} // namespace px

#ifdef USE_CUDA
#include "cuda/LayerNorm.h"
#endif
