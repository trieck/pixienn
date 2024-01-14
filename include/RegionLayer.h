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

#include "Layer.h"

namespace px {

template<Device D = Device::CPU>
class RegionLayer : public Layer<D>
{
public:
    using V = typename Layer<D>::V;

    RegionLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;
    void update() override;

    std::ostream& print(std::ostream& os) override;
};

template<Device D>
RegionLayer<D>::RegionLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
}

template<Device D>
std::ostream& RegionLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "region", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

template<Device D>
void RegionLayer<D>::forward(const V& input)
{
    std::cout << "RegionLayer::forward" << std::endl;
}

template<Device D>
void RegionLayer<D>::backward(const V& input)
{
    std::cout << "RegionLayer::backward" << std::endl;
}

template<Device D>
void RegionLayer<D>::update()
{

}

using CpuRegion = RegionLayer<>;
using CudaRegion = RegionLayer<Device::CUDA>;

} // px
