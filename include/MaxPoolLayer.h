/********************************************************************************
* Copyright 2020 Thomas A. Rieck, All Rights Reserved
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

#ifndef PIXIENN_MAXPOOLLAYER_H
#define PIXIENN_MAXPOOLLAYER_H

#include <xtensor/xtensor.hpp>
#include "Layer.h"

namespace px {

class MaxPoolLayer : public Layer
{
protected:
    MaxPoolLayer(const Model& model, const YAML::Node& layerDef);

public:
    ~MaxPoolLayer() override = default;

    std::ostream& print(std::ostream& os) override;
    void forward(const xt::xarray<float>& input) override;

#ifdef USE_CUDA
    void forwardGpu(const PxDevVector<float>& input) override;
#endif

private:
    void setup() override;

    friend LayerFactories;
    int kernel_ = 0, stride_ = 0, padding_;
};

} // px

#endif // PIXIENN_MAXPOOLLAYER_H
