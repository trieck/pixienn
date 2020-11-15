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

#ifndef PIXIENN_BATCHNORMLAYER_H
#define PIXIENN_BATCHNORMLAYER_H

#include <xtensor/xtensor.hpp>

#include "Layer.h"

namespace px {

class BatchNormLayer : public Layer
{
protected:
    BatchNormLayer(const YAML::Node& layerDef);

public:
    virtual ~BatchNormLayer() = default;

    std::ostream& print(std::ostream& os) override;
    xt::xarray<float> forward(const xt::xarray<float>& input) override;
    void loadDarknetWeights(std::istream& is) override;

private:
    friend LayerFactories;
    xt::xtensor<float, 1> biases_, scales_, rollingMean_, rollingVar_;
    xt::xtensor<float, 4> output_;
};

} // px

#endif // PIXIENN_BATCHNORMLAYER_H
