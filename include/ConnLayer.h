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

#ifndef PIXIENN_CONNLAYER_H
#define PIXIENN_CONNLAYER_H

#include <xtensor/xtensor.hpp>

#include "Activation.h"
#include "Layer.h"

namespace px {

class ConnLayer : public Layer
{
protected:
    ConnLayer(const Model& model, const YAML::Node& layerDef);

public:
    virtual ~ConnLayer() = default;

    std::ostream& print(std::ostream& os) override;
    std::streamoff loadDarknetWeights(std::istream& is) override;
    void forward(const xt::xarray<float>& input) override;

private:
    friend LayerFactories;

    xt::xtensor<float, 2> weights_;
    xt::xtensor<float, 1> biases_;

    Activation::Ptr activationFnc_;

    Layer::Ptr batchNormalize_;
    std::string activation_;
    float scales_, rollingMean_, rollingVar_;
};

} // px

#endif // PIXIENN_CONNLAYER_H
