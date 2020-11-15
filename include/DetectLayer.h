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

#ifndef PIXIENN_DETECTLAYER_H
#define PIXIENN_DETECTLAYER_H


#include "Layer.h"

namespace px {

class DetectLayer : public Layer
{
protected:
    DetectLayer(const YAML::Node& layerDef);

public:
    virtual ~DetectLayer() = default;

    std::ostream& print(std::ostream& os) override;
    xt::xarray<float> forward(const xt::xarray<float>& input) override;

private:
    friend LayerFactories;

    int coords_, classes_ , num_, side_, maxBoxes_;
    bool rescore_, softmax_, sqrt_, forced_, random_, reorg_;
    float coordScale_, objectScale_, noObjectScale_, classScale, jitter_;
};

} // px

#endif // PIXIENN_DETECTLAYER_H
