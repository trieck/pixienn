/********************************************************************************
* Copyright 2023 trieck, All Rights Reserved
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

#ifndef PIXIENN_SHORTCUTLAYER_H
#define PIXIENN_SHORTCUTLAYER_H

#include "Activation.h"
#include "Layer.h"

#include "ShortcutAlgo.h"

namespace px {

class ShortcutLayer : public Layer
{
protected:
    ShortcutLayer(Model& model, const YAML::Node& layerDef);

public:
    ~ShortcutLayer() override = default;

    std::ostream& print(std::ostream& os) override;
    void forward(const PxCpuVector& input) override;
    void backward(const PxCpuVector& input) override;

#ifdef USE_CUDA
    void forwardGpu(const PxCudaVector& input) override;
#endif

private:
    void setup() override;
    ShortcutContext makeContext(const PxCpuVector&);

#ifdef USE_CUDA
    ShortcutContext makeContext(const PxCudaVector&);
#endif
    friend LayerFactories;

    Activation::Ptr activationFnc_;
    Layer::Ptr from_;
    float alpha_, beta_;
};

}   // px

#endif // PIXIENN_SHORTCUTLAYER_H
