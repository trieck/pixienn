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

#ifndef PIXIENN_UPSAMPLELAYER_H
#define PIXIENN_UPSAMPLELAYER_H

#include "UpsampleAlgo.h"
#include "Layer.h"

namespace px {

class UpsampleLayer : public Layer
{
protected:
    UpsampleLayer(Model& model, const YAML::Node& layerDef);

public:
    ~UpsampleLayer() override = default;

    std::ostream& print(std::ostream& os) override;
    void forward(const PxCpuVector& input) override;
    void backward(const PxCpuVector& input) override;

#ifdef USE_CUDA
    void forwardGpu(const PxCudaVector& input) override;
#endif

private:
    void setup() override;
    UpsampleContext makeContext(const PxCpuVector& input);

#ifdef USE_CUDA
    UpsampleContext makeContext(const PxCudaVector& input);
#endif
    friend LayerFactories;

    float scale_;
    int stride_;
};

} // px

#endif // PIXIENN_UPSAMPLELAYER_H
