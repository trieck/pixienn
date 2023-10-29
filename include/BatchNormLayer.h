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

#include "Layer.h"
#include <xtensor/xtensor.hpp>

#ifdef USE_CUDA

#include "Cudnn.h"

#endif

namespace px {

class BatchNormLayer : public Layer
{
protected:
    BatchNormLayer(const Model& model, const YAML::Node& layerDef);

public:
    virtual void forward(const xt::xarray<float>& input) override;

#ifdef USE_CUDA
    virtual void forwardGpu(const PxDevVector<float>& input) override;
#endif

    std::ostream& print(std::ostream& os) override;
    std::streamoff loadDarknetWeights(std::istream& is) override;

private:
    friend LayerFactories;

    xt::xtensor<float, 1> biases_, scales_, rollingMean_, rollingVar_;

#ifdef USE_CUDA
    PxDevVector<float> xGpu_, biasesGpu_, scalesGpu_, rollingMeanGpu_, rollingVarGpu_;
    CudnnTensorDesc normTens_, dstTens_;
#endif
};

} // px

#endif // PIXIENN_BATCHNORMLAYER_H
