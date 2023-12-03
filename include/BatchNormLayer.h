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

#ifndef PIXIENN_BATCHNORMLAYER_H
#define PIXIENN_BATCHNORMLAYER_H

#include "Layer.h"
#include "BatchNormAlgo.h"

#ifdef USE_CUDA

#include "Cudnn.h"

#endif

namespace px {

class BatchNormLayer : public Layer
{
protected:
    BatchNormLayer(Model& model, const YAML::Node& layerDef);

public:
    void forward(const PxCpuVector& input) override;
    void backward(const PxCpuVector& input) override;

#ifdef USE_CUDA
    void forwardGpu(const PxCudaVector& input) override;
#endif

    std::ostream& print(std::ostream& os) override;
    std::streamoff loadWeights(std::istream& is) override;

private:
    void setup() override;
    BNContext makeContext(const PxCpuVector& input);

    friend LayerFactories;
    PxCpuTensor<1> biases_, scales_, rollingMean_, rollingVar_;

#ifdef USE_CUDA
    void setupGpu();
    BNContext makeContext(const PxCudaVector& input);

    PxCudaTensor<1> biasesGpu_, scalesGpu_, rollingMeanGpu_, rollingVarGpu_;
    CudnnTensorDesc::Ptr normTens_, dstTens_;
#endif
};

} // px

#endif // PIXIENN_BATCHNORMLAYER_H
