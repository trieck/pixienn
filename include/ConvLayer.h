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

#ifndef PIXIENN_CONVLAYER_H
#define PIXIENN_CONVLAYER_H

#include "Activation.h"
#include "BatchNormAlgo.h"
#include "ConvAlgo.h"
#include "Layer.h"

#ifdef USE_CUDA

#include "Cudnn.h"

#endif

namespace px {

class ConvLayer : public Layer
{
protected:
    ConvLayer(Model& model, const YAML::Node& layerDef);

public:
    ~ConvLayer() override = default;

    std::ostream& print(std::ostream& os) override;
    std::streamoff loadWeights(std::istream& is) override;
    std::streamoff saveWeights(std::ostream& os) override;
    void forward(const PxCpuVector& input) override;
    void backward(const PxCpuVector& input) override;
    void update() override;

#ifdef USE_CUDA
    void forwardGpu(const PxCudaVector& input) override;
#endif

private:
    void setup() override;
    ConvContext makeContext(const PxCpuVector& input);
    BNContext makeBNContext(const PxCpuVector& input);

#ifdef USE_CUDA
    void setup_gpu();
    ConvContext makeContext(const PxCudaVector& input);
#endif
    friend LayerFactories;

    PxCpuTensor<4> weights_, weightUpdates_;
    PxCpuTensor<1> biases_, biasUpdates_, scales_, scaleUpdates_, rollingMean_, rollingVar_;
    PxCpuTensor<1> mean_, meanDelta_, var_, varDelta_;
    PxCpuTensor<2> column_;
    PxCpuVector x_, xNorm_;

    int dilation_ = 0, filters_, kernel_, padding_, stride_, groups_;
    bool batchNormalize_;
    Activations::Ptr activationFnc_;

#ifdef USE_CUDA
    PxCudaTensor<4> weightsGpu_;
    PxCudaTensor<1> biasesGpu_, workspace_;
    CudnnTensorDesc::Ptr xDesc_, yDesc_;
    CudnnConvDesc::Ptr convDesc_;
    CudnnFilterDesc::Ptr wDesc_;
    cudnnConvolutionFwdAlgo_t bestAlgo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
#endif
};

} // px

#endif // PIXIENN_CONVLAYER_H
