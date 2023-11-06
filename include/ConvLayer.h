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

#ifndef PIXIENN_CONVLAYER_H
#define PIXIENN_CONVLAYER_H

#include <xtensor/xtensor.hpp>

#include "Activation.h"
#include "Layer.h"

#ifdef USE_CUDA

#include "Cudnn.h"

#endif

namespace px {

class ConvLayer : public Layer
{
protected:
    ConvLayer(const Model& model, const YAML::Node& layerDef);

public:
    ~ConvLayer() override = default;

    std::ostream& print(std::ostream& os) override;
    std::streamoff loadDarknetWeights(std::istream& is) override;
    void forward(const xt::xarray<float>& input) override;

#ifdef USE_CUDA
    void forwardGpu(const PxDevVector<float>& input) override;
#endif

private:
    void setup() override;

#ifdef USE_CUDA
    void setup_gpu();
#endif
    friend LayerFactories;

    xt::xtensor<float, 4> weights_;
    xt::xtensor<float, 1> biases_;
    xt::xtensor<float, 2> column_;

    int dilation_ = 0, filters_, kernel_, padding_, stride_, groups_;
    std::string activation_;
    Layer::Ptr batchNormalize_;
    Activation::Ptr activationFnc_;

#ifdef USE_CUDA
    PxDevVector<float> weightsGpu_, biasesGpu_, workspace_;
    CudnnTensorDesc::Ptr xDesc_, yDesc_;
    CudnnConvDesc::Ptr convDesc_;
    CudnnFilterDesc::Ptr wDesc_;
    cudnnConvolutionFwdAlgo_t bestAlgo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
#endif
};

} // px

#endif // PIXIENN_CONVLAYER_H
