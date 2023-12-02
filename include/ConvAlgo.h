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

#ifndef PIXIENN_CONVALGO_H
#define PIXIENN_CONVALGO_H

#include "PxTensor.h"

#ifdef USE_CUDA

#include "Cudnn.h"

#endif

namespace px {

// Represents the context needed for a convolutional operation
struct ConvContext
{
    const PxCpuVector* input = nullptr;
    const PxCpuTensor<4>* weights = nullptr;
    PxCpuTensor<2>* column = nullptr;
    PxCpuVector* output = nullptr;

#ifdef USE_CUDA
    const PxCudaVector* inputGpu = nullptr;
    const PxCudaTensor<4>* weightsGpu = nullptr;
    const CudnnContext* cudnnContext = nullptr;
    const CudnnTensorDesc* xDesc = nullptr;
    const CudnnTensorDesc* yDesc = nullptr;
    const CudnnConvDesc* convDesc = nullptr;
    const CudnnFilterDesc* wDesc = nullptr;
    PxCudaTensor<1>* workspace = nullptr;
    PxCudaVector* outputGpu = nullptr;
    cudnnConvolutionFwdAlgo_t bestAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
#endif

    int batch = 0;
    int channels = 0;
    int dilation = 1;
    int filters = 1;
    int groups = 1;
    int height = 0;
    int kernel = 0;
    int outHeight = 0;
    int outWidth = 0;
    int padding = 0;
    int stride = 1;
    int width = 0;
};

void convolutionalForward(const ConvContext& ctxt);
void convolutionalBackward(const ConvContext& ctxt);

#ifdef USE_CUDA
void convolutionalForwardGpu(const ConvContext& ctxt);
#endif // USE_CUDA

}   // px

#endif // PIXIENN_CONVALGO_H
