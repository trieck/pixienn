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

#ifndef PIXIENN_BATCHNORM_ALGO_H
#define PIXIENN_BATCHNORM_ALGO_H

#include "PxTensor.h"

#ifdef USE_CUDA

#include "Cudnn.h"

#endif // USE_CUDA

namespace px {

// Represents the context needed for a batch norm operation
struct BNContext
{
    PxCpuTensor<1>* biasUpdates = nullptr;
    PxCpuTensor<1>* mean = nullptr;
    PxCpuTensor<1>* meanDelta = nullptr;
    PxCpuTensor<1>* rollingMean = nullptr;
    PxCpuTensor<1>* rollingVar = nullptr;
    PxCpuTensor<1>* scaleUpdates = nullptr;
    PxCpuTensor<1>* var = nullptr;
    PxCpuVector* delta = nullptr;
    PxCpuVector* output = nullptr;
    PxCpuVector* xNorm = nullptr;
    const PxCpuTensor<1>* biases = nullptr;
    const PxCpuTensor<1>* scales = nullptr;
    const PxCpuVector* input = nullptr;

#ifdef USE_CUDA
    PxCudaTensor<1>* biasUpdatesGpu = nullptr;
    PxCudaTensor<1>* meanGpu = nullptr;
    PxCudaTensor<1>* meanDeltaGpu = nullptr;
    PxCudaTensor<1>* scalesUpdatesGpu = nullptr;
    PxCudaVector* deltaGpu = nullptr;
    PxCudaVector* outputGpu = nullptr;
    const CudnnContext* cudnnContext = nullptr;
    const CudnnTensorDesc* dstTens = nullptr;
    const CudnnTensorDesc* normTens = nullptr;
    const PxCudaTensor<1>* biasesGpu = nullptr;
    const PxCudaTensor<1>* rollingMeanGpu = nullptr;
    const PxCudaTensor<1>* rollingVarGpu = nullptr;
    const PxCudaTensor<1>* scalesGpu = nullptr;
    const PxCudaVector* inputGpu = nullptr;
#endif // USE_CUDA

    int batch = 0;
    int channels = 0;
    int outHeight = 0;
    int outWidth = 0;
    bool training = false;
};

void batchNormForward(const BNContext& ctxt);
void batchNormBackward(const BNContext& ctxt);

#ifdef USE_CUDA
void batchNormForwardGpu(const BNContext& ctxt);
#endif // USE_CUDA

}   // px


#endif // PIXIENN_BATCHNORM_ALGO_H
