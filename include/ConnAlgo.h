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

#ifndef PIXIENN_CONN_ALGO_H
#define PIXIENN_CONN_ALGO_H

#include "PxTensor.h"

#ifdef USE_CUDA

#include "Cublas.h"

#endif // USE_CUDA

namespace px {

// Represents the context needed for a connected operation
struct ConnContext
{
    const PxCpuVector* input = nullptr;
    PxCpuVector* output = nullptr;
    const PxCpuVector* delta = nullptr;
    PxCpuVector* netDelta = nullptr;
    const PxCpuTensor<2>* weights = nullptr;
    PxCpuTensor<2>* weightUpdates = nullptr;

#ifdef USE_CUDA
    const PxCudaVector* inputGpu = nullptr;
    const PxCudaTensor<2>* weightsGpu = nullptr;
    PxCudaVector* outputGpu = nullptr;
    const CublasContext* cublasContext = nullptr;
#endif // USE_CUDA

    int batch = 0;
    int inputs = 0;
    int outputs = 0;
};

void connectedForward(const ConnContext& ctxt);
void connectedBackward(const ConnContext& ctxt);

#ifdef USE_CUDA

void connectedForwardGpu(const ConnContext& ctxt);

#endif

}   // px

#endif // PIXIENN_CONN_ALGO_H
