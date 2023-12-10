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

#ifndef PIXIENN_MAXPOOL_ALGO_H
#define PIXIENN_MAXPOOL_ALGO_H

#include "PxTensor.h"

#ifdef USE_CUDA

#include "Cudnn.h"

#endif // USE_CUDA

namespace px {

// Represents the context needed for a batch norm operation
struct MaxPoolContext
{
    const PxCpuVector* input = nullptr;
    PxCpuVector* output = nullptr;
    PxCpuVectorT<int>* indexes = nullptr;
    const PxCpuVector* delta = nullptr;
    PxCpuVector* netDelta = nullptr;

#ifdef USE_CUDA
    const PxCudaVector* inputGpu = nullptr;
    PxCudaVector* outputGpu = nullptr;
    PxCudaVectorT<int>* indexesGpu = nullptr;
#endif // USE_CUDA

    int batch = 0;
    int channels = 0;
    int height = 0;
    int width = 0;
    int outHeight = 0;
    int outWidth = 0;
    int kernel = 0;
    int stride = 0;
    int padding = 0;
};

void maxPoolForward(const MaxPoolContext& ctxt);
void maxPoolBackward(const MaxPoolContext& ctxt);

#ifdef USE_CUDA
void maxPoolForwardGpu(const MaxPoolContext& ctxt);
#endif // USE_CUDA

}   // px


#endif // PIXIENN_MAXPOOL_ALGO_H
