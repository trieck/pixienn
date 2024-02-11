/********************************************************************************
* Copyright 2023 Thomas A. Rieck, All Rights Reserved
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

#include <cuda_runtime.h>

#include "CudaError.h"
#include "CudaUtils.cuh"
#include "SoftmaxKernel.cuh"

namespace px {

__global__ void softmaxCrossEntropyKernel(int n, const float* pred, const float* truth, float* delta, float* error)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    constexpr auto epsilon = 1e-9f;  // Small epsilon for numerical stability

    for (int i = 0; i < n; ++i) {
        float t = truth[i];
        float p = pred[i];
        error[i] = t ? -log(p + epsilon) : 0;
        delta[i] = t - p;
    }
}

void softmaxCrossEntropy(int n, const float* pred, const float* truth, float* delta, float* error)
{
    softmaxCrossEntropyKernel<<<cudaGridsize(n), CUDA_BLOCK_SIZE>>>(n, pred, truth, delta, error);
    cudaDeviceSynchronize();

    PX_CUDA_CHECK_LAST();
}

}   // px
