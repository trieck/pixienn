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

#include "BiasKernels.cuh"
#include "CudaUtils.cuh"
#include "CudaError.h"
#include <cuda_runtime.h>

namespace px {

__global__ void addBiasKernel(float* output, const float* biases, int batch, int n, int size)
{
    auto index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index < n * size * batch) {
        auto i = index % size;
        index /= size;
        auto j = index % n;
        index /= n;
        auto k = index;

        output[(k * n + j) * size + i] += biases[j];
    }
}

void addBiasGpu(float* output, float* biases, int batch, int n, int size)
{
    auto num = n * size * batch;

    addBiasKernel<<<cuda_gridsize(num), CUDA_BLOCK_SIZE>>>(output, biases, batch, n, size);

    PX_CUDA_CHECK_LAST();
}

}   // px
