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

#include "BiasKernels.cuh"
#include "CudaUtils.cuh"
#include "CudaError.h"

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

__global__ void backwardBiasConnKernel(float* biasUpdates, const float* delta, int batch, int n)
{
    auto index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    auto sum = 0.0f;
    for (auto b = 0; b < batch; ++b) {
        auto i = b * n + index;
        sum += delta[i];
    }

    biasUpdates[index] += sum;
}

__global__ void backwardBiasKernel(float* biasUpdates, const float* delta, int batch, int n, int size)
{
    __shared__ float part[CUDA_BLOCK_SIZE];

    auto filter = blockIdx.x;
    auto p = threadIdx.x;

    auto sum = 0.0f;

    for (auto b = 0; b < batch; ++b) {
        for (auto i = 0; i < size; i += CUDA_BLOCK_SIZE) {
            auto index = p + i + size * (filter + n * b);
            sum += (p + i < size) ? delta[index] : 0;
        }
    }

    part[p] = sum;

    __syncthreads();

    if (p == 0) {
        for (auto i = 0; i < CUDA_BLOCK_SIZE; ++i) {
            biasUpdates[filter] += part[i];
        }
    }
}

void addBiasGpu(float* output, const float* biases, int batch, int n, int size)
{
    auto num = n * size * batch;

    addBiasKernel<<<cudaGridsize(num), CUDA_BLOCK_SIZE>>>(output, biases, batch, n, size);

    PX_CUDA_CHECK_LAST();
}

void backwardBiasGpu(float* biasUpdates, const float* delta, int batch, int n, int size)
{
    if (size == 1) {
        backwardBiasConnKernel<<<cudaGridsize(n), CUDA_BLOCK_SIZE>>>(biasUpdates, delta, batch, n);
    } else {
        backwardBiasKernel<<<cudaGridsize(n), CUDA_BLOCK_SIZE>>>(biasUpdates, delta, batch, n, size);
    }

    cudaDeviceSynchronize();

    PX_CUDA_CHECK_LAST();
}

}   // px
