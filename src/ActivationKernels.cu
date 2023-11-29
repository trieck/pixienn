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

#include "ActivationKernels.cuh"
#include "CudaUtils.cuh"
#include "CudaError.h"

namespace px {

struct LeakyActivation
{
    __device__ static float activate(float x)
    {
        return (x > 0) ? x : .1f * x;
    }
};

struct LinearActivation
{
    __device__ static float activate(float x)
    {
        return x;
    }
};

struct LogisticActivation
{
    __device__ static float activate(float x)
    {
        return 1.f / (1.f + expf(-x));
    }
};

struct LoggyActivation
{
    __device__ static float activate(float x)
    {
        return 2.f / (1.f + expf(-x)) - 1;
    }
};

struct ReluActivation
{
    __device__ static float activate(float x)
    {
        return x * float(x > 0);
    }
};

template<typename T>
__global__ void activateKernel(float* x, std::size_t n)
{
    std::size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = T::activate(x[i]);
    }
}

void leakyActivateGpu(float* x, std::size_t n)
{
    activateKernel<LeakyActivation><<<cuda_gridsize(n), CUDA_BLOCK_SIZE>>>(x, n);
    PX_CUDA_CHECK_LAST();
}

void linearActivateGpu(float* x, std::size_t n)
{
    activateKernel<LinearActivation><<<cuda_gridsize(n), CUDA_BLOCK_SIZE>>>(x, n);
    PX_CUDA_CHECK_LAST();
}

void loggyActivateGpu(float* x, std::size_t n)
{
    activateKernel<LoggyActivation><<<cuda_gridsize(n), CUDA_BLOCK_SIZE>>>(x, n);
    PX_CUDA_CHECK_LAST();
}

void logisticActivateGpu(float* x, std::size_t n)
{
    activateKernel<LogisticActivation><<<cuda_gridsize(n), CUDA_BLOCK_SIZE>>>(x, n);
    PX_CUDA_CHECK_LAST();
}

void reluActivateGpu(float* x, std::size_t n)
{
    activateKernel<ReluActivation><<<cuda_gridsize(n), CUDA_BLOCK_SIZE>>>(x, n);
    PX_CUDA_CHECK_LAST();
}

}   // px