/********************************************************************************
* Copyright 2023 Maxar Technologies Inc.
* Author: Thomas A. Rieck
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
*
* SBIR DATA RIGHTS
* Contract No. HM0476-16-C-0022
* Contractor Name: Radiant Analytic Solutions Inc.
* Contractor Address: 2325 Dulles Corner Blvd. STE 1000, Herndon VA 20171
* Expiration of SBIR Data Rights Period: 2/13/2029
*
* The Government's rights to use, modify, reproduce, release, perform, display,
* or disclose technical data or computer software marked with this legend are
* restricted during the period shown as provided in paragraph (b)(4) of the
* Rights in Noncommercial Technical Data and Computer Software-Small Business
* Innovation Research (SBIR) Program clause contained in the above identified
* contract. No restrictions apply after the expiration date shown above. Any
* reproduction of technical data, computer software, or portions thereof marked
* with this legend must also reproduce the markings.
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
        return x * (x > 0);
    }
};

template<typename T>
__global__ void activate_kernel(float* x, int n)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = T::activate(x[i]);
    }
}

void leaky_activate_gpu(float* x, std::size_t n)
{
    activate_kernel<LeakyActivation><<<cuda_gridsize(n), CUDA_BLOCK_SIZE>>>(x, n);
    PX_CUDA_CHECK_LAST();
}

void linear_activate_gpu(float* x, std::size_t n)
{
    activate_kernel<LinearActivation><<<cuda_gridsize(n), CUDA_BLOCK_SIZE>>>(x, n);
    PX_CUDA_CHECK_LAST();
}

void loggy_activate_gpu(float* x, std::size_t n)
{
    activate_kernel<LoggyActivation><<<cuda_gridsize(n), CUDA_BLOCK_SIZE>>>(x, n);
    PX_CUDA_CHECK_LAST();
}

void logistic_activate_gpu(float* x, std::size_t n)
{
    activate_kernel<LogisticActivation><<<cuda_gridsize(n), CUDA_BLOCK_SIZE>>>(x, n);
    PX_CUDA_CHECK_LAST();
}

void relu_activate_gpu(float* x, std::size_t n)
{
    activate_kernel<ReluActivation><<<cuda_gridsize(n), CUDA_BLOCK_SIZE>>>(x, n);
    PX_CUDA_CHECK_LAST();
}

}   // px