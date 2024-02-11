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
#include <thrust/device_ptr.h>

#include "Adam.cuh"
#include "CudaUtils.cuh"
#include "CudaError.h"

namespace px {

__global__ void adamKernel(int n, float* x, float* m, float* v, float beta1, float beta2, float rate, float eps, int t)
{
    auto index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    auto mhat = m[index] / (1.0f - powf(beta1, t));
    auto vhat = v[index] / (1.0f - powf(beta2, t));

    x[index] += rate * mhat / (sqrtf(vhat) + eps);
}

void adamGpu(int n, float* x, float* m, float* v, float beta1, float beta2, float rate, float eps, int t)
{
    adamKernel<<<cudaGridsize(n), CUDA_BLOCK_SIZE>>>(n, x, m, v, beta1, beta2, rate, eps, t);

    cudaDeviceSynchronize();

    PX_CUDA_CHECK_LAST();
}

}   // px
