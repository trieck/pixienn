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
#include "UpsampleKernels.cuh"

namespace px {

__global__ void
upsampleKernel(size_t N, const float* x, int w, int h, int c, int batch, int stride, int forward, float scale,
               float* acc, float* out)
{
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    auto outIndex = i;
    auto outW = i % (w * stride);
    i = i / (w * stride);
    auto outH = i % (h * stride);
    i = i / (h * stride);
    auto outC = i % c;
    i = i / c;
    auto b = i % batch;

    auto inW = outW / stride;
    auto inH = outH / stride;
    auto inC = outC;

    auto inIndex = b * w * h * c + inC * w * h + inH * w + inW;

    if (forward) {
        out[outIndex] += scale * x[inIndex];
    } else {
        atomicAdd(acc + inIndex, scale * out[outIndex]);
    }
}

void upsampleGpu(const float* in, int w, int h, int c, int batch, int stride, int forward, float scale, float* acc,
                 float* out)
{
    size_t size = w * h * c * batch * stride * stride;

    upsampleKernel<<<cudaGridsize(size), CUDA_BLOCK_SIZE>>>(size, in, w, h, c, batch, stride, forward, scale, acc, out);
    cudaDeviceSynchronize();

    PX_CUDA_CHECK_LAST();
}

}