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

#include "UpsampleKernels.cuh"
#include "CudaUtils.cuh"
#include "CudaError.h"
#include <cuda_runtime.h>

namespace px {

__global__ void
upsampleKernel(size_t N, const float* x, int w, int h, int c, int batch, int stride, int forward, float scale,
               float* out)
{
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    auto out_index = i;
    auto out_w = i % (w * stride);
    i = i / (w * stride);
    auto out_h = i % (h * stride);
    i = i / (h * stride);
    auto out_c = i % c;
    i = i / c;
    auto b = i % batch;

    auto in_w = out_w / stride;
    auto in_h = out_h / stride;
    auto in_c = out_c;

    auto in_index = b * w * h * c + in_c * w * h + in_h * w + in_w;

    if (forward) {
        out[out_index] += scale * x[in_index];
    } else {
        // FIXME: THIS IS BAD SINCE IT ATTEMPTS TO UPDATE A CONST*
        // FIND ANOTHER WAY
        // atomicAdd(x + in_index), scale * out[out_index]);
    }
}

void upsampleGpu(const float* in, int w, int h, int c, int batch, int stride, int forward, float scale, float* out)
{
    size_t size = w * h * c * batch * stride * stride;

    upsampleKernel<<<cudaGridsize(size), CUDA_BLOCK_SIZE>>>(size, in, w, h, c, batch, stride, forward, scale, out);
    cudaDeviceSynchronize();

    PX_CUDA_CHECK_LAST();
}

}