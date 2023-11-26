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

#include "PoolKernels.cuh"
#include "CudaUtils.cuh"
#include "CudaError.h"

namespace px {

__global__ void maxpoolKernel(int n, int in_h, int in_w, int in_c, int stride, int kernel, int pad, const float* input,
                              float* output)
{
    auto h = (in_h + pad - kernel) / stride + 1;
    auto w = (in_w + pad - kernel) / stride + 1;
    auto c = in_c;

    auto id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;

    auto j = id % w;
    id /= w;
    auto i = id % h;
    id /= h;
    auto k = id % c;
    id /= c;
    auto b = id;

    auto w_offset = -pad / 2;
    auto h_offset = -pad / 2;

    auto out_index = j + w * (i + h * (k + c * b));
    float max = -INFINITY;

    for (auto l = 0; l < kernel; ++l) {
        for (auto m = 0; m < kernel; ++m) {
            int cur_h = h_offset + i * stride + l;
            int cur_w = w_offset + j * stride + m;
            int index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                         cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max = (val > max) ? val : max;
        }
    }
    output[out_index] = max;
}

void maxpoolGpu(int n, int h, int w, int c, int stride, int kernel, int pad, const float* input, float* output)
{
    maxpoolKernel<<<cuda_gridsize(n), CUDA_BLOCK_SIZE>>>(n, h, w, c, stride, kernel, pad, input, output);

    PX_CUDA_CHECK_LAST();
}

}   // px