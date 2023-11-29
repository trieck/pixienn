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

#include "ShortcutKernel.cuh"
#include "CudaUtils.cuh"
#include "CudaError.h"

namespace px {

__global__ void
shortcutKernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1,
               const float* add, int w2, int h2, int c2, float s1, float s2, float* out)
{
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
    int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
    out[out_index] = s1 * out[out_index] + s2 * add[add_index];
}

void shortcutGpu(int batch, int w1, int h1, int c1, const float* add, int w2, int h2, int c2, float s1, float s2,
                 float* out)
{
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1 / w2;
    int sample = w2 / w1;

    if (stride < 1) stride = 1;
    if (sample < 1) sample = 1;

    int size = batch * minw * minh * minc;

    shortcutKernel<<<cuda_gridsize(size), CUDA_BLOCK_SIZE>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1,
                                                             add, w2,
                                                             h2, c2, s1, s2, out);

    PX_CUDA_CHECK_LAST();

}

}