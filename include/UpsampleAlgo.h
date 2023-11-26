/********************************************************************************
* Copyright 2020-2023 Thomas A. Rieck, All Rights Reserved
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

#ifndef PIXIENN_UPSAMPLE_ALGO_H
#define PIXIENN_UPSAMPLE_ALGO_H

#include "PxTensor.h"

namespace px {

void upsampleGpu(const float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

// Represents the context needed for an upsample operation
struct UpsampleContext
{
    const PxCpuVector* input = nullptr;
    PxCpuVector* output = nullptr;

#ifdef USE_CUDA
    const PxCudaVector* inputGpu = nullptr;
    PxCudaVector* outputGpu = nullptr;
#endif // USE_CUDA

    bool forward = true;
    float scale = 0.0f;
    int batch = 0;
    int channels = 0;
    int flags = 0;
    int height = 0;
    int outChannels = 0;
    int outHeight = 0;
    int outWidth = 0;
    int stride = 0;
    int width = 0;
};

void upsampleForward(const UpsampleContext& ctxt);

#ifdef USE_CUDA
void upsampleForwardGpu(const UpsampleContext& ctxt);
#endif // USE_CUDA

}   // px

#endif // PIXIENN_UPSAMPLE_ALGO_H
