/********************************************************************************
* Copyright 2023 trieck, All Rights Reserved
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

#ifndef PIXIENN_SHORTCUTALGO_H
#define PIXIENN_SHORTCUTALGO_H

#include "PxTensor.h"

#ifdef USE_CUDA

#include "Cublas.h"

#endif // USE_CUDA

namespace px {

// Represents the context needed for a shortcut operation
struct ShortcutContext
{
    const PxCpuVector* from = nullptr;
    PxCpuVector* output = nullptr;
    const PxCpuVector* delta = nullptr;
    PxCpuVector* fromDelta = nullptr;

#ifdef USE_CUDA
    const PxCudaVector* addGpu = nullptr;
    PxCudaVector* outputGpu = nullptr;

#endif // USE_CUDA

    int batch = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    int outWidth = 0;
    int outHeight = 0;
    int outChannels = 0;
    float alpha = 0.0f;
    float beta = 0.0f;
};

void shortcutForward(const ShortcutContext& ctxt);
void shortcutBackward(const ShortcutContext& ctxt);

#ifdef USE_CUDA

void shortcutForwardGpu(const ShortcutContext& ctxt);

#endif

}   // px

#endif // PIXIENN_SHORTCUTALGO_H