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

#include "Common.h"
#include "UpsampleAlgo.h"

#ifdef USE_CUDA

#include "UpsampleKernels.cuh"

#endif

namespace px {

static void upsampleCommon(const float* in, int w, int h, int c, int batch, int stride, float scale, float* out,
                           float* acc, bool forward)
{
    for (auto b = 0; b < batch; ++b) {
        for (auto k = 0; k < c; ++k) {
            for (auto j = 0; j < h * stride; ++j) {
                for (auto i = 0; i < w * stride; ++i) {
                    auto inIndex = b * w * h * c + k * w * h + (j / stride) * w + i / stride;
                    auto outIndex = b * w * h * c * stride * stride + k * w * h * stride * stride + j * w * stride + i;

                    if (forward) {
                        out[outIndex] = scale * in[inIndex];
                    } else {
                        acc[inIndex] += scale * out[outIndex];
                    }
                }
            }
        }
    }
}

void upsampleForward(const UpsampleContext& ctxt)
{
    upsampleCommon(ctxt.input->data(), ctxt.width, ctxt.height, ctxt.channels, ctxt.batch, ctxt.stride,
                   ctxt.scale, ctxt.output->data(), nullptr, true);
}

void upsampleBackward(const UpsampleContext& ctxt)
{
    upsampleCommon(nullptr, ctxt.width, ctxt.height, ctxt.channels, ctxt.batch, ctxt.stride,
                   ctxt.scale, ctxt.delta->data(), ctxt.netDelta->data(), false);

}

#ifdef USE_CUDA

void upsampleForwardGpu(const UpsampleContext& ctxt)
{
    upsampleGpu(ctxt.inputGpu->data(), ctxt.width, ctxt.height, ctxt.channels, ctxt.batch, ctxt.stride, 1, ctxt.scale,
                ctxt.outputGpu->data());
}

#endif // USE_CUDA

}   // px
