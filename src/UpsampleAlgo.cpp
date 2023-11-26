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

void upsampleForward(const UpsampleContext& ctxt)
{
    auto inputs = ctxt.channels * ctxt.height * ctxt.width;
    auto outputs = ctxt.outChannels * ctxt.outHeight * ctxt.outWidth;

    for (auto b = 0; b < ctxt.batch; ++b) {
        auto* pinput = ctxt.input->data() + b * inputs;
        auto* poutput = ctxt.output->data() + b * outputs;

        for (int yIn = 0; yIn < ctxt.height; ++yIn) {
            for (int xIn = 0; xIn < ctxt.width; ++xIn) {
                float value = pinput[yIn * ctxt.width + xIn];
                for (int yStride = 0; yStride < ctxt.stride; ++yStride) {
                    for (int xStride = 0; xStride < ctxt.stride; ++xStride) {
                        int yOut = std::min(yIn * ctxt.stride + yStride, ctxt.outHeight - 1);
                        int xOut = std::min(xIn * ctxt.stride + xStride, ctxt.outWidth - 1);
                        poutput[yOut * ctxt.outWidth + xOut] = ctxt.scale * value;
                    }
                }
            }
        }
    }
}

#ifdef USE_CUDA

void upsampleForwardGpu(const UpsampleContext& ctxt)
{
    upsampleGpu(ctxt.inputGpu->data(), ctxt.width, ctxt.height, ctxt.channels, ctxt.batch, ctxt.stride, 1, ctxt.scale,
                ctxt.outputGpu->data());
}

#endif // USE_CUDA

}   // px
