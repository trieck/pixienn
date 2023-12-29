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

#include "ShortcutAlgo.h"

#ifdef USE_CUDA

#include "ShortcutKernel.cuh"

#endif

namespace px {

static void shortcutCpu(int batch, int w1, int h1, int c1, const float* add, int w2, int h2, int c2, float s1,
                        float s2, float* out);

void shortcutForward(const ShortcutContext& ctxt)
{
    shortcutCpu(ctxt.batch, ctxt.width, ctxt.height, ctxt.channels, ctxt.from->data(), ctxt.outWidth, ctxt.outHeight,
                ctxt.outChannels, ctxt.alpha, ctxt.beta, ctxt.output->data());
}

void shortcutBackward(const ShortcutContext& ctxt)
{
    shortcutCpu(ctxt.batch, ctxt.outWidth, ctxt.outHeight, ctxt.outChannels, ctxt.delta->data(), ctxt.width,
                ctxt.height, ctxt.channels, ctxt.alpha, ctxt.beta, ctxt.fromDelta->data());
}

void shortcutCpu(int batch, int w1, int h1, int c1, const float* add, int w2, int h2, int c2, float s1, float s2,
                 float* out)
{
    auto stride = w1 / w2;
    auto sample = w2 / w1;

    PX_CHECK(stride == h1 / h2, "Stride mismatch.");
    PX_CHECK(sample == h2 / h1, "Sample mismatch.");

    if (stride < 1) {
        stride = 1;
    }

    if (sample < 1) {
        sample = 1;
    }

    auto minw = (w1 < w2) ? w1 : w2;
    auto minh = (h1 < h2) ? h1 : h2;
    auto minc = (c1 < c2) ? c1 : c2;

    for (auto b = 0; b < batch; ++b) {
        for (auto k = 0; k < minc; ++k) {
            for (auto j = 0; j < minh; ++j) {
                for (auto i = 0; i < minw; ++i) {
                    auto outIndex = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                    auto addIndex = i * stride + w1 * (j * stride + h1 * (k + c1 * b));

                    out[outIndex] = s1 * out[outIndex] + s2 * add[addIndex];
                }
            }
        }
    }
}

#ifdef USE_CUDA

void shortcutForwardGpu(const ShortcutContext& ctxt)
{
    shortcutGpu(ctxt.batch, ctxt.width, ctxt.height, ctxt.channels, ctxt.addGpu->data(), ctxt.outWidth, ctxt.outHeight,
                ctxt.outChannels, ctxt.alpha, ctxt.beta, ctxt.outputGpu->data());
}

#endif

}   // px
