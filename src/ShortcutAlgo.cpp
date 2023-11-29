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

void shortcutForward(const ShortcutContext& ctxt)
{
    int stride = ctxt.width / ctxt.outWidth;
    int sample = ctxt.outWidth / ctxt.width;

    if (stride < 1) stride = 1;
    if (sample < 1) sample = 1;
    int minw = (ctxt.width < ctxt.outWidth) ? ctxt.width : ctxt.outWidth;
    int minh = (ctxt.height < ctxt.outHeight) ? ctxt.height : ctxt.outHeight;
    int minc = (ctxt.channels < ctxt.outChannels) ? ctxt.channels : ctxt.outChannels;

    const auto* add = ctxt.add->data();
    auto* out = ctxt.output->data();

    int i, j, k, b;
    for (b = 0; b < ctxt.batch; ++b) {
        for (k = 0; k < minc; ++k) {
            for (j = 0; j < minh; ++j) {
                for (i = 0; i < minw; ++i) {
                    int out_index =
                            i * sample + ctxt.outWidth * (j * sample + ctxt.outWidth * (k + ctxt.outChannels * b));
                    int add_index = i * stride + ctxt.width * (j * stride + ctxt.height * (k + ctxt.channels * b));
                    out[out_index] = ctxt.alpha * out[out_index] + ctxt.beta * add[add_index];
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
