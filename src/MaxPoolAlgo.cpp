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
#include "MaxPoolAlgo.h"

#ifdef USE_CUDA

#include "PoolKernels.cuh"

#endif

namespace px {

void maxPoolForward(const MaxPoolContext& ctxt)
{
    int wOffset = -ctxt.padding / 2;
    int hOffset = -ctxt.padding / 2;

    auto ih = ctxt.height;
    auto iw = ctxt.width;
    auto oh = ctxt.outHeight;
    auto ow = ctxt.outWidth;
    auto c = ctxt.channels;
    const auto min = -std::numeric_limits<float>::max();

    const auto* pin = ctxt.input->data();
    auto* pout = ctxt.output->data();

    for (auto b = 0; b < ctxt.batch; ++b) {
        for (auto k = 0; k < c; ++k) {
            for (auto i = 0; i < oh; ++i) {
                for (auto j = 0; j < ow; ++j) {
                    auto outIndex = j + ow * (i + oh * (k + c * b));
                    float max = min;

                    for (auto n = 0; n < ctxt.kernel; ++n) {
                        for (auto m = 0; m < ctxt.kernel; ++m) {
                            auto curH = hOffset + i * ctxt.stride + n;
                            auto curW = wOffset + j * ctxt.stride + m;
                            auto index = curW + iw * (curH + ih * (k + b * c));
                            auto valid = (curH >= 0 && curH < ih && curW >= 0 && curW < iw);
                            auto val = valid ? pin[index] : min;
                            max = (val > max) ? val : max;
                        }
                    }

                    pout[outIndex] = max;
                }
            }
        }
    }
}

#ifdef USE_CUDA

void maxPoolForwardGpu(const MaxPoolContext& ctxt)
{
    auto outputs = ctxt.batch * ctxt.channels * ctxt.outHeight * ctxt.outWidth;

    maxpoolGpu(outputs, ctxt.height, ctxt.width, ctxt.channels, ctxt.stride, ctxt.kernel, ctxt.padding,
               ctxt.inputGpu->data(), ctxt.outputGpu->data());

}

#endif // USE_CUDA


}