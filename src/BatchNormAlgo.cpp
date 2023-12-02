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
#include "BatchNormAlgo.h"
#include "CpuUtil.h"

namespace px {

static void normalizeCpu(float* x, const float* mean, const float* variance, int batch, int filters, int spatial)
{
    for (auto b = 0; b < batch; ++b) {
        for (auto f = 0; f < filters; ++f) {
            for (auto i = 0; i < spatial; ++i) {
                auto index = b * filters * spatial + f * spatial + i;
                x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + .000001f);
            }
        }
    }
}

static void scaleBias(float* output, const float* scales, int batch, int n, int size)
{
    for (auto b = 0; b < batch; ++b) {
        for (auto i = 0; i < n; ++i) {
            for (auto j = 0; j < size; ++j) {
                output[(b * n + i) * size + j] *= scales[i];
            }
        }
    }
}

void batchNormForward(const BNContext& ctxt)
{
    ctxt.output->copy(*ctxt.input);

    auto b = ctxt.batch;
    auto c = ctxt.channels;
    auto size = ctxt.outHeight * ctxt.outWidth;

    normalizeCpu(ctxt.output->data(), ctxt.rollingMean->data(), ctxt.rollingVar->data(), b, c, size);
    scaleBias(ctxt.output->data(), ctxt.scales->data(), b, c, size);
    addBias(ctxt.output->data(), ctxt.biases->data(), b, c, size);
}

#ifdef USE_CUDA

void batchNormForwardGpu(const BNContext& ctxt)
{
    float alpha = 1;
    float beta = 0;

    auto status = cudnnBatchNormalizationForwardInference(*ctxt.cudnnContext,
                                                          CUDNN_BATCHNORM_SPATIAL,
                                                          &alpha,
                                                          &beta,
                                                          *ctxt.dstTens,
                                                          ctxt.inputGpu->data(),
                                                          *ctxt.dstTens,
                                                          ctxt.outputGpu->data(),
                                                          *ctxt.normTens,
                                                          ctxt.scalesGpu->data(),
                                                          ctxt.biasesGpu->data(),
                                                          ctxt.rollingMeanGpu->data(),
                                                          ctxt.rollingVarGpu->data(),
                                                          0.00001);
    PX_CHECK_CUDNN(status);
}

#endif // USE_CUDA

}   // px
