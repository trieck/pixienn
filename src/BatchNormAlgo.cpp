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

#include <cblas.h>

#include "Common.h"
#include "BatchNormAlgo.h"
#include "CpuUtil.h"

namespace px {

void batchNormBackward(const BNContext& ctxt)
{
    backwardScaleCpu(ctxt.xNorm->data(), ctxt.delta->data(), ctxt.batch, ctxt.channels,
                     ctxt.outWidth * ctxt.outHeight, ctxt.scaleUpdates->data());

    scaleBias(ctxt.delta->data(), ctxt.scales->data(), ctxt.batch, ctxt.channels, ctxt.outHeight * ctxt.outWidth);

    meanDeltaCpu(ctxt.delta->data(), ctxt.var->data(), ctxt.batch, ctxt.channels, ctxt.outHeight * ctxt.outWidth,
                 ctxt.meanDelta->data());

    varianceDeltaCpu(ctxt.x->data(), ctxt.delta->data(), ctxt.mean->data(), ctxt.var->data(), ctxt.batch, ctxt.channels,
                     ctxt.outHeight * ctxt.outWidth, ctxt.varDelta->data());

    normalizeDeltaCpu(ctxt.x->data(), ctxt.mean->data(), ctxt.var->data(), ctxt.meanDelta->data(),
                      ctxt.varDelta->data(), ctxt.batch, ctxt.channels, ctxt.outHeight * ctxt.outWidth,
                      ctxt.delta->data());

    // FIXME: we need another way
    //  if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

void batchNormForward(const BNContext& ctxt)
{
    if (ctxt.input != ctxt.output) {
        ctxt.output->copy(*ctxt.input);
    }

    auto b = ctxt.batch;
    auto c = ctxt.channels;
    auto size = ctxt.outHeight * ctxt.outWidth;
    auto outputs = c * size;

    if (ctxt.training) {
        cblas_scopy(b * outputs, ctxt.output->data(), 1, ctxt.x->data(), 1);

        meanCpu(ctxt.output->data(), b, c, size, ctxt.mean->data());
        varianceCpu(ctxt.output->data(), ctxt.mean->data(), b, c, size, ctxt.var->data());

        cblas_sscal(c, 0.9f, ctxt.rollingMean->data(), 1);
        cblas_saxpy(c, .1f, ctxt.mean->data(), 1, ctxt.rollingMean->data(), 1);
        cblas_sscal(c, 0.9f, ctxt.rollingVar->data(), 1);
        cblas_saxpy(c, .1f, ctxt.var->data(), 1, ctxt.rollingVar->data(), 1);

        normalizeCpu(ctxt.output->data(), ctxt.mean->data(), ctxt.var->data(), b, c, size);
        cblas_scopy(b * outputs, ctxt.output->data(), 1, ctxt.xNorm->data(), 1);
    } else {
        normalizeCpu(ctxt.output->data(), ctxt.rollingMean->data(), ctxt.rollingVar->data(), b, c, size);
    }

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
