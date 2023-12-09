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
#include "ConvAlgo.h"
#include "CpuUtil.h"

namespace px {

void convolutionalForward(const ConvContext& ctxt)
{
    auto m = ctxt.filters / ctxt.groups;
    auto n = ctxt.outWidth * ctxt.outHeight;
    auto k = ctxt.kernel * ctxt.kernel * ctxt.channels / ctxt.groups;

    int nweights = ctxt.weights->size();
    const auto* pweights = ctxt.weights->data();

    const auto* pin = ctxt.input->data();
    auto* pout = ctxt.output->data();

    auto alpha = 1.0f;
    auto beta = 1.0f;

    for (auto i = 0; i < ctxt.batch; ++i) {
        for (auto j = 0; j < ctxt.groups; ++j) {
            const auto* im = pin + (i * ctxt.groups + j) * ctxt.channels / ctxt.groups * ctxt.height * ctxt.width;
            const auto* a = pweights + j * nweights / ctxt.groups;
            const auto* b = ctxt.kernel == 1 ? im : ctxt.column->data();
            auto* c = pout + (i * ctxt.groups + j) * n * m;

            if (ctxt.kernel != 1) {
                im2ColCpu(im, ctxt.channels / ctxt.groups, ctxt.height, ctxt.width, ctxt.kernel, ctxt.stride,
                          ctxt.padding, ctxt.column->data());
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);
        }
    }
}

void convolutionalBackward(const ConvContext& ctxt)
{
    auto m = ctxt.filters / ctxt.groups;
    auto n = ctxt.kernel * ctxt.kernel * ctxt.channels / ctxt.groups;
    auto k = ctxt.outWidth * ctxt.outHeight;

    int nweights = ctxt.weights->size();
    const auto* pweights = ctxt.weights->data();

    const auto* pin = ctxt.input->data();
    auto* pout = ctxt.output->data();

    auto alpha = 1.0f;
    auto beta = 1.0f;

    for (auto i = 0; i < ctxt.batch; ++i) {
        for (auto j = 0; j < ctxt.groups; ++j) {
            auto* a = ctxt.delta + (i * ctxt.groups + j) * m * k;
            float* b = ctxt.column->data(); // WTF ctxt.netWorkspace;
            auto* c = ctxt.weightUpdates + j * ctxt.nweights / ctxt.groups;

            auto* im = pin + (i * ctxt.groups + j) * ctxt.channels / ctxt.groups * ctxt.height * ctxt.width;

            // float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            /*const auto* im = pin + (i * ctxt.groups + j) * ctxt.channels / ctxt.groups * ctxt.height * ctxt.width;
            const auto* a = pweights + j * nweights / ctxt.groups;
            const auto* b = ctxt.kernel == 1 ? im : ctxt.column->data();
            auto* c = pout + (i * ctxt.groups + j) * n * m;

            if (ctxt.kernel != 1) {
                im2ColCpu(im, ctxt.channels / ctxt.groups, ctxt.height, ctxt.width, ctxt.kernel, ctxt.stride,
                          ctxt.padding, ctxt.column->data());
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);*/
        }
    }
}

#ifdef USE_CUDA

void convolutionalForwardGpu(const ConvContext& ctxt)
{
    float alpha = 1.f;
    float beta = 1.f;

    auto status = cudnnConvolutionForward(*ctxt.cudnnContext,
                                          &alpha,
                                          *ctxt.xDesc,
                                          ctxt.inputGpu->data(),
                                          *ctxt.wDesc,
                                          ctxt.weightsGpu->data(),
                                          *ctxt.convDesc,
                                          ctxt.bestAlgo,
                                          ctxt.workspace->data(),
                                          ctxt.workspace->size() * sizeof(float),
                                          &beta,
                                          *ctxt.yDesc,
                                          ctxt.outputGpu->data());
    PX_CHECK_CUDNN(status);
}

#endif // USE_CUDA

}   // px