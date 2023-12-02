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
#include "ConnAlgo.h"
#include "CpuUtil.h"

#ifdef USE_CUDA
#include <cublas_api.h>

#include "Cudnn.h"

#endif

namespace px {

void connectedForward(const ConnContext& ctxt)
{
    auto m = ctxt.batch;
    auto n = ctxt.outputs;
    auto k = ctxt.inputs;
    auto* a = ctxt.input->data();
    auto* b = ctxt.weights->data();
    auto* c = ctxt.output->data();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a, k, b, k, 1.0f, c, n);
}

#ifdef USE_CUDA

void connectedForwardGpu(const ConnContext& ctxt)
{
    auto m = ctxt.outputs;
    auto n = ctxt.batch;
    auto k = ctxt.inputs;
    auto* a = ctxt.weightsGpu->data();
    auto* b = ctxt.inputGpu->data();
    auto* c = ctxt.outputGpu->data();

    float alpha = 1.0f, beta = 1.0f;


    auto status = cublasSgemm(*ctxt.cublasContext,
                              CUBLAS_OP_T,  /* transpose A */
                              CUBLAS_OP_N,  /* transpose B */
                              m,            /* M */
                              n,            /* N */
                              k,            /* K */
                              &alpha,       /* alpha */
                              a,            /* A */
                              k,            /* lda */
                              b,            /* B */
                              k,            /* ldb */
                              &beta,        /* beta */
                              c,            /* C */
                              m             /* ldc */
    );

    PX_CHECK_CUBLAS(status);
}

#endif

}   // px
