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

#include "Gemm.h"

namespace px {

void cublasGemm(const CublasContext& ctx, bool tA, bool tB, int m, int n, int k, float alpha, const float* a, int lda,
                const float* b, int ldb, float beta, float* c, int ldc)
{
    // Things are re-arranged here due to the fact that cuBLAS uses column-major order.

    auto transA = tB ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto transB = tA ? CUBLAS_OP_T : CUBLAS_OP_N;

    auto M = n;
    auto N = m;
    auto K = k;
    auto* A = b;
    auto* B = a;
    auto LDA = ldb;
    auto LDB = lda;

    auto status = cublasSgemm(ctx, transA, transB, M, N, K, &alpha, A, LDA, B, LDB, &beta, c, ldc);

    PX_CHECK_CUBLAS(status);
}

}   // px
