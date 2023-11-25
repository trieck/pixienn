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

#ifndef PIXIENN_BATCHNORM_ALGO_H
#define PIXIENN_BATCHNORM_ALGO_H

#include "PxTensor.h"

namespace px {

// Represents the context needed for a batch norm operation
struct BNContext
{
    const PxCpuVector* input = nullptr;
    PxCpuVector* output = nullptr;

    const PxCpuTensor<1>* biases = nullptr;
    const PxCpuTensor<1>* scales = nullptr;
    const PxCpuTensor<1>* rollingMean = nullptr;
    const PxCpuTensor<1>* rollingVar = nullptr;

    int batch = 0;
    int channels = 0;
    int outHeight = 0;
    int outWidth = 0;
};

void batchNormForward(const BNContext& ctxt);

}   // px


#endif // PIXIENN_BATCHNORM_ALGO_H
