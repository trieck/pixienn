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

#include "BatchNorm.h"
#include "CpuUtil.h"

namespace px {

void batchNormForward(bool training,
                      int batch,
                      int channels,
                      int height,
                      int width,
                      const PxCpuVector& input,
                      PxCpuVector& output,
                      PxCpuVector& mean,
                      PxCpuVector& var,
                      PxCpuVector& rollingMean,
                      PxCpuVector& rollingVar,
                      PxCpuVector& scales,
                      PxCpuVector& biases,
                      PxCpuVector& x,
                      PxCpuVector& xNorm)
{
    if (input.data() != output.data()) {
        output.copy(input);
    }

    auto size = height * width;
    auto outputs = channels * size;

    if (training) {
        cblas_scopy(batch * outputs, output.data(), 1, x.data(), 1);

        meanCpu(output.data(), batch, channels, size, mean.data());
        varianceCpu(output.data(), mean.data(), batch, channels, size, var.data());

        cblas_sscal(channels, 0.99f, rollingMean.data(), 1);
        cblas_saxpy(channels, .01f, mean.data(), 1, rollingMean.data(), 1);
        cblas_sscal(channels, 0.99f, rollingVar.data(), 1);
        cblas_saxpy(channels, .01f, var.data(), 1, rollingVar.data(), 1);

        normalizeCpu(output.data(), mean.data(), var.data(), batch, channels, size);
        cblas_scopy(batch * outputs, output.data(), 1, xNorm.data(), 1);

    } else {
        normalizeCpu(output.data(), rollingMean.data(), rollingVar.data(), batch, channels, size);
    }

    scaleBias(output.data(), scales.data(), batch, channels, size);
    addBias(output.data(), biases.data(), batch, channels, size);
}

void batchNormBackward(int batch,
                       int channels,
                       int height,
                       int width,
                       PxCpuVector& delta,
                       PxCpuVector& mean,
                       PxCpuVector& var,
                       PxCpuVector& meanDelta,
                       PxCpuVector& varDelta,
                       PxCpuVector& scales,
                       PxCpuVector& scaleUpdates,
                       PxCpuVector& biasUpdates,
                       PxCpuVector& x,
                       PxCpuVector& xNorm)
{
    backwardBias(biasUpdates.data(), delta.data(), batch, channels, height * width);

    backwardScaleCpu(xNorm.data(), delta.data(), batch, channels, width * height, scaleUpdates.data());

    scaleBias(delta.data(), scales.data(), batch, channels, height * width);

    meanDeltaCpu(delta.data(), var.data(), batch, channels, height * width, meanDelta.data());

    varianceDeltaCpu(x.data(), delta.data(), mean.data(), var.data(), batch, channels, height * width, varDelta.data());

    normalizeDeltaCpu(x.data(), mean.data(), var.data(), meanDelta.data(), varDelta.data(), batch, channels,
                      height * width, delta.data());
}

}   // px
