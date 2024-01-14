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

#pragma once

#include "Common.h"
#include "PxTensor.h"

namespace px {

void batchNormForward(bool training,
                      int batch,
                      int channels,
                      int height,
                      int width,
                      const PxCpuVector& input,
                      PxCpuVector& output,
                      PxCpuVector& mean,
                      PxCpuVector& variance,
                      PxCpuVector& rollingMean,
                      PxCpuVector& rollingVariance,
                      PxCpuVector& scales,
                      PxCpuVector& biases,
                      PxCpuVector& x,
                      PxCpuVector& xNorm);

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
                       PxCpuVector& xNorm);

}   // px
