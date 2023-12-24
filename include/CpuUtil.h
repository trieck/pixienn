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

#ifndef PIXIENN_CPUUTIL_H
#define PIXIENN_CPUUTIL_H

#include "Common.h"

namespace px {


float magArray(const float* a, int n);
float sumArray(const float* a, int n);
void addBias(float* output, const float* biases, int batch, int n, int size);
void backwardBias(float* biasUpdates, const float* delta, int batch, int n, int size);
void backwardScaleCpu(const float* xNorm, const float* delta, int batch, int n, int size, float* scaleUpdates);
void col2ImCpu(const float* dataCol, int channels, int height, int width, int ksize, int stride, int pad,
               float* dataIm);
void col2ImCpuExt(const float* dataCol, const int channels,
                  const int height, const int width, const int kernelH, const int kernelW,
                  const int padH, const int padW,
                  const int strideH, const int strideW,
                  const int dilationH, const int dilationW,
                  float* dataIm);

void constrain(int n, float alpha, float* x, int incX);

void im2ColCpu(const float* im, int channels, int height, int width, int ksize, int stride, int pad, float* dataCol);

void im2ColCpuExt(const float* im, const int channels,
                  const int height, const int width, const int kernelH, const int kernelW,
                  const int padH, const int padW,
                  const int strideH, const int strideW,
                  const int dilationH, const int dilationW,
                  float* dataCol);


void flatten(float* x, int size, int layers, int batch, bool forward);

void meanCpu(const float* x, int batch, int filters, int spatial, float* mean);
void meanDeltaCpu(const float* delta, const float* variance, int batch, int filters, int spatial, float* meanDelta);
void normalizeCpu(float* x, const float* mean, const float* variance, int batch, int filters, int spatial);
void normalizeDeltaCpu(const float* x, const float* mean, const float* variance, const float* meanDelta,
                       const float* varianceDelta, int batch, int filters, int spatial, float* delta);
void randomCpu(float* ptr, std::size_t n, float a = 0.f, float b = 1.f);
void scaleBias(float* output, const float* scales, int batch, int n, int size);
void varianceCpu(const float* x, float* mean, int batch, int filters, int spatial, float* variance);
void varianceDeltaCpu(const float* x, const float* delta, const float* mean, const float* variance, int batch,
                      int filters, int spatial, float* varianceDelta);

}   // px

#endif // PIXIENN_CPUUTIL_H
