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

void im2col_cpu(const float* im, int channels, int height, int width, int ksize, int stride, int pad, float* dataCol);
void addBias(float* output, const float* biases, int batch, int n, int size);
void random_generate_cpu(float* ptr, std::size_t n, float a = 0.f, float b = 1.f);

}   // px

#endif // PIXIENN_CPUUTIL_H
