/********************************************************************************
* Copyright 2020 Thomas A. Rieck, All Rights Reserved
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

#ifndef PIXIENN_UTILITY_H
#define PIXIENN_UTILITY_H

namespace px {

void im2col_cpu(const float* im, int channels, int height, int width, int ksize, int stride, int pad, float* dataCol);

void normalize_cpu(float* x, float* mean, float* variance, int batch, int filters, int spatial);
void scale_bias(float* output, float* scales, int batch, int n, int size);
void add_bias(float* output, float* biases, int batch, int n, int size);

}   // px

#endif // PIXIENN_UTILITY_H
