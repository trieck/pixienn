/********************************************************************************
* Copyright 2023 Thomas A. Rieck, All Rights Reserved
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

#ifndef PIXIENN_POOLKERNELS_CUH
#define PIXIENN_POOLKERNELS_CUH

namespace px {

void maxpool_gpu(int n, int h, int w, int c, int stride, int kernel, int pad, const float* input, float* output);

}   // px

#endif // PIXIENN_POOLKERNELS_CUH