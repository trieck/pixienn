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

#ifndef PIXIENN_IMAGETOCOL_H
#define PIXIENN_IMAGETOCOL_H

namespace px {

void im2col_cpu(const float* data_im, int channels, int height, int width, int ksize, int stride, int pad,
                float* data_col);

}   // px

#endif // PIXIENN_IMAGETOCOL_H
