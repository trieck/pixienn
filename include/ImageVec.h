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

#ifndef PIXIE_IMAGEVEC_H
#define PIXIE_IMAGEVEC_H

#include <opencv2/core/types.hpp>
#include "PxTensor.h"

namespace px {

struct ImageVec
{
    ImageVec();
    ImageVec(const ImageVec& rhs);
    ImageVec(ImageVec&& rhs);

    ImageVec& operator=(const ImageVec& rhs);
    ImageVec& operator=(ImageVec&& rhs);

    std::string imagePath;
    PxCpuVector data;
    cv::Size size;
    cv::Size originalSize;
    int channels;
};

}   // px


#endif // PIXIE_IMAGEVEC_H
