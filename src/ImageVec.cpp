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

#include "Common.h"
#include "ImageVec.h"

namespace px {

ImageVec::ImageVec() : channels(0)
{
}

ImageVec::ImageVec(const ImageVec& rhs)
{
    *this = rhs;
}

ImageVec::ImageVec(ImageVec&& rhs)
{
    *this = std::move(rhs);
}

ImageVec& ImageVec::operator=(const ImageVec& rhs)
{
    if (this != &rhs) {
        size = rhs.size;
        originalSize = rhs.originalSize;
        imagePath = rhs.imagePath;
        channels = rhs.channels;
        data = rhs.data;
    }

    return *this;
}

ImageVec& ImageVec::operator=(ImageVec&& rhs)
{
    size = std::move(rhs.size);
    originalSize = std::move(rhs.originalSize);
    imagePath = std::move(rhs.imagePath);
    data = std::move(rhs.data);
    channels = std::move(rhs.channels);

    return *this;
}

}   // px
