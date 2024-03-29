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

#include "Detection.h"
#include "Error.h"

namespace px {

Detection::Detection(cv::Rect2f box, int batchId, int classIndex, float prob)
        : box_(std::move(box)), batchId_(batchId), classIndex_(classIndex), prob_(prob)
{
}

const cv::Rect2f& Detection::box() const noexcept
{
    return box_;
}

float Detection::prob() const noexcept
{
    return prob_;
}

int Detection::classIndex() const noexcept
{
    return classIndex_;
}

int Detection::batchId() const noexcept
{
    return batchId_;
}

}   // px

