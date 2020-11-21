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

#include "Detection.h"
#include "Error.h"

namespace px {

Detection::Detection(int classes, cv::Rect2f box, float objectness) : box_(std::move(box)), objectness_(objectness)
{
    prob_.resize(classes);
}

float& Detection::operator[](int clazz)
{
    PX_CHECK(clazz < prob_.size(), "Class out of range");

    prob_[clazz];
}

const float& Detection::operator[](int clazz) const
{
    PX_CHECK(clazz < prob_.size(), "Class out of range");

    prob_[clazz];
}

int Detection::size() const noexcept
{
    return prob_.size();
}

}   // px

