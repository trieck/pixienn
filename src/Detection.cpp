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

Detection::Detection(int classes, cv::Rect box, float objectness) : box_(std::move(box)), objectness_(objectness)
{
    prob_.resize(classes);
}

float& Detection::operator[](int clazz)
{
    PX_CHECK(clazz < prob_.size(), "Class out of range");

    return prob_[clazz];
}

const float& Detection::operator[](int clazz) const
{
    PX_CHECK(clazz < prob_.size(), "Class out of range");

    return prob_[clazz];
}

int Detection::size() const noexcept
{
    return (int)prob_.size();
}

const cv::Rect& Detection::box() const noexcept
{
    return box_;
}

const std::vector<float>& Detection::prob() const noexcept
{
    return prob_;
}

void Detection::setMaxClass(int max)
{
    PX_CHECK(max >= 0 && max < prob_.size(), "Index out of range.");
    maxClass_ = max;
}

int Detection::maxClass() const noexcept
{
    return maxClass_;
}

float Detection::max() const noexcept
{
    return prob_[maxClass_];
}

}   // px

