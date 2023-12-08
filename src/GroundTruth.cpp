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

#include "Common.h"
#include "GroundTruth.h"

namespace px {

ImageTruths::ImageTruths()
{
}

ImageTruths::~ImageTruths()
{
}

ImageTruths::const_iterator ImageTruths::begin() const noexcept
{
    return truths_.begin();
}

ImageTruths::const_iterator ImageTruths::end() const noexcept
{
    return truths_.end();
}

ImageTruths::iterator ImageTruths::begin() noexcept
{
    return truths_.begin();
}

ImageTruths::iterator ImageTruths::end() noexcept
{
    return truths_.end();
}

void ImageTruths::emplaceBack(ImageTruth&& item)
{
    truths_.emplace_back(std::move(item));
}

const ImageTruth& ImageTruths::operator[](size_type index) const
{
    PX_CHECK(index <= truths_.size(), "Index out of range.");

    return truths_[index];
}

auto ImageTruths::size() const noexcept -> size_type
{
    return truths_.size();
}

const GroundTruthVec& ImageTruths::groundTruth(ImageTruths::size_type index) const
{
    PX_CHECK(index <= truths_.size(), "Index out of range.");

    return truths_[index].truth;
}

}
