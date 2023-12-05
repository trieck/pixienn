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

#ifndef PIXIENN_GROUNDTRUTH_H
#define PIXIENN_GROUNDTRUTH_H

#include <opencv2/core/types.hpp>

#include "PxTensor.h"

namespace px {

struct GroundTruth
{
    std::size_t classId;
    float x, y, width, height;
    cv::Rect2f box;
};

using GroundTruthVec = std::vector<GroundTruth>;

struct ImageTruth
{
    PxCpuVector image;
    GroundTruthVec truth;
};

class ImageTruths
{
public:
    using C = std::vector<ImageTruth>;
    using reference = typename C::reference;
    using const_reference = typename C::const_reference;
    using iterator = typename C::iterator;
    using const_iterator = typename C::const_iterator;
    using size_type = C::size_type;

    ImageTruths();
    ImageTruths(const ImageTruths&) = default;
    ImageTruths(ImageTruths&&) = default;
    ~ImageTruths();

    ImageTruths& operator=(const ImageTruths&) = default;
    ImageTruths& operator=(ImageTruths&&) = default;
    const ImageTruth& operator[](size_type index) const;
    bool hasObject(size_type index) const noexcept;

    void emplaceBack(ImageTruth&& item);
    size_type size() const noexcept;

    iterator begin() noexcept;
    iterator end() noexcept;
    const_iterator begin() const noexcept;
    const_iterator end() const noexcept;

private:
    C truths_;
};

}

#endif // PIXIENN_GROUNDTRUTH_H
