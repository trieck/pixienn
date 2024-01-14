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

#pragma once

#include <opencv2/core/types.hpp>

namespace px {

class DarkBox
{
public:
    DarkBox();
    DarkBox(float x, float y, float w, float h);
    DarkBox(const DarkBox&) = default;
    DarkBox(DarkBox&&) = default;

    DarkBox(const cv::Rect2f& rect);
    DarkBox& operator=(const cv::Rect2f& rect);

    ~DarkBox() = default;

    DarkBox& operator=(const DarkBox&) = default;
    DarkBox& operator=(DarkBox&&) = default;

    float x() const noexcept;
    float y() const noexcept;
    float w() const noexcept;
    float h() const noexcept;

    float& x() noexcept;
    float& y() noexcept;
    float& w() noexcept;
    float& h() noexcept;

    float left() const noexcept;
    float right() const noexcept;
    float top() const noexcept;
    float bottom() const noexcept;

    float intersection(const DarkBox& other) const noexcept;
    float unionArea(const DarkBox& other) const noexcept;
    float iou(const DarkBox& other) const noexcept;

    cv::Rect2f rect() const noexcept;
    bool empty() const noexcept;

private:
    float x_;    // x-coordinate of the center of the box
    float y_;    // y-coordinate of the center of the box
    float w_;    // width of the box
    float h_;    // height of the box
};

bool operator==(const DarkBox& box1, const DarkBox& box2) noexcept;
bool operator!=(const DarkBox& box1, const DarkBox& box2) noexcept;

DarkBox darkBox(const cv::Rect2f& box, const cv::Size& size);
cv::Rect2f lightBox(const DarkBox& darkBox, const cv::Size& size);

}   // px
