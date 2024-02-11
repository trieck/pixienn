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

#include "DarkBox.h"

namespace px {

DarkBox::DarkBox() : x_{ 0.0f }, y_{ 0.0f }, w_{ 0.0f }, h_{ 0.0f }
{
}

DarkBox::DarkBox(float x, float y, float w, float h) : x_{ x }, y_{ y }, w_{ w }, h_{ h }
{
}

DarkBox::DarkBox(const cv::Rect2f& rect)
{
    *this = rect;
}

DarkBox& DarkBox::operator=(const cv::Rect2f& rect)
{
    x_ = rect.x + rect.width / 2.0f;
    y_ = rect.y + rect.height / 2.0f;
    w_ = rect.width;
    h_ = rect.height;

    return *this;
}

float DarkBox::x() const noexcept
{
    return x_;
}

float DarkBox::y() const noexcept
{
    return y_;
}

float DarkBox::w() const noexcept
{
    return w_;
}

float DarkBox::h() const noexcept
{
    return h_;
}

float& DarkBox::x() noexcept
{
    return x_;
}

float& DarkBox::y() noexcept
{
    return y_;
}

float& DarkBox::w() noexcept
{
    return w_;
}

float& DarkBox::h() noexcept
{
    return h_;
}

float DarkBox::left() const noexcept
{
    if (w_ > 0.0f) {
        return x_ - w_ / 2.0f;
    }

    return x_;
}

float DarkBox::right() const noexcept
{
    if (w_ > 0.0f) {
        return x_ + w_ / 2.0f;
    }

    return x_;
}

float DarkBox::top() const noexcept
{
    if (h_ > 0.0f) {
        return y_ - h_ / 2.0f;
    }

    return y_;
}

float DarkBox::bottom() const noexcept
{
    if (h_ > 0.0f) {
        return y_ + h_ / 2.0f;
    }

    return y_;
}

cv::Rect2f DarkBox::rect() const noexcept
{
    return cv::Rect2f{ left(), top(), w_, h_ };
}

float DarkBox::intersection(const DarkBox& other) const noexcept
{
    cv::Rect2f rc1(left(), top(), w(), h());
    cv::Rect2f rc2(other.left(), other.top(), other.w(), other.h());

    auto intersection = rc1 & rc2;

    auto area = intersection.area();

    return area;
}

float DarkBox::unionArea(const DarkBox& other) const noexcept
{
    cv::Rect2f rc1(left(), top(), w(), h());
    cv::Rect2f rc2(other.left(), other.top(), other.w(), other.h());

    auto area = (rc1 | rc2).area();

    return area;
}

float DarkBox::iou(const DarkBox& other) const noexcept
{
    auto iarea = intersection(other);
    auto uarea = unionArea(other);
    if (uarea == 0.0f) {
        return 0.0f;
    }

    return iarea / uarea;
}

bool DarkBox::empty() const noexcept
{
    return w_ == 0.0f || h_ == 0.0f;
}

bool operator==(const DarkBox& box1, const DarkBox& box2) noexcept
{
    return box1.x() == box2.x() && box1.y() == box2.y() && box1.w() == box2.w() && box1.h() == box2.h();
}

bool operator!=(const DarkBox& box1, const DarkBox& box2) noexcept
{
    return !(box1 == box2);
}

DarkBox darkBox(const cv::Rect2f& box, const cv::Size& size)
{
    auto dw = 1.0f / size.width;
    auto dh = 1.0f / size.height;

    float x1 = box.tl().x;
    float y1 = box.tl().y;
    float x2 = x1 + box.width;
    float y2 = y1 + box.height;

    auto x = (x1 + x2) / 2.0f;
    auto y = (y1 + y2) / 2.0f;
    auto w = x2 - x1;
    auto h = y2 - y1;

    x *= dw;
    w *= dw;
    y *= dh;
    h *= dh;

    return { x, y, w, h };
}

cv::Rect2f lightBox(const DarkBox& darkBox, const cv::Size& size)
{
    auto x1 = darkBox.x() * size.width;
    auto y1 = darkBox.y() * size.height;
    auto w = darkBox.w() * size.width;
    auto h = darkBox.h() * size.height;

    auto x = x1 - w / 2.0f;
    auto y = y1 - h / 2.0f;

    return { x, y, w, h };
}

}   // px
