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
#include "Utility.h"

namespace px {

std::string fmtInt(int number)
{
    std::locale locale("");

    std::ostringstream ss;
    ss.imbue(locale);
    ss << number;

    std::string output = ss.str();

    return output;
}

cv::Rect2f darkBox(const cv::Rect2f& box, const cv::Size& size)
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

cv::Rect2f lightBox(const cv::Rect2f& darkBox, const cv::Size& size)
{
    auto x1 = darkBox.x * size.width;
    auto y1 = darkBox.y * size.height;
    auto w = darkBox.width * size.width;
    auto h = darkBox.height * size.height;

    auto x = x1 - w / 2.0f;
    auto y = y1 - h / 2.0f;

 /*   x = std::max(0.0f, std::min<float>(x, size.width - 1));
    y = std::max(0.0f, std::min<float>(y, size.height - 1));
    w = std::max(1.0f, std::min<float>(w, size.width - x));
    h = std::max(1.0f, std::min<float>(h, size.height - y));*/

    return { x, y, w, h };
}

}