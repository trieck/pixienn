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

#include "Box.h"

namespace px {

static float boxIntersection(const cv::Rect2f& a, const cv::Rect2f& b)
{
    return float((a & b).area());
}

static float boxUnion(const cv::Rect2f& a, const cv::Rect2f& b)
{
    return float((a | b).area());
}

float boxIou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    auto _inter = boxIntersection(a, b);
    auto _union = boxUnion(a, b);

    auto result = _inter / _union;

    return result;
}

float boxRmse(const cv::Rect2f& a, const cv::Rect2f& b)
{
    return std::sqrt(pow(a.x - b.x, 2) +
                pow(a.y - b.y, 2) +
                pow(a.width - b.width, 2) +
                pow(a.height - b.height, 2));
}

void nms(Detections& detects, float threshold)
{
    for (auto i = 0; i < detects.size(); ++i) {
        for (auto j = i + 1; j < detects.size(); ++j) {
            if (boxIou(detects[i].box(), detects[j].box()) > threshold) {
                for (auto k = 0; k < detects[i].size(); ++k) {
                    if (detects[i][k] < detects[j][k]) {
                        detects[i][k] = 0;
                    } else {
                        detects[j][k] = 0;
                    }
                }
            }
        }
    }
}

}   // px
