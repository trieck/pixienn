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
#include "Timer.h"

namespace px {

static float boxIntersection(const cv::Rect2f& a, const cv::Rect2f& b)
{
    return float((a & b).area());
}

static float boxUnion(const cv::Rect2f& a, const cv::Rect2f& b)
{
    return float((a | b).area());
}

static float boxIoU(const cv::Rect2f& a, const cv::Rect2f& b)
{
    auto _inter = boxIntersection(a, b);
    auto _union = boxUnion(a, b);
    auto result = _inter / _union;
    if (std::isnan(result) || std::isinf(result)) {
        return 0.0f;
    }

    return result;
}

Detections nms(const Detections& detects, float threshold)
{
    Detections output(detects);

    std::vector<bool> discard(detects.size(), false);

    for (auto i = 0; i < detects.size(); ++i) {
        for (auto j = 0; j < detects.size(); ++j) {
            if (i == j || discard[j]) {
                continue;
            }

            if (detects[i].classIndex() != detects[j].classIndex()) {
                continue;
            }

            if (boxIoU(detects[i].box(), detects[j].box()) > threshold) {
                if (detects[i].prob() < detects[j].prob()) {
                    discard[i] = true;
                } else {
                    discard[j] = true;
                }
            }
        }
    }

    auto pred = [&discard, &output](const auto& detection) {
        return discard[&detection - &output[0]];
    };

    output.erase(std::remove_if(output.begin(), output.end(), pred), output.end());

    return output;
}

}   // px
