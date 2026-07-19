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
    return std::max(0.0f, a.area() + b.area() - (a & b).area());
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

    std::stable_sort(output.begin(), output.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.prob() > rhs.prob();
    });

    std::vector<bool> discard(output.size(), false);

    // Validation deliberately keeps low-confidence predictions so AP can be
    // calculated over a useful precision/recall curve. Group candidates before
    // the quadratic overlap pass; detections from different images or classes
    // can never suppress one another.
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> groups;
    groups.reserve(output.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
        const auto batch = static_cast<std::uint32_t>(output[i].batchId());
        const auto cls = static_cast<std::uint32_t>(output[i].classIndex());
        const auto key = (static_cast<std::uint64_t>(batch) << 32U) | cls;
        groups[key].push_back(i);
    }

    for (const auto& [key, indices]: groups) {
        (void) key;
        for (std::size_t candidate = 0; candidate < indices.size(); ++candidate) {
            const auto i = indices[candidate];
            if (discard[i]) {
                continue;
            }

            for (std::size_t other = candidate + 1; other < indices.size(); ++other) {
                const auto j = indices[other];
                if (discard[j]) {
                    continue;
                }

                if (boxIoU(output[i].box(), output[j].box()) > threshold) {
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
