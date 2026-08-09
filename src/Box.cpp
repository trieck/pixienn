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

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

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

static cv::Rect2f normalizedRect(const cv::Rect2f& box)
{
    const auto x1 = std::min(box.x, box.x + box.width);
    const auto y1 = std::min(box.y, box.y + box.height);
    const auto x2 = std::max(box.x, box.x + box.width);
    const auto y2 = std::max(box.y, box.y + box.height);
    return {x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1)};
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
    // the overlap pass; detections from different images or classes
    // can never suppress one another.
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> groups;
    groups.reserve(output.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
        const auto batch = static_cast<std::uint32_t>(output[i].batchId());
        const auto cls = static_cast<std::uint32_t>(output[i].classIndex());
        const auto key = (static_cast<std::uint64_t>(batch) << 32U) | cls;
        groups[key].push_back(i);
    }

    namespace bgi = boost::geometry::index;
    using Point = boost::geometry::model::point<float, 2, boost::geometry::cs::cartesian>;
    using RTreeBox = boost::geometry::model::box<Point>;
    using RTreeValue = std::pair<RTreeBox, std::size_t>;
    constexpr std::size_t maxCandidatesPerGroup = 1024;
    for (const auto& [key, indices]: groups) {
        (void) key;
        bgi::rtree<RTreeValue, bgi::quadratic<16>> index;
        const auto count = std::min(indices.size(), maxCandidatesPerGroup);
        for (std::size_t position = 0; position < count; ++position) {
            const auto i = indices[position];
            if (discard[i]) {
                continue;
            }

            const auto box = normalizedRect(output[i].box());
            const auto query = RTreeBox(Point{box.x, box.y}, Point{box.x + box.width, box.y + box.height});
            std::vector<RTreeValue> candidates;
            index.query(bgi::intersects(query), std::back_inserter(candidates));
            bool suppressed = false;
            for (const auto& [_, j]: candidates) {
                if (boxIoU(box, normalizedRect(output[j].box())) > threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (suppressed) {
                discard[i] = true;
            } else {
                // Only retained boxes enter the index. A suppressed box must
                // never suppress a later, lower-confidence box.
                index.insert({query, i});
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
