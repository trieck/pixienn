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

#include "CenterNetTargetBuilder.h"

namespace px {

CenterNetTargetBuilder::CenterNetTargetBuilder(int numClasses, int stride, int imageW, int imageH)
        : numClasses_(numClasses), stride_(stride), imageW_(imageW), imageH_(imageH)
{
    fmapW_ = imageW / stride_;
    fmapH_ = imageH / stride_;
}

CenterNetTargets CenterNetTargetBuilder::buildTargets(const GroundTruthVec& truth)
{
    CenterNetTargets targets;

    targets.heatmap = PxCpuVector(numClasses_ * fmapH_ * fmapW_, 0.0f);
    targets.size = PxCpuVector(2 * fmapH_ * fmapW_, 0.0f);
    targets.offset = PxCpuVector(2 * fmapH_ * fmapW_, 0.0f);
    targets.mask = PxCpuVector(fmapH_ * fmapW_, 0.0f);

    for (const auto& gt: truth) {
        auto cx = gt.box.x() * imageW_;
        auto cy = gt.box.y() * imageH_;
        auto width = gt.box.w() * imageW_;
        auto height = gt.box.h() * imageH_;

        auto cxFmap = cx / stride_;
        auto cyFmap = cy / stride_;
        auto cxInt = static_cast<int>(cxFmap);
        auto cyInt = static_cast<int>(cyFmap);

        if (cxInt < 0 || cxInt >= fmapW_ || cyInt < 0 || cyInt >= fmapH_) {
            continue; // Skip if the center is outside the feature map
        }

        auto radius = gaussianRadius(width / stride_, height / stride_);

        drawGaussian(targets.heatmap, gt.classId, cxInt, cyInt, radius);

        auto dx = cxFmap - cxInt;
        auto dy = cyFmap - cyInt;

        auto index = cyInt * fmapW_ + cxInt;

        targets.offset[0 * fmapH_ * fmapW_ + index] = dx;
        targets.offset[1 * fmapH_ * fmapW_ + index] = dy;

        targets.size[0 * fmapH_ * fmapW_ + index] = width / imageW_;   // normalized
        targets.size[1 * fmapH_ * fmapW_ + index] = height / imageH_;  // normalized

        targets.mask[index] = 1.0f;
    }

    return targets;
}

float CenterNetTargetBuilder::gaussianRadius(float width, float height) const
{
    auto minOverlap = 0.7f; // Minimum overlap for Gaussian radius

    auto a1 = 1.0f;
    auto b1 = height + width;
    auto c1 = width * height * (1 - minOverlap) / (1 + minOverlap);
    auto sq1 = std::sqrt(b1 * b1 - 4 * a1 * c1);
    auto r1 = (b1 + sq1) / 2;

    return std::max(0.0f, r1);
}

void CenterNetTargetBuilder::drawGaussian(PxCpuVector& heatmap, int classId, int cx, int cy, float radius)
{
    auto diameter = static_cast<int>(2 * radius + 1);
    auto radiusInt = static_cast<int>(radius);
    auto sigma = diameter / 6.0f; // 3 sigma rule
    auto twoSigma2 = 2 * sigma * sigma;

    // Precompute Gaussian values
    std::vector<std::vector<float>> kernel(diameter, std::vector<float>(diameter));
    for (auto y = 0; y < diameter; ++y) {
        for (auto x = 0; x < diameter; ++x) {
            auto dx = x - radius;
            auto dy = y - radius;
            kernel[y][x] = std::exp(-(dx * dx + dy * dy) / twoSigma2);
        }
    }

    // Place Gaussian into heatmap
    for (auto y = -radiusInt; y <= radiusInt; ++y) {
        auto py = cy + y;
        if (py < 0 || py >= fmapH_) {
            continue; // Skip if outside feature map height
        }

        for (auto x = -radiusInt; x <= radiusInt; ++x) {
            auto px = cx + x;
            if (px < 0 || px >= fmapW_) {
                continue; // Skip if outside feature map width
            }

            auto value = kernel[y + radiusInt][x + radiusInt];

            auto index = classId * fmapH_ * fmapW_ + py * fmapW_ + px;
            heatmap[index] = std::max(heatmap[index], value); // preserve max if overlapping
        }
    }
}

float
CenterNetTargetBuilder::focalLoss(const PxCpuVector& pred, const PxCpuVector& target, int numClasses, int H, int W,
                                  float alpha, float beta)
{
    float loss = 0.0f;
    auto spatialSize = H * W;

    for (auto c = 0; c < numClasses; ++c) {
        auto base = c * spatialSize;
        for (auto i = 0; i < spatialSize; ++i) {
            auto p = std::clamp(pred[base + i], 1e-6f, 1.0f - 1e-6f);
            auto y = target[base + i];

            if (y == 1.0f) {
                // positive center
                loss += std::pow(1 - p, alpha) * std::log(p);
            } else {
                // background
                loss += std::pow(1 - y, beta) * std::pow(p, alpha) * std::log(1 - p);
            }
        }
    }

    return -loss / numClasses;
}

} // namespace px
