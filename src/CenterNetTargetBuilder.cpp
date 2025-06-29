/********************************************************************************
* Copyright 2020-2025 Thomas A. Rieck, All Rights Reserved
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

/**
 * @class CenterNetTargetBuilder
 * @brief Generates training targets for CenterNet-style object detection.
 *
 * This class prepares the dense regression targets needed to train a
 * CenterNet model head from ground truth bounding boxes. The targets are
 * designed to supervise a fully convolutional detection head, which predicts:
 *
 *   1. A per-class heatmap indicating object center likelihood.
 *   2. A size map encoding object width and height at each center.
 *   3. A subpixel offset map for refining the object center position.
 *   4. A binary mask that indicates where valid object centers exist.
 *
 * Ground truth boxes are provided in normalized format (relative to image
 * width and height), and the feature map resolution is derived from the
 * image dimensions and stride (downsampling factor of the backbone).
 *
 * Each target is stored in a flat PxCpuVector in the following format:
 *
 *   - heatmap  : [numClasses x H x W] — Gaussian peaks per object center
 *   - size     : [2 x H x W] — width and height (normalized) per center
 *   - offset   : [2 x H x W] — subpixel center offset (cx, cy) - (int(cx), int(cy))
 *   - mask     : [H x W] — 1 if a center exists at that location, 0 otherwise
 *
 * The heatmap uses a 2D Gaussian kernel centered at the integer coordinate
 * nearest the ground truth center, with radius computed to guarantee a
 * minimum IOU between the predicted and actual box.
 *
 * This structure allows for efficient dense detection by enabling the network
 * to predict object locations and sizes directly on the feature map, without
 * needing anchor boxes or proposal mechanisms.
 *
 * Example usage:
 *     CenterNetTargetBuilder builder(numClasses, stride, imageW, imageH);
 *     CenterNetTargets targets = builder.buildTargets(groundTruthVec);
 *
 * Reference: CenterNet (Objects as Points), Zhou et al. (2019)
 *  https://arxiv.org/abs/1904.07850
 *
 * @author
 *     Thomas A. Rieck (2020–2025)
 */


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

    // Heatmap: [numClasses x H x W], where each channel holds a 2D Gaussian for object centers
    targets.heatmap = PxCpuVector(numClasses_ * fmapH_ * fmapW_, 0.0f);

    // Size: [2 x H x W], where channel 0 holds width and channel 1 holds height (normalized to image size)
    targets.size = PxCpuVector(2 * fmapH_ * fmapW_, 0.0f);

    // Offset: [2 x H x W], holds subpixel offset from the integer (cx, cy) to the true center
    targets.offset = PxCpuVector(2 * fmapH_ * fmapW_, 0.0f);

    // Mask: [H x W], indicates whether a valid object center exists at each location
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

/**
 * @brief Computes the Gaussian radius for a given object size to ensure sufficient heatmap overlap.
 *
 * This function calculates the radius of a 2D Gaussian to be drawn on the heatmap
 * such that the Gaussian blob overlaps with the ground truth bounding box by at least
 * a specified minimum IoU (Intersection over Union), typically 0.7.
 *
 * The formula is derived from geometric constraints and ensures that the effective
 * receptive field of the heatmap Gaussian provides sufficient supervision signal.
 *
 * @param width  Width of the object (in feature map coordinates)
 * @param height Height of the object (in feature map coordinates)
 * @return Radius (in pixels) of the 2D Gaussian kernel
 */
float CenterNetTargetBuilder::gaussianRadius(float width, float height) const
{
    // Desired minimum IoU between the generated blob and ground truth box
    constexpr auto minOverlap = 0.7f; // Minimum overlap for Gaussian radius

    // Coefficients for a quadratic equation derived from geometric constraints
    constexpr auto a1 = 1.0f;

    // Linear coefficient: sum of box width and height
    auto b1 = height + width;

    // Constant term: derived from the desired minimum IoU
    auto c1 = width * height * (1 - minOverlap) / (1 + minOverlap);

    // Discriminant of the quadratic equation
    auto sq1 = std::sqrt(b1 * b1 - 4 * a1 * c1);

    // Positive root of the quadratic — gives the required radius
    auto r1 = (b1 + sq1) / 2;

    // Clamp to non-negative radius to ensure numerical safety
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

} // namespace px
