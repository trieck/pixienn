#include "CenterNetTargetBuilder.h"

namespace px {

CenterNetTargetBuilder::CenterNetTargetBuilder(int classes, int featureWidth, int featureHeight)
        : classes_(classes), width_(featureWidth), height_(featureHeight)
{
    PX_CHECK(classes_ > 0, "CenterNet requires at least one class.");
    PX_CHECK(width_ > 0 && height_ > 0, "CenterNet feature-map dimensions must be positive.");
}

int CenterNetTargetBuilder::gaussianRadius(float width, float height, float minOverlap)
{
    if (width <= 0.0f || height <= 0.0f) {
        return 0;
    }

    const auto a1 = 1.0f;
    const auto b1 = height + width;
    const auto c1 = width * height * (1.0f - minOverlap) / (1.0f + minOverlap);
    const auto r1 = (b1 + std::sqrt(std::max(0.0f, b1 * b1 - 4.0f * a1 * c1))) / 2.0f;

    const auto a2 = 4.0f;
    const auto b2 = 2.0f * (height + width);
    const auto c2 = (1.0f - minOverlap) * width * height;
    const auto r2 = (b2 + std::sqrt(std::max(0.0f, b2 * b2 - 4.0f * a2 * c2))) / 2.0f;

    const auto a3 = 4.0f * minOverlap;
    const auto b3 = -2.0f * minOverlap * (height + width);
    const auto c3 = (minOverlap - 1.0f) * width * height;
    const auto r3 = (b3 + std::sqrt(std::max(0.0f, b3 * b3 - 4.0f * a3 * c3))) / 2.0f;

    return std::max(0, static_cast<int>(std::floor(std::min({ r1, r2, r3 }))));
}

CenterNetTargets CenterNetTargetBuilder::build(const GroundTruthVec& truth) const
{
    const auto area = width_ * height_;
    CenterNetTargets targets{
            PxCpuVector(classes_ * area, 0.0f),
            PxCpuVector(2 * area, 0.0f),
            PxCpuVector(2 * area, 0.0f),
            PxCpuVector(area, 0.0f)
    };

    for (const auto& gt: truth) {
        PX_CHECK(gt.classId >= 0 && gt.classId < classes_, "CenterNet ground-truth class is out of range.");
        if (!std::isfinite(gt.box.x()) || !std::isfinite(gt.box.y()) ||
            !std::isfinite(gt.box.w()) || !std::isfinite(gt.box.h()) ||
            gt.box.w() <= 0.0f || gt.box.h() <= 0.0f) {
            continue;
        }

        const auto centerX = std::clamp(gt.box.x() * width_, 0.0f, std::nextafter(float(width_), 0.0f));
        const auto centerY = std::clamp(gt.box.y() * height_, 0.0f, std::nextafter(float(height_), 0.0f));
        const auto cellX = static_cast<int>(centerX);
        const auto cellY = static_cast<int>(centerY);
        const auto index = cellY * width_ + cellX;
        ++targets.objects;

        drawGaussian(targets.heatmap, gt.classId, cellX, cellY,
                     std::max(1, gaussianRadius(gt.box.w() * width_, gt.box.h() * height_)));

        if (targets.mask[index] > 0.0f) {
            ++targets.collisions;
            const auto oldArea = targets.size[index] * targets.size[area + index];
            const auto newArea = (gt.box.w() * width_) * (gt.box.h() * height_);
            if (newArea <= oldArea) {
                continue;
            }
        }

        targets.size[index] = gt.box.w() * width_;
        targets.size[area + index] = gt.box.h() * height_;
        targets.offset[index] = centerX - cellX;
        targets.offset[area + index] = centerY - cellY;
        targets.mask[index] = 1.0f;
    }

    return targets;
}

void CenterNetTargetBuilder::drawGaussian(PxCpuVector& heatmap, int classId, int centerX, int centerY,
                                          int radius) const
{
    const auto diameter = 2 * radius + 1;
    const auto sigma = std::max(1.0f, diameter / 6.0f);
    const auto denominator = 2.0f * sigma * sigma;
    const auto classOffset = classId * width_ * height_;

    for (auto y = -radius; y <= radius; ++y) {
        const auto py = centerY + y;
        if (py < 0 || py >= height_) {
            continue;
        }
        for (auto x = -radius; x <= radius; ++x) {
            const auto px = centerX + x;
            if (px < 0 || px >= width_) {
                continue;
            }
            const auto value = std::exp(-(x * x + y * y) / denominator);
            const auto index = classOffset + py * width_ + px;
            heatmap[index] = std::max(heatmap[index], value);
        }
    }
}

} // namespace px
