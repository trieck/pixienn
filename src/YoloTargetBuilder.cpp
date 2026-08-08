#include "YoloTargetBuilder.h"

namespace px {

YoloTargetBuilder::YoloTargetBuilder(std::vector<int> anchors, std::vector<int> mask,
                                     int width, int height, int networkWidth, int networkHeight)
        : anchors_(std::move(anchors)), mask_(std::move(mask)), width_(width), height_(height),
          networkWidth_(networkWidth), networkHeight_(networkHeight)
{
    PX_CHECK(!mask_.empty() && anchors_.size() % 2 == 0, "Invalid YOLO anchors or mask.");
    PX_CHECK(width_ > 0 && height_ > 0 && networkWidth_ > 0 && networkHeight_ > 0,
             "Invalid YOLO target dimensions.");
}

int YoloTargetBuilder::maskIndex(int anchor) const
{
    const auto found = std::find(mask_.begin(), mask_.end(), anchor);
    return found == mask_.end() ? -1 : static_cast<int>(std::distance(mask_.begin(), found));
}

YoloAssignmentTargets YoloTargetBuilder::build(const GroundTruthVec& truths) const
{
    const auto area = width_ * height_;
    const auto slots = mask_.size() * area;
    YoloAssignmentTargets targets{
            PxCpuVectorT<int>(slots, -1), PxCpuVectorT<int>(slots, -1),
            PxCpuVector(slots * 4, 0.0f), 0
    };
    const auto anchorCount = static_cast<int>(anchors_.size() / 2);
    for (const auto& gt: truths) {
        auto bestIoU = std::numeric_limits<float>::lowest();
        auto bestAnchor = 0;
        const DarkBox shiftedTruth(0.0f, 0.0f, gt.box.w(), gt.box.h());
        for (auto anchor = 0; anchor < anchorCount; ++anchor) {
            const DarkBox candidate(0.0f, 0.0f,
                                    static_cast<float>(anchors_[2 * anchor]) / networkWidth_,
                                    static_cast<float>(anchors_[2 * anchor + 1]) / networkHeight_);
            const auto iou = candidate.iou(shiftedTruth);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestAnchor = anchor;
            }
        }
        auto maskSlot = maskIndex(bestAnchor);
        if (maskSlot < 0) continue;
        const auto x = std::clamp(static_cast<int>(gt.box.x() * width_), 0, width_ - 1);
        const auto y = std::clamp(static_cast<int>(gt.box.y() * height_), 0, height_ - 1);
        // Darknet writes the best-anchor target directly. If another truth
        // occupies the same cell and anchor, the later truth overwrites it.
        // Do not move collisions to a different anchor: that changes the
        // positive-gradient assignment relative to Darknet.
        const auto slot = maskSlot * area + y * width_ + x;
        targets.classes[slot] = gt.classId;
        targets.anchors[slot] = bestAnchor;
        targets.boxes[slot * 4] = gt.box.x();
        targets.boxes[slot * 4 + 1] = gt.box.y();
        targets.boxes[slot * 4 + 2] = gt.box.w();
        targets.boxes[slot * 4 + 3] = gt.box.h();
        ++targets.assigned;
    }
    return targets;
}

} // namespace px
