#pragma once

#include "GroundTruth.h"
#include "PxTensor.h"

namespace px {

struct YoloAssignmentTargets
{
    PxCpuVectorT<int> classes;
    PxCpuVectorT<int> anchors;
    PxCpuVector boxes;
    std::size_t assigned = 0;
};

class YoloTargetBuilder
{
public:
    YoloTargetBuilder(std::vector<int> anchors, std::vector<int> mask,
                      int width, int height, int networkWidth, int networkHeight);
    YoloAssignmentTargets build(const GroundTruthVec& truths) const;

private:
    int maskIndex(int anchor) const;
    std::vector<int> anchors_, mask_;
    int width_, height_, networkWidth_, networkHeight_;
};

} // namespace px
