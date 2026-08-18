#pragma once

#include "GroundTruth.h"
#include "PxTensor.h"

namespace px {

struct CenterNetTargets
{
    PxCpuVector heatmap;   // [classes, height, width]
    PxCpuVector size;      // [2, height, width], width/height in feature-map cells
    PxCpuVector offset;    // [2, height, width], fractional center offset
    PxCpuVector mask;      // [height, width]
    std::size_t objects = 0;
    std::size_t collisions = 0;
};

class CenterNetTargetBuilder
{
public:
    CenterNetTargetBuilder(int classes, int featureWidth, int featureHeight);

    CenterNetTargets build(const GroundTruthVec& truth) const;

    static int gaussianRadius(float width, float height, float minOverlap = 0.7f);

private:
    void drawGaussian(PxCpuVector& heatmap, int classId, int centerX, int centerY, int radius) const;

    int classes_;
    int width_;
    int height_;
};

} // namespace px
