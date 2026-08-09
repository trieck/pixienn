#pragma once

namespace px {

constexpr int YOLO_STATS_SIZE = 12;

void yoloActivateGpu(const float* input, float* output, int batch, int masks,
                     int classes, int area);

void yoloLossGpu(const float* output, float* delta, const float* truths,
                 const int* truthCounts, int maxTruth, const int* assignedClasses,
                 const int* assignedAnchors, const float* assignedBoxes,
                 const int* masks, const float* anchors, float* stats, float* cost,
                 int batch, int maskCount, int anchorCount, int classes, int width,
                 int height, int networkWidth, int networkHeight, float ignoreThreshold,
                 float truthThreshold, float coordScale, float objectScale,
                 float noObjectScale, float objectNormalizer, float classScale,
                 float classNegativeScale);

} // namespace px
