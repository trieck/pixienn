#pragma once

namespace px {

void centerNetActivateGpu(const float* input, float* output, int batch, int classes,
                          int area, float heatmapBias);

void centerNetLossGpu(const float* output, float* delta,
                      const float* targetHeatmap, const float* targetSize,
                      const float* targetOffset, const float* targetMask,
                      float* costs, int batch, int classes, int area,
                      float normalizer, float focalAlpha, float focalBeta,
                      float sizeWeight);

} // namespace px
