#include <cuda_runtime.h>
#include <cfloat>

#include "CudaError.h"
#include "CudaUtils.cuh"
#include "YoloKernels.cuh"

namespace px {
namespace {

__device__ float overlap(float x1, float w1, float x2, float w2)
{
    return fminf(x1 + w1 / 2.0f, x2 + w2 / 2.0f)
           - fmaxf(x1 - w1 / 2.0f, x2 - w2 / 2.0f);
}

__device__ float boxIou(const float* a, const float* b)
{
    const auto w = overlap(a[0], a[2], b[0], b[2]);
    const auto h = overlap(a[1], a[3], b[1], b[3]);
    const auto intersection = w > 0.0f && h > 0.0f ? w * h : 0.0f;
    const auto unionArea = fmaxf(0.0f, a[2] * a[3] + b[2] * b[3] - intersection);
    return unionArea > 0.0f ? intersection / unionArea : 0.0f;
}

__device__ int entryIndex(int batch, int anchor, int cell, int entry,
                          int masks, int classes, int area)
{
    return batch * masks * (classes + 5) * area
           + anchor * (classes + 5) * area + entry * area + cell;
}

__device__ void decodeBox(const float* output, float* box, int batch, int maskSlot,
                          int cell, int anchor, const float* anchors, int masks,
                          int classes, int width, int height, int netWidth, int netHeight)
{
    const auto area = width * height;
    const auto x = cell % width;
    const auto y = cell / width;
    const auto index = entryIndex(batch, maskSlot, cell, 0, masks, classes, area);
    box[0] = (x + output[index]) / width;
    box[1] = (y + output[index + area]) / height;
    box[2] = expf(output[index + 2 * area]) * anchors[2 * anchor] / netWidth;
    box[3] = expf(output[index + 3 * area]) * anchors[2 * anchor + 1] / netHeight;
}

__device__ void boxDelta(const float* truth, const float* output, float* delta,
                         int index, int area, int x, int y, int width, int height,
                         int netWidth, int netHeight, const float* anchors, int anchor,
                         float coordScale)
{
    const auto scale = coordScale * (2.0f - truth[2] * truth[3]);
    const auto tx = truth[0] * width - x;
    const auto ty = truth[1] * height - y;
    const auto tw = logf(fmaxf(1e-9f, truth[2] * netWidth / anchors[2 * anchor]));
    const auto th = logf(fmaxf(1e-9f, truth[3] * netHeight / anchors[2 * anchor + 1]));
    delta[index] = scale * (tx - output[index]);
    delta[index + area] = scale * (ty - output[index + area]);
    delta[index + 2 * area] = scale * (tw - output[index + 2 * area]);
    delta[index + 3 * area] = scale * (th - output[index + 3 * area]);
}

__global__ void activateKernel(const float* input, float* output, std::size_t size,
                               int classes, int area)
{
    const auto i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const auto entry = (i / area) % (classes + 5);
    const auto value = input[i];
    output[i] = entry < 2 || entry >= 4 ? 1.0f / (1.0f + expf(-value)) : value;
}

__global__ void regionLossKernel(const float* output, float* delta, const float* truths,
                                 const int* truthCounts, int maxTruth, const int* masks,
                                 const float* anchors, float* stats, int slots, int maskCount,
                                 int classes, int width, int height, int netWidth, int netHeight,
                                 float ignoreThreshold, float truthThreshold, float coordScale,
                                 float objectScale, float noObjectScale, float classScale)
{
    const auto slot = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (slot >= slots) return;
    const auto area = width * height;
    const auto batch = slot / (maskCount * area);
    const auto withinBatch = slot % (maskCount * area);
    const auto maskSlot = withinBatch / area;
    const auto cell = withinBatch % area;
    float prediction[4];
    decodeBox(output, prediction, batch, maskSlot, cell, masks[maskSlot], anchors,
              maskCount, classes, width, height, netWidth, netHeight);
    auto bestIou = -FLT_MAX;
    auto bestTruth = -1;
    for (auto t = 0; t < truthCounts[batch]; ++t) {
        const auto truthIndex = (batch * maxTruth + t) * 5;
        const auto iou = boxIou(prediction, truths + truthIndex);
        // Darknet's ignore rule is based on the best box IoU only.  Do not
        // gate this search on class confidence: early in training, a box can
        // overlap a truth while its class sigmoid is still below 0.25.
        if (iou > bestIou) {
            bestIou = iou;
            bestTruth = truthIndex;
        }
    }
    const auto objectIndex = entryIndex(batch, maskSlot, cell, 4, maskCount, classes, area);
    const auto objectness = output[objectIndex];
    const auto positiveScale = objectScale;
    const auto negativeScale = noObjectScale;
    atomicAdd(stats + 5, objectness);
    if (bestTruth < 0 || bestIou < ignoreThreshold) {
        delta[objectIndex] = -negativeScale * objectness;
    }
    // A truth-threshold match must override the no-object penalty.  With the
    // usual truth_thresh < ignore_thresh configuration, else-if would mark
    // every prediction in that useful IoU interval as background.
    if (bestTruth >= 0 && bestIou > truthThreshold) {
        delta[objectIndex] = positiveScale * (1.0f - objectness);
        const auto classId = static_cast<int>(truths[bestTruth + 4]);
        const auto classIndex = entryIndex(batch, maskSlot, cell, 5, maskCount, classes, area);
        for (auto c = 0; c < classes; ++c) {
            const auto truthClass = c == classId ? 1.0f : 0.0f;
            delta[classIndex + c * area] = classScale * (truthClass - output[classIndex + c * area]);
        }
        const auto boxIndex = entryIndex(batch, maskSlot, cell, 0, maskCount, classes, area);
        boxDelta(truths + bestTruth, output, delta, boxIndex, area, cell % width,
                 cell / width, width, height, netWidth, netHeight, anchors,
                 masks[maskSlot], coordScale);
    }
}

__global__ void objectLossKernel(const float* output, float* delta,
                                 const int* assignedClasses, const int* assignedAnchors,
                                 const float* assignedBoxes, const float* anchors, float* stats,
                                 int slots, int maskCount, int classes, int width, int height,
                                 int netWidth, int netHeight, float coordScale,
                                 float objectScale, float classScale)
{
    const auto slot = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (slot >= slots || assignedClasses[slot] < 0) return;
    const auto area = width * height;
    const auto batch = slot / (maskCount * area);
    const auto withinBatch = slot % (maskCount * area);
    const auto maskSlot = withinBatch / area;
    const auto cell = withinBatch % area;
    const auto* truth = assignedBoxes + slot * 4;
    const auto anchor = assignedAnchors[slot];
    float prediction[4];
    decodeBox(output, prediction, batch, maskSlot, cell, anchor, anchors,
              maskCount, classes, width, height, netWidth, netHeight);
    const auto iou = boxIou(prediction, truth);
    const auto boxIndex = entryIndex(batch, maskSlot, cell, 0, maskCount, classes, area);
    boxDelta(truth, output, delta, boxIndex, area, cell % width, cell / width,
             width, height, netWidth, netHeight, anchors, anchor, coordScale);
    const auto objectIndex = entryIndex(batch, maskSlot, cell, 4, maskCount, classes, area);
    atomicAdd(stats + 4, output[objectIndex]);
    delta[objectIndex] = objectScale * (1.0f - output[objectIndex]);
    const auto classIndex = entryIndex(batch, maskSlot, cell, 5, maskCount, classes, area);
    const auto classId = assignedClasses[slot];
    for (auto c = 0; c < classes; ++c) {
        const auto truthClass = c == classId ? 1.0f : 0.0f;
        delta[classIndex + c * area] = classScale * (truthClass - output[classIndex + c * area]);
        if (truthClass != 0.0f) atomicAdd(stats + 3, fminf(1.0f, output[classIndex + c * area]));
    }
    atomicAdd(stats, iou);
    if (iou > 0.5f) atomicAdd(stats + 1, 1.0f);
    if (iou > 0.75f) atomicAdd(stats + 2, 1.0f);
    atomicAdd(stats + 6, 1.0f);
    atomicAdd(stats + 7, 1.0f);
}

__global__ void squaredNormKernel(const float* delta, float* cost, std::size_t size)
{
    __shared__ float sums[CUDA_BLOCK_SIZE];
    const auto i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    sums[threadIdx.x] = i < size ? delta[i] * delta[i] : 0.0f;
    __syncthreads();
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(cost, sums[0]);
}

__global__ void lossComponentsKernel(const float* delta, float* stats, int slots,
                                     int masks, int classes, int area)
{
    const auto slot = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (slot >= slots) return;

    const auto batch = slot / (masks * area);
    const auto withinBatch = slot % (masks * area);
    const auto maskSlot = withinBatch / area;
    const auto cell = withinBatch % area;
    const auto stride = area;
    const auto index = batch * masks * (classes + 5) * stride
                       + maskSlot * (classes + 5) * stride + cell;

    auto boxLoss = 0.0f;
    for (auto component = 0; component < 4; ++component) {
        const auto value = delta[index + component * stride];
        boxLoss += value * value;
    }
    atomicAdd(stats + 8, boxLoss);

    const auto object = delta[index + 4 * stride];
    atomicAdd(stats + (object >= 0.0f ? 9 : 10), object * object);

    auto classLoss = 0.0f;
    for (auto c = 0; c < classes; ++c) {
        const auto value = delta[index + (5 + c) * stride];
        classLoss += value * value;
    }
    atomicAdd(stats + 11, classLoss);
}

} // namespace

void yoloActivateGpu(const float* input, float* output, int batch, int masks, int classes, int area)
{
    const auto size = static_cast<std::size_t>(batch) * masks * (classes + 5) * area;
    activateKernel<<<cudaGridsize(size), CUDA_BLOCK_SIZE>>>(input, output, size, classes, area);
    PX_CUDA_CHECK_LAST();
}

void yoloLossGpu(const float* output, float* delta, const float* truths,
                 const int* truthCounts, int maxTruth, const int* assignedClasses,
                 const int* assignedAnchors, const float* assignedBoxes,
                 const int* masks, const float* anchors, float* stats, float* cost,
                 int batch, int maskCount, int anchorCount, int classes, int width,
                 int height, int networkWidth, int networkHeight, float ignoreThreshold,
                 float truthThreshold, float coordScale, float objectScale,
                 float noObjectScale, float classScale)
{
    (void) anchorCount;
    const auto slots = batch * maskCount * width * height;
    regionLossKernel<<<cudaGridsize(slots), CUDA_BLOCK_SIZE>>>(
            output, delta, truths, truthCounts, maxTruth, masks, anchors, stats, slots,
            maskCount, classes, width, height, networkWidth, networkHeight,
            ignoreThreshold, truthThreshold, coordScale, objectScale, noObjectScale,
            classScale);
    PX_CUDA_CHECK_LAST();
    objectLossKernel<<<cudaGridsize(slots), CUDA_BLOCK_SIZE>>>(
            output, delta, assignedClasses, assignedAnchors, assignedBoxes, anchors, stats,
            slots, maskCount, classes, width, height, networkWidth, networkHeight,
            coordScale, objectScale, classScale);
    PX_CUDA_CHECK_LAST();
    const auto outputs = static_cast<std::size_t>(batch) * maskCount * (classes + 5) * width * height;
    squaredNormKernel<<<cudaGridsize(outputs), CUDA_BLOCK_SIZE>>>(delta, cost, outputs);
    PX_CUDA_CHECK_LAST();
    const auto slotsForComponents = batch * maskCount * width * height;
    lossComponentsKernel<<<cudaGridsize(slotsForComponents), CUDA_BLOCK_SIZE>>>(
            delta, stats, slotsForComponents, maskCount, classes, width * height);
    PX_CUDA_CHECK_LAST();
}

} // namespace px
