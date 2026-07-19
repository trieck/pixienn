#include <cuda_runtime.h>

#include "CenterNetKernels.cuh"
#include "CudaError.h"
#include "CudaUtils.cuh"

namespace px {

__global__ void centerNetActivateKernel(const float* input, float* output,
                                        std::size_t size, int outputs,
                                        int heatmapSize, float heatmapBias)
{
    const auto i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const auto withinBatch = i % outputs;
    const auto value = input[i] + (withinBatch < heatmapSize ? heatmapBias : 0.0f);
    output[i] = withinBatch < heatmapSize ? 1.0f / (1.0f + expf(-value)) : value;
}

__global__ void centerNetHeatmapLossKernel(const float* output, float* delta,
                                           const float* target, float* costs,
                                           std::size_t size, int outputs,
                                           int heatmapSize, float normalizer,
                                           float alpha, float beta)
{
    __shared__ float losses[CUDA_BLOCK_SIZE];
    const auto i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    auto loss = 0.0f;
    if (i < size) {
        const auto batch = i / heatmapSize;
        const auto withinHeatmap = i % heatmapSize;
        const auto outputIndex = batch * outputs + withinHeatmap;
        const auto prediction = fminf(fmaxf(output[outputIndex], 1e-4f), 1.0f - 1e-4f);
        const auto truth = target[i];
        if (truth >= 1.0f - 1e-6f) {
            const auto oneMinus = 1.0f - prediction;
            loss = -powf(oneMinus, alpha) * logf(prediction);
            delta[outputIndex] = normalizer *
                    (powf(oneMinus, alpha + 1.0f)
                     - alpha * prediction * powf(oneMinus, alpha) * logf(prediction));
        } else {
            const auto weight = powf(1.0f - truth, beta);
            loss = -weight * powf(prediction, alpha) * logf(1.0f - prediction);
            delta[outputIndex] = normalizer * weight *
                    (alpha * powf(prediction, alpha) * (1.0f - prediction)
                     * logf(1.0f - prediction) - powf(prediction, alpha + 1.0f));
        }
    }
    losses[threadIdx.x] = loss;
    __syncthreads();
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) losses[threadIdx.x] += losses[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(costs, losses[0]);
}

__global__ void centerNetRegressionLossKernel(const float* output, float* delta,
                                              const float* targetSize,
                                              const float* targetOffset,
                                              const float* targetMask,
                                              float* costs, std::size_t size,
                                              int classes, int area,
                                              float normalizer, float sizeWeight)
{
    __shared__ float sizeLosses[CUDA_BLOCK_SIZE];
    __shared__ float offsetLosses[CUDA_BLOCK_SIZE];
    const auto i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    auto sizeLoss = 0.0f;
    auto offsetLoss = 0.0f;
    if (i < size) {
        const auto batch = i / area;
        const auto cell = i % area;
        if (targetMask[i] != 0.0f) {
            const auto outputs = (classes + 4) * area;
            const auto heatmapSize = classes * area;
            const auto targetBase = batch * 2 * area;
            const auto outputBase = batch * outputs;
            for (auto channel = 0; channel < 2; ++channel) {
                const auto mapIndex = channel * area + cell;
                const auto sizeIndex = outputBase + heatmapSize + mapIndex;
                const auto offsetIndex = outputBase + heatmapSize + 2 * area + mapIndex;
                const auto sizeError = targetSize[targetBase + mapIndex] - output[sizeIndex];
                const auto offsetError = targetOffset[targetBase + mapIndex] - output[offsetIndex];
                sizeLoss += fabsf(sizeError);
                offsetLoss += fabsf(offsetError);
                delta[sizeIndex] = sizeWeight * normalizer *
                        ((sizeError > 0.0f) - (sizeError < 0.0f));
                delta[offsetIndex] = normalizer *
                        ((offsetError > 0.0f) - (offsetError < 0.0f));
            }
        }
    }
    sizeLosses[threadIdx.x] = sizeLoss;
    offsetLosses[threadIdx.x] = offsetLoss;
    __syncthreads();
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sizeLosses[threadIdx.x] += sizeLosses[threadIdx.x + stride];
            offsetLosses[threadIdx.x] += offsetLosses[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(costs + 1, sizeLosses[0]);
        atomicAdd(costs + 2, offsetLosses[0]);
    }
}

void centerNetActivateGpu(const float* input, float* output, int batch, int classes,
                          int area, float heatmapBias)
{
    const auto outputs = (classes + 4) * area;
    const auto size = static_cast<std::size_t>(batch) * outputs;
    centerNetActivateKernel<<<cudaGridsize(size), CUDA_BLOCK_SIZE>>>(
            input, output, size, outputs, classes * area, heatmapBias);
    PX_CUDA_CHECK_LAST();
}

void centerNetLossGpu(const float* output, float* delta,
                      const float* targetHeatmap, const float* targetSize,
                      const float* targetOffset, const float* targetMask,
                      float* costs, int batch, int classes, int area,
                      float normalizer, float focalAlpha, float focalBeta,
                      float sizeWeight)
{
    const auto heatmapElements = static_cast<std::size_t>(batch) * classes * area;
    centerNetHeatmapLossKernel<<<cudaGridsize(heatmapElements), CUDA_BLOCK_SIZE>>>(
            output, delta, targetHeatmap, costs, heatmapElements,
            (classes + 4) * area, classes * area, normalizer, focalAlpha, focalBeta);
    PX_CUDA_CHECK_LAST();
    const auto cells = static_cast<std::size_t>(batch) * area;
    centerNetRegressionLossKernel<<<cudaGridsize(cells), CUDA_BLOCK_SIZE>>>(
            output, delta, targetSize, targetOffset, targetMask, costs, cells,
            classes, area, normalizer, sizeWeight);
    PX_CUDA_CHECK_LAST();
}

} // namespace px
