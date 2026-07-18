#include <cuda_runtime.h>
#include <cfloat>

#include "CudaError.h"
#include "CudaUtils.cuh"
#include "MaxPoolKernels.cuh"

namespace px {

__global__ void maxPoolForwardKernel(const float* input, float* output, int* indexes, std::size_t size,
                                     int channels, int inputHeight, int inputWidth, int outputHeight,
                                     int outputWidth, int kernel, int stride, int padding)
{
    const auto outIndex = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (outIndex >= size) return;

    auto value = static_cast<int>(outIndex);
    const auto x = value % outputWidth;
    value /= outputWidth;
    const auto y = value % outputHeight;
    value /= outputHeight;
    const auto channel = value % channels;
    const auto batch = value / channels;
    const auto offset = -padding / 2;

    auto maxValue = -FLT_MAX;
    auto maxIndex = -1;
    for (auto ky = 0; ky < kernel; ++ky) {
        for (auto kx = 0; kx < kernel; ++kx) {
            const auto inputY = offset + y * stride + ky;
            const auto inputX = offset + x * stride + kx;
            if (inputY < 0 || inputY >= inputHeight || inputX < 0 || inputX >= inputWidth) continue;
            const auto inputIndex = inputX + inputWidth * (inputY + inputHeight * (channel + channels * batch));
            const auto candidate = input[inputIndex];
            if (maxIndex < 0 || candidate > maxValue) {
                maxValue = candidate;
                maxIndex = inputIndex;
            }
        }
    }
    output[outIndex] = maxValue;
    indexes[outIndex] = maxIndex;
}

__global__ void maxPoolBackwardKernel(const float* delta, const int* indexes, float* grad, int size)
{
    const auto i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < size && indexes[i] >= 0) atomicAdd(grad + indexes[i], delta[i]);
}

void maxPoolForwardGpu(const float* input, float* output, int* indexes, int batch, int channels, int inputHeight,
                       int inputWidth, int outputHeight, int outputWidth, int kernel, int stride, int padding)
{
    const std::size_t size = static_cast<std::size_t>(batch) * channels * outputHeight * outputWidth;
    maxPoolForwardKernel<<<cudaGridsize(size), CUDA_BLOCK_SIZE>>>(input, output, indexes, size, channels, inputHeight,
                                                                  inputWidth, outputHeight, outputWidth, kernel,
                                                                  stride, padding);
    PX_CUDA_CHECK_LAST();
}

void maxPoolBackwardGpu(const float* delta, const int* indexes, float* grad, int size)
{
    maxPoolBackwardKernel<<<cudaGridsize(size), CUDA_BLOCK_SIZE>>>(delta, indexes, grad, size);
    PX_CUDA_CHECK_LAST();
}

} // px
