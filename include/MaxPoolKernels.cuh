#pragma once

namespace px {

void maxPoolForwardGpu(const float* input, float* output, int* indexes, int batch, int channels, int inputHeight,
                       int inputWidth, int outputHeight, int outputWidth, int kernel, int stride, int padding);
void maxPoolBackwardGpu(const float* delta, const int* indexes, float* grad, int size);

} // px
