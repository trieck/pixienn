/********************************************************************************
* Copyright 2020-2023 Thomas A. Rieck, All Rights Reserved
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
********************************************************************************/

#include <cblas.h>
#include <cmath>

#include "CpuUtil.h"

namespace px {

static float
im2ColGetPixel(const float* im, int height, int width, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) {
        return 0;
    }

    return im[col + width * (row + height * channel)];
}

void im2ColCpu(const float* im, int channels, int height, int width, int ksize, int stride, int pad,
               float* dataCol)
{
    int c, h, w;
    int heightCol = (height + 2 * pad - ksize) / stride + 1;
    int widthCol = (width + 2 * pad - ksize) / stride + 1;

    int channelsCol = channels * ksize * ksize;
    for (c = 0; c < channelsCol; ++c) {
        int wOffset = c % ksize;
        int hOffset = (c / ksize) % ksize;
        int cIm = c / ksize / ksize;
        for (h = 0; h < heightCol; ++h) {
            for (w = 0; w < widthCol; ++w) {
                int imRow = hOffset + h * stride;
                int imCol = wOffset + w * stride;
                int colIndex = (c * heightCol + h) * widthCol + w;
                dataCol[colIndex] = im2ColGetPixel(im, height, width, imRow, imCol, cIm, pad);
            }
        }
    }
}

void addBias(float* output, const float* biases, int batch, int n, int size)
{
    for (auto b = 0; b < batch; ++b) {
        for (auto i = 0; i < n; ++i) {
            auto offset = b * n + i;
            cblas_saxpy(size, 1.0f, biases + i, 0, output + (b * n + i) * size, 1);
        }
    }
}

void backwardBias(float* biasUpdates, const float* delta, int batch, int n, int size)
{
    for (auto b = 0; b < batch; ++b) {
        for (auto i = 0; i < n; ++i) {
            biasUpdates[i] += sumArray(delta + size * (i + b * n), size);
        }
    }
}

float sumArray(const float* a, int n)
{
    float sum = 0;

    for (auto i = 0; i < n; ++i) {
        sum += a[i];
    }

    return sum;
}

float magArray(const float* a, int n)
{
    float sum = 0;

    for (auto i = 0; i < n; ++i) {
        sum += a[i] * a[i];
    }

    return sqrt(sum);
}

void randomCpu(float* ptr, std::size_t n, float a, float b)
{
    std::random_device device;
    std::mt19937 engine{ device() };
    std::uniform_real_distribution<float> dist{ a, b };

    auto gen = [&dist, &engine]() {
        return dist(engine);
    };

    std::generate(ptr, ptr + n, gen);
}

void meanCpu(const float* x, int batch, int filters, int spatial, float* mean)
{
    auto scale = 1.0f / (batch * spatial);

    for (auto i = 0; i < filters; ++i) {
        mean[i] = 0;
        for (auto j = 0; j < batch; ++j) {
            for (auto k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void varianceCpu(const float* x, float* mean, int batch, int filters, int spatial, float* variance)
{
    auto scale = 1.0f / (batch * spatial - 1);

    for (auto i = 0; i < filters; ++i) {
        variance[i] = 0;
        for (auto j = 0; j < batch; ++j) {
            for (auto k = 0; k < spatial; ++k) {
                int index = i * spatial + j * filters * spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalizeCpu(float* x, const float* mean, const float* variance, int batch, int filters, int spatial)
{
    for (auto b = 0; b < batch; ++b) {
        for (auto f = 0; f < filters; ++f) {
            for (auto i = 0; i < spatial; ++i) {
                auto index = b * filters * spatial + f * spatial + i;
                x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void scaleBias(float* output, const float* scales, int batch, int n, int size)
{
    for (auto b = 0; b < batch; ++b) {
        for (auto i = 0; i < n; ++i) {
            for (auto j = 0; j < size; ++j) {
                output[(b * n + i) * size + j] *= scales[i];
            }
        }
    }
}


}   // px
