/********************************************************************************
* Copyright 2020 Thomas A. Rieck, All Rights Reserved
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

#include "Utility.h"

#include <cmath>

namespace px {

static float
im2col_get_pixel(const float* im, int height, int width, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width)
        return 0;

    return im[col + width * (row + height * channel)];
}


void im2col_cpu(const float* im, int channels, int height, int width, int ksize, int stride, int pad,
                float* dataCol)
{
    int c, h, w;
    int heightCol = (height + 2 * pad - ksize) / stride + 1;
    int widthCol = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int wOffset = c % ksize;
        int hOffset = (c / ksize) % ksize;
        int cIm = c / ksize / ksize;
        for (h = 0; h < heightCol; ++h) {
            for (w = 0; w < widthCol; ++w) {
                int imRow = hOffset + h * stride;
                int imCol = wOffset + w * stride;
                int colIndex = (c * heightCol + h) * widthCol + w;
                dataCol[colIndex] = im2col_get_pixel(im, height, width, imRow, imCol, cIm, pad);
            }
        }
    }
}

void normalize_cpu(float* x, const float* mean, float* variance, int batch, int filters, int spatial)
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

void scale_bias(float* output, const float* scales, int batch, int n, int size)
{
    for (auto b = 0; b < batch; ++b) {
        for (auto i = 0; i < n; ++i) {
            for (auto j = 0; j < size; ++j) {
                output[(b * n + i) * size + j] *= scales[i];
            }
        }
    }
}

void add_bias(float* output, const float* biases, int batch, int n, int size)
{
    for (auto b = 0; b < batch; ++b) {
        for (auto i = 0; i < n; ++i) {
            for (auto j = 0; j < size; ++j) {
                output[(b * n + i) * size + j] += biases[i];
            }
        }
    }
}

void random_generate_cpu(float* ptr, std::size_t n, float a, float b)
{
    std::random_device device;
    std::mt19937 engine{ device() };
    std::uniform_real_distribution<float> dist{ a, b };

    auto gen = [&dist, &engine]() {
        return dist(engine);
    };

    std::generate(ptr, ptr + n, gen);
}

}   // px
