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
#include "PxTensor.h"

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

static void col2ImAddPixel(float* im, int height, int width, int channels, int row, int col, int channel, int pad,
                           float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) {
        return;
    }

    im[col + width * (row + height * channel)] += val;
}

void im2ColCpu(const float* im, int channels, int height, int width, int ksize, int stride, int pad,
               float* dataCol)
{
    auto heightCol = (height + 2 * pad - ksize) / stride + 1;
    auto widthCol = (width + 2 * pad - ksize) / stride + 1;

    auto channelsCol = channels * ksize * ksize;
    for (auto c = 0; c < channelsCol; ++c) {
        auto wOffset = c % ksize;
        auto hOffset = (c / ksize) % ksize;
        auto cIm = c / ksize / ksize;
        for (auto h = 0; h < heightCol; ++h) {
            for (auto w = 0; w < widthCol; ++w) {
                auto imRow = hOffset + h * stride;
                auto imCol = wOffset + w * stride;
                auto colIndex = (c * heightCol + h) * widthCol + w;
                dataCol[colIndex] = im2ColGetPixel(im, height, width, imRow, imCol, cIm, pad);
            }
        }
    }
}

// Function uses casting from int to unsigned to compare if value of
// parameter "a" is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline static int isAGeZeroAndALtB(int a, int b)
{
    return (unsigned) (a) < (unsigned) (b);
}

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
void im2ColCpuExt(const float* im, const int channels, const int height, const int width, const int kernelH,
                  const int kernelW, const int padH, const int padW, const int strideH, const int strideW,
                  const int dilationH, const int dilationW, float* dataCol)
{
    const auto outputH = (height + 2 * padH - (dilationH * (kernelH - 1) + 1)) / strideH + 1;
    const auto outputW = (width + 2 * padW - (dilationW * (kernelW - 1) + 1)) / strideW + 1;
    const auto channelSize = height * width;
    int channel, kernelRow, kernelCol, outputRows, outputCol;

    for (channel = channels; channel--; im += channelSize) {
        for (kernelRow = 0; kernelRow < kernelH; kernelRow++) {
            for (kernelCol = 0; kernelCol < kernelW; kernelCol++) {
                auto inputRow = -padH + kernelRow * dilationH;
                for (outputRows = outputH; outputRows; outputRows--) {
                    if (!isAGeZeroAndALtB(inputRow, height)) {
                        for (outputCol = outputW; outputCol; outputCol--) {
                            *(dataCol++) = 0;
                        }
                    } else {
                        auto inputCol = -padW + kernelCol * dilationW;
                        for (outputCol = outputW; outputCol; outputCol--) {
                            if (isAGeZeroAndALtB(inputCol, width)) {
                                *(dataCol++) = im[inputRow * width + inputCol];
                            } else {
                                *(dataCol++) = 0;
                            }
                            inputCol += strideW;
                        }
                    }
                    inputRow += strideH;
                }
            }
        }
    }
}

void col2ImCpu(const float* dataCol, int channels, int height, int width, int ksize, int stride, int pad,
               float* dataIm)
{
    auto heightCol = (height + 2 * pad - ksize) / stride + 1;
    auto widthCol = (width + 2 * pad - ksize) / stride + 1;

    auto channelsCol = channels * ksize * ksize;
    for (auto c = 0; c < channelsCol; ++c) {
        auto wOffset = c % ksize;
        auto hOffset = (c / ksize) % ksize;
        auto cIm = c / ksize / ksize;
        for (auto h = 0; h < heightCol; ++h) {
            for (auto w = 0; w < widthCol; ++w) {
                auto imRow = hOffset + h * stride;
                auto imCol = wOffset + w * stride;
                auto colIndex = (c * heightCol + h) * widthCol + w;
                auto val = dataCol[colIndex];
                col2ImAddPixel(dataIm, height, width, channels,
                               imRow, imCol, cIm, pad, val);
            }
        }
    }
}

inline static int aGeZeroAndALtB(int a, int b)
{
    return (unsigned) (a) < (unsigned) (b);
}

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
void col2ImCpuExt(const float* dataCol, const int channels,
                  const int height, const int width, const int kernelH, const int kernelW,
                  const int padH, const int padW,
                  const int strideH, const int strideW,
                  const int dilationH, const int dilationW,
                  float* dataIm)
{
    memset(dataIm, 0, sizeof(float) * height * width * channels);

    const auto outputH = (height + 2 * padH - (dilationH * (kernelH - 1) + 1)) / strideH + 1;
    const auto outputW = (width + 2 * padW - (dilationW * (kernelW - 1) + 1)) / strideW + 1;
    const auto channelSize = height * width;

    int channel, kernelRow, kernelCol, outputRows, outputCol;

    for (channel = channels; channel--; dataIm += channelSize) {
        for (kernelRow = 0; kernelRow < kernelH; kernelRow++) {
            for (kernelCol = 0; kernelCol < kernelW; kernelCol++) {
                auto inputRow = -padH + kernelRow * dilationH;
                for (outputRows = outputH; outputRows; outputRows--) {
                    if (!aGeZeroAndALtB(inputRow, height)) {
                        dataCol += outputW;
                    } else {
                        auto inputCol = -padW + kernelCol * dilationW;
                        for (outputCol = outputW; outputCol; outputCol--) {
                            if (aGeZeroAndALtB(inputCol, width)) {
                                dataIm[inputRow * width + inputCol] += *dataCol;
                            }
                            dataCol++;
                            inputCol += strideW;
                        }
                    }
                    inputRow += strideH;
                }
            }
        }
    }
}

void addBias(float* output, const float* biases, int batch, int n, int size)
{
    for (auto b = 0; b < batch; ++b) {
        for (auto i = 0; i < n; ++i) {
            auto offset = b * n + i;
            cblas_saxpy(size, 1.0f, biases + i, 0, output + offset * size, 1);
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

void backwardScaleCpu(const float* xNorm, const float* delta, int batch, int n, int size, float* scaleUpdates)
{
    for (auto i = 0; i < n; ++i) {
        scaleUpdates[i] += cblas_sdot(size * batch, delta + size * i, 1, xNorm + size * i, 1);
    }
}

float sumArray(const float* a, int n)
{
    return cblas_sasum(n, a, 1);
}

float magArray(const float* a, int n)
{
    return cblas_snrm2(n, a, 1);
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
                auto index = j * filters * spatial + i * spatial + k;
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
                auto index = i * spatial + j * filters * spatial + k;
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
            cblas_sscal(size, scales[i], &output[(b * n + i) * size], 1);
        }
    }
}

void meanDeltaCpu(const float* delta, const float* variance, int batch, int filters, int spatial, float* meanDelta)
{
    for (auto i = 0; i < filters; ++i) {
        meanDelta[i] = 0;
        for (int j = 0; j < batch; ++j) {
            meanDelta[i] += cblas_sasum(spatial, &delta[j * filters * spatial + i * spatial], 1);
        }
        meanDelta[i] *= (-1. / std::sqrt(variance[i] + .00001f));
    }
}

void varianceDeltaCpu(const float* x, const float* delta, const float* mean, const float* variance, int batch,
                      int filters, int spatial, float* varianceDelta)
{
    for (auto i = 0; i < filters; ++i) {
        varianceDelta[i] = 0;
        for (auto j = 0; j < batch; ++j) {
            for (auto k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                varianceDelta[i] += delta[index] * (x[index] - mean[i]);
            }
        }
        varianceDelta[i] *= -.5f * std::pow(variance[i] + .00001f, -3.0f / 2.0f);
    }
}

void normalizeDeltaCpu(const float* x, const float* mean, const float* variance, const float* meanDelta,
                       const float* varianceDelta, int batch, int filters, int spatial, float* delta)
{
    const auto spatialSize = batch * spatial;

    for (auto j = 0; j < batch; ++j) {
        for (auto f = 0; f < filters; ++f) {
            for (auto k = 0; k < spatial; ++k) {
                auto index = j * filters * spatial + f * spatial + k;
                delta[index] = delta[index] * 1. / (sqrt(variance[f] + .00001f)) +
                               varianceDelta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) +
                               meanDelta[f] / (spatial * batch);
            }
        }
    }
}

float constrain(float min, float max, float a)
{
    if (a < min) {
        a = min;
    }

    if (a > max) {
        a = max;
    }

    return a;
}

void constrain(int n, float alpha, float* x, int incX)
{
    for (auto i = 0; i < n; ++i) {
        x[i * incX] = std::fminf(alpha, std::fmaxf(-alpha, x[i * incX]));
    }
}

void flatten(float* x, int size, int layers, int batch, bool forward)
{
    PxCpuVector swap(size * layers * batch, 0.0f);
    auto* pswap = swap.data();

    for (auto b = 0; b < batch; ++b) {
        for (auto c = 0; c < layers; ++c) {
            for (auto i = 0; i < size; ++i) {
                auto i1 = b * layers * size + c * size + i;
                auto i2 = b * layers * size + i * layers + c;
                if (forward) {
                    pswap[i2] = x[i1];
                } else {
                    pswap[i1] = x[i2];
                }
            }
        }
    }

    memcpy(x, pswap, size * layers * batch * sizeof(float));
}

}   // px
