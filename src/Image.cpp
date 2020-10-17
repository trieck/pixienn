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

#include "common.h"
#include "Image.h"
#include "Error.h"

#include <opencv2/imgcodecs.hpp>

PX_BEGIN

cv::Mat imread(const char* path)
{
    cv::Mat image = cv::imread(path);
    PX_CHECK(!image.empty(), "Could not open image \"%s\".", path);

    // convert to float and normalize
    cv::Mat out;

    image.convertTo(out, CV_MAKETYPE(CV_32F, image.channels()));

    out /= 255.0f;

    return out;
}

cv::Mat imchannel(const cv::Mat& image, int c)
{
    PX_CHECK(c < image.channels(), "Channel out of bounds.");

    cv::Mat channel;
    cv::extractChannel(image, channel, c);

    return channel;
}

float imget(const cv::Mat& image, int x, int y, int c)
{
    PX_CHECK(image.rows > y, "Row out of bounds");
    PX_CHECK(image.cols > x, "Column out of bounds");
    PX_CHECK(image.channels() > c, "Channel out of bounds");

    return image.ptr<float>(y, x)[c];
}

float imgetextend(const cv::Mat& image, int x, int y, int c)
{
    if (x < 0 || x >= image.cols || y < 0 || y >= image.rows || c < 0 || c >= image.channels()) return 0;

    return imget(image, x, y, c);
}

void imset(cv::Mat& image, int x, int y, int c, float value)
{
    PX_CHECK(image.rows > y, "Row out of bounds");
    PX_CHECK(image.cols > x, "Column out of bounds");
    PX_CHECK(image.channels() > c, "Channel out of bounds");

    image.ptr<float>(y, x)[c] = value;
}

void imadd(cv::Mat& image, int x, int y, int c, float value)
{
    PX_CHECK(image.rows > y, "Row out of bounds");
    PX_CHECK(image.cols > x, "Column out of bounds");
    PX_CHECK(image.channels() > c, "Channel out of bounds");

    image.ptr<float>(y, x)[c] += value;
}

cv::Mat imrandom(int height, int width, int channels)
{
    cv::Mat image = immake(height, width, channels);

    cv::randu(image, cv::Scalar(0.f), cv::Scalar(1.f));

    return image;
}

cv::Mat immake(int height, int width, int channels)
{
    return cv::Mat(height, width, CV_MAKETYPE(CV_32F, channels), cv::Scalar(0.0f));
}

void imconvolve(const cv::Mat& image, const cv::Mat& kernel, int stride, int channel, cv::Mat& out)
{
    PX_CHECK(image.channels() == kernel.channels(), "Image and kernel have different number of channels.");

    imzero(out, channel);

    for (auto i = 0; i < image.channels(); ++i) {
        im2dconvolve(image, i, kernel, i, stride, out, channel);
    }
}

void im2dconvolve(const cv::Mat& image, int imChannel, const cv::Mat& kernel, int kernelChannel, int stride,
                  cv::Mat& out, int outChannel)
{
    PX_CHECK(stride > 0, "Stride must be greater than zero.");

    for (auto y = 0; y < image.rows; y += stride) {
        for (auto x = 0; x < image.cols; x += stride) {
            float sum = 0;
            for (auto i = 0; i < kernel.rows; ++i) {
                for (auto j = 0; j < kernel.cols; ++j) {
                    auto a = imget(kernel, i, j, kernelChannel);
                    auto b = imgetextend(image, x + i - kernel.cols / 2, y + j - kernel.rows / 2, imChannel);
                    sum += a * b;
                }
            }

            imadd(out, x / stride, y / stride, outChannel, sum);
        }
    }
}

void imzero(const cv::Mat& image, int c)
{
    PX_CHECK(c < image.channels(), "Channel out of bounds.");

    auto channel = imchannel(image, c);
    channel.setTo(cv::Scalar::all(0.0f));
}

PX_END
