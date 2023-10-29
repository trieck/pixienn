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

#include "Common.h"
#include "Error.h"
#include "Image.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#include "SHA1.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <tiffio.h>
#include <xtensor/xadapt.hpp>

using namespace cv;

namespace px {

Mat imread(const char* path)
{
    Mat image = imread(path, IMREAD_UNCHANGED);
    PX_CHECK(!image.empty(), "Could not open image \"%s\".", path);

    Mat swapped;
    if (image.channels() == 3) {
        cv::cvtColor(image, swapped, CV_BGR2RGB);
    } else {
        swapped = image;
    }

    // convert to float and normalize
    Mat out;
    swapped.convertTo(out, CV_32FC(swapped.channels()));

    out /= 255.0f;

    return out;
}

cv::Mat imread_stb(const char* path)
{
    int w, h, c;
    unsigned char* data = stbi_load(path, &w, &h, &c, 3);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", path, stbi_failure_reason());
        exit(0);
    }

    cv::Mat mat(h, w, CV_32FC3);
    for (auto i = 0; i < w; ++i) {
        for (auto j = 0; j < h; ++j) {
            for (auto k = 0; k < c; ++k) {
                int src_index = k + c * i + c * w * j;
                auto value = (float) data[src_index] / 255.0f;
                imset(mat, i, j, k, value);
            }
        }
    }

    free(data);

    auto result = sha1(mat.data, h * w * c * sizeof(float));
    std::cout << "imread_stb RAW image:  " << result << std::endl;

    return mat;
}

void imsave(const char* path, const cv::Mat& image)
{
    auto* tif = TIFFOpen(path, "w");
    PX_CHECK(tif != nullptr, "Cannot open image \"%s\".", path);

    int channels = image.channels();
    int width = image.cols, height = image.rows;
    int type = image.type();
    int depth = CV_MAT_DEPTH(type);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);

    size_t fileStep = (width * channels * 32) / 8;

    auto rowsPerStrip = (int) ((1 << 13) / fileStep);
    rowsPerStrip = std::max(1, std::min(height, rowsPerStrip));

    auto colorspace = channels > 1 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;

    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, colorspace);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, channels);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rowsPerStrip);

    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, depth >= CV_32F ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT);
    //TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);

    auto scanlineSize = TIFFScanlineSize(tif);
    AutoBuffer<uchar> buffer(scanlineSize + 32);

    for (auto y = 0; y < height; ++y) {
        memcpy(buffer, image.ptr(y), scanlineSize);
        PX_CHECK(TIFFWriteScanline(tif, buffer, y, 0) == 1, "Cannot write scan line.");
    }

    TIFFClose(tif);
}

Mat imletterbox(const Mat& image, int width, int height)
{
    int newWidth, newHeight;
    int imageWidth = image.cols;
    int imageHeight = image.rows;

    if (((float) width / imageWidth) < ((float) height / imageHeight)) {
        newWidth = width;
        newHeight = (imageHeight * width) / imageWidth;
    } else {
        newHeight = height;
        newWidth = (imageWidth * height) / imageHeight;
    }

    Mat resized;
    resize(image, resized, { newWidth, newHeight });

    auto boxed = immake(height, width, image.channels(), 0.5f);

    auto x = (width - newWidth) / 2;
    auto y = (height - newHeight) / 2;

    resized.copyTo(boxed(Rect(x, y, resized.cols, resized.rows)));

    return boxed;
}

Mat imchannel(const Mat& image, int c)
{
    PX_CHECK(c < image.channels(), "Channel out of bounds.");

    Mat channel;
    extractChannel(image, channel, c);

    return channel;
}

float imget(const Mat& image, int x, int y, int c)
{
    PX_CHECK(image.rows > y, "Row out of bounds");
    PX_CHECK(image.cols > x, "Column out of bounds");
    PX_CHECK(image.channels() > c, "Channel out of bounds");

    return image.ptr<float>(y, x)[c];
}

float imgetextend(const Mat& image, int x, int y, int c)
{
    if (x < 0 || x >= image.cols || y < 0 || y >= image.rows || c < 0 || c >= image.channels()) return 0;

    return imget(image, x, y, c);
}

void imset(Mat& image, int x, int y, int c, float value)
{
    PX_CHECK(image.rows > y, "Row out of bounds.");
    PX_CHECK(image.cols > x, "Column out of bounds.");
    PX_CHECK(image.channels() > c, "Channel out of bounds.");

    image.ptr<float>(y, x)[c] = value;
}

void imadd(Mat& image, int x, int y, int c, float value)
{
    PX_CHECK(image.rows > y, "Row out of bounds.");
    PX_CHECK(image.cols > x, "Column out of bounds.");
    PX_CHECK(image.channels() > c, "Channel out of bounds.");

    image.ptr<float>(y, x)[c] += value;
}

Mat imrandom(int height, int width, int channels)
{
    Mat image = immake(height, width, channels);

    randu(image, Scalar::all(0.f), Scalar::all(1.f));

    return image;
}

Mat immake(int height, int width, int channels, float value)
{
    return Mat(height, width, CV_32FC(channels), Scalar_<float>::all(value));
}

void imconvolve(const Mat& image, const Mat& kernel, int stride, int channel, Mat& out)
{
    PX_CHECK(image.channels() == kernel.channels(), "Image and kernel have different number of channels.");

    imzero(out, channel);

    for (auto i = 0; i < image.channels(); ++i) {
        im2dconvolve(image, i, kernel, i, stride, out, channel);
    }
}

void im2dconvolve(const Mat& image, int imChannel, const Mat& kernel, int kernelChannel, int stride,
                  Mat& out, int outChannel)
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

void imzero(const Mat& image, int c)
{
    PX_CHECK(c < image.channels(), "Channel out of bounds.");

    auto channel = imchannel(image, c);
    channel.setTo(Scalar::all(0.0f));
}

xt::xarray<float> imarray(const cv::Mat& image)
{
    PX_CHECK(image.isContinuous(), "Non-continuous mat not supported.");

    int channels = image.channels();
    int width = image.cols;
    int height = image.rows;
    auto pimage = std::make_unique<float[]>(height * width * channels);

    // convert interleaved mat to planar image
    for (auto i = 0; i < height; ++i) {
        for (auto j = 0; j < width; ++j) {
            for (auto k = 0; k < channels; ++k) {
                pimage[k * width * height + i * width + j] = image.ptr<float>(i, j)[k];
            }
        }
    }

    std::vector<int> shape({ height, width, channels });

    auto array = xt::adapt(pimage.release(), height * width * channels, xt::acquire_ownership(), shape);

    return array;
}

} // px

