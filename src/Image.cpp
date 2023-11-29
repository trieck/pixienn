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

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <tiffio.h>

#include "Common.h"
#include "Error.h"
#include "Image.h"
#include "TiffIO.h"

#define COLOR_RED(c)        ((uint8_t)(((c) & 0xFF0000) >> 16))
#define COLOR_GREEN(c)      ((uint8_t)(((c) & 0xFF00) >> 8))
#define COLOR_BLUE(c)       ((uint8_t)(((c) & 0xFF)))
#define MAKE_CV_COLOR(c)    CV_RGB(COLOR_RED(c), COLOR_GREEN(c), COLOR_BLUE(c))

using namespace cv;

namespace px {

static constexpr uint32_t tab20c[] = {
        0x3182bd,
        0x6baed6,
        0x9ecae1,
        0xc6dbef,
        0xe6550d,
        0xfd8d3c,
        0xfdae6b,
        0xfdd0a2,
        0x31a354,
        0x74c476,
        0xa1d99b,
        0xc7e9c0,
        0x756bb1,
        0x9e9ac8,
        0xbcbddc,
        0xdadaeb,
        0x636363,
        0x969696,
        0xbdbdbd,
        0xd9d9d9
};

static constexpr uint32_t Set1[] = {
        0xe41a1c,
        0x377eb8,
        0x4daf4a,
        0x984ea3,
        0xff7f00,
        0xffff33,
        0xa65628,
        0xf781bf,
        0x999999
};

static constexpr uint32_t Paired[] = {
        0xa6cee3,
        0x1f78b4,
        0xb2df8a,
        0x33a02c,
        0xfb9a99,
        0xe31a1c,
        0xfdbf6f,
        0xff7f00,
        0xcab2d6,
        0x6a3d9a,
        0xffff99,
        0xb15928
};

static constexpr uint32_t Accent[] = {
        0x7fc97f,
        0xbeaed4,
        0xfdc086,
        0xffff99,
        0x386cb0,
        0xf0027f,
        0xbf5b17,
        0x666666
};

static constexpr uint32_t tab10[] = {
        0x1f77b4,
        0xff7f0e,
        0x2ca02c,
        0xd62728,
        0x9467bd,
        0x8c564b,
        0xe377c2,
        0x7f7f7f,
        0xbcbd22,
        0x17becf
};

static constexpr uint32_t tab20[] = {
        0x1f77b4,
        0xaec7e8,
        0xff7f0e,
        0xffbb78,
        0x2ca02c,
        0x98df8a,
        0xd62728,
        0xff9896,
        0x9467bd,
        0xc5b0d5,
        0x8c564b,
        0xc49c94,
        0xe377c2,
        0xf7b6d2,
        0x7f7f7f,
        0xc7c7c7,
        0xbcbd22,
        0xdbdb8d,
        0x17becf,
        0x9edae5
};

static constexpr uint32_t crayola16[] = {
        0xed0a3f,
        0xff681f,
        0xff8833,
        0xffae42,
        0xfbe870,
        0xc5e17a,
        0x3aa655,
        0x0095b7,
        0x0066ff,
        0x6456b7,
        0x8359a3,
        0xbb3385,
        0xffa6c9,
        0xaf593e,
        0x000000,
        0xffffff
};

#define COLOR_ENTRY(i, cmap) cmap[i % (sizeof(cmap) / sizeof(cmap[0]))]

cv::Mat imread_tiff(const char* path)
{
    return readTIFF(path);
}

cv::Mat imread_8cu(const char* path)
{
    auto image = imread(path);

    if (image.depth() == CV_32F) {
        cv::Mat swapped;
        if (image.channels() == 3) {
            cv::cvtColor(image, swapped, CV_RGB2BGR);
        } else {
            swapped = image;
        }

        swapped *= 255.0f;  // assume in range 0..1

        Mat out;
        swapped.convertTo(out, CV_8UC(swapped.channels()));

        return out;
    }

    return image;
}

Mat imread(const char* path)
{
    boost::filesystem::path filePath(path);

    Mat image;

    auto extension = filePath.extension().string();
    if (extension == ".tiff" || extension == ".tif") {
        image = imread_tiff(path);
    } else {
        image = imread(path, IMREAD_UNCHANGED);
        PX_CHECK(!image.empty(), "Could not open image \"%s\".", path);
    }

    return image;
}

// normalize 8-bit RGB bands and convert to float
cv::Mat imnormalize(const cv::Mat& image)
{
    if (image.type() == CV_32FC3 || image.type() == CV_32FC1) {
        return image;
    }

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


// read an image and normalize
Mat imread_normalize(const char* path)
{
    auto image = imread(path);

    auto normal = imnormalize(image);

    return normal;
}

void imsave(const char* path, const cv::Mat& image)
{
    auto result = imwrite(path, image);
    PX_CHECK(result, "Could not save image \"%s\".", path);
}

// save an image in normalized float format as TIFF
void imsave_tiff(const char* path, const cv::Mat& image)
{
    Mat tiffImage(image);
    if (tiffImage.type() != CV_32FC3 && tiffImage.type() != CV_32FC1) {
        tiffImage = imnormalize(image);
    }

    auto* tif = TIFFOpen(path, "w");
    PX_CHECK(tif != nullptr, "Cannot open image \"%s\".", path);

    auto channels = tiffImage.channels();
    auto width = tiffImage.cols, height = tiffImage.rows;
    auto type = tiffImage.type();
    auto depth = CV_MAT_DEPTH(type);

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

    auto scanlineSize = TIFFScanlineSize(tif);
    AutoBuffer<uchar> buffer(scanlineSize + 32);

    for (auto y = 0; y < height; ++y) {
        memcpy(buffer, tiffImage.ptr(y), scanlineSize);
        PX_CHECK(TIFFWriteScanline(tif, buffer, y, 0) == 1, "Cannot write scan line.");
    }

    TIFFClose(tif);
}

Mat imletterbox(const Mat& image, int width, int height)
{
    int newWidth, newHeight;
    auto imageWidth = image.cols;
    auto imageHeight = image.rows;

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
    return { height, width, CV_32FC(channels), Scalar_<float>::all(value) };
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

PxCpuVector imvector(const cv::Mat& image)
{
    PX_CHECK(image.isContinuous(), "Non-continuous mat not supported.");

    int channels = image.channels();
    int width = image.cols;
    int height = image.rows;

    PxCpuVector vector(height * width * channels);
    auto* pimage = vector.data();

    // convert interleaved mat to planar image
    for (auto i = 0; i < height; ++i) {
        for (auto j = 0; j < width; ++j) {
            for (auto k = 0; k < channels; ++k) {
                pimage[k * width * height + i * width + j] = image.ptr<float>(i, j)[k];
            }
        }
    }

    return vector;
}

void imrect(cv::Mat& image, const cv::Rect& rect, uint32_t rgb, int thickness)
{
    rectangle(image, rect, MAKE_CV_COLOR(rgb), thickness, LINE_8, 0);
}

void imtext(Mat& image, const char* text, const cv::Point& ptOrg, uint32_t textColor, uint32_t bgColor, int thickness)
{
    constexpr auto fontFace = FONT_HERSHEY_SIMPLEX;
    constexpr auto fontScale = 0.618f;
    constexpr auto xbuffer = 4;
    constexpr auto ybuffer = 2;

    auto baseline = 0;
    auto textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);

    baseline += thickness;
    textSize.width += xbuffer;
    textSize.height += baseline;

    Point ptStart(ptOrg.x - (thickness / 2), ptOrg.y - (thickness / 2));
    Point ptEnd(ptStart.x + textSize.width, ptStart.y - textSize.height);
    Point ptText(ptStart.x + xbuffer, ptStart.y - baseline + thickness + ybuffer);

    rectangle(image, ptStart, ptEnd, MAKE_CV_COLOR(bgColor), FILLED);
    putText(image, text, ptText, fontFace, fontScale, MAKE_CV_COLOR(textColor), 1, LINE_AA);
}

uint32_t imgetcolor(uint32_t index)
{
    return COLOR_ENTRY(index, crayola16);
}

uint32_t imtextcolor(uint32_t bgColor)
{
    constexpr auto black = 0x000000;
    constexpr auto white = 0xFFFFFF;
    constexpr auto gamma = 2.2f;

    auto r = COLOR_RED(bgColor) / 255.0f;
    auto g = COLOR_GREEN(bgColor) / 255.0f;
    auto b = COLOR_BLUE(bgColor) / 255.0f;

    const auto luma = 0.2126f * std::pow(r, gamma) + 0.7152f * std::pow(g, gamma) + 0.0722f * std::pow(b, gamma);

    return luma > 0.314f ? black : white;
}

} // px

