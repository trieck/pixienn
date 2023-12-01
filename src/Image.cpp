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
#include <cairo/cairo.h>
#include <pango/pangocairo.h>

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

#define COLOR_REDF(c)       (COLOR_RED(c) / 255.0f)
#define COLOR_GREENF(c)     (COLOR_GREEN(c) / 255.0f)
#define COLOR_BLUEF(c)      (COLOR_BLUE(c) / 255.0f)

#define MAKE_CV_COLOR(c)    CV_RGB(COLOR_RED(c), COLOR_GREEN(c), COLOR_BLUE(c))

using namespace cv;

namespace px {

cv::Mat imread_tiff(const char* path)
{
    return readTIFF(path);
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

void imrect(cv::Mat& image, const cv::Rect& rect, uint32_t rgb, int thickness, int lineType)
{
    rectangle(image, rect, MAKE_CV_COLOR(rgb), thickness, lineType, 0);
}

void imtabbed_rect(cv::Mat& image, const cv::Point& pt1, const cv::Point& pt2, uint32_t rgb, int thickness,
                   int lineType, int cornerRadius)
{
    auto lineColor = MAKE_CV_COLOR(rgb);

    cv::Rect rect(pt1, pt2);
    auto p1 = rect.tl();
    auto p2 = Point(rect.br().x, rect.tl().y);
    auto p3 = rect.br();
    auto p4 = Point(rect.tl().x, rect.br().y);

    auto q1 = Point(p1.x + cornerRadius, p1.y);
    auto q2 = Point(p2.x - cornerRadius, p2.y);
    auto q3 = Point(p2.x, p2.y + cornerRadius);
    auto q4 = Point(p3.x, p3.y);
    auto q5 = Point(p4.x, p4.y);
    auto q6 = Point(p3.x, p3.y);
    auto q7 = Point(p1.x, p1.y + cornerRadius);
    auto q8 = Point(p4.x, p4.y);

    line(image, q1, q2, lineColor, thickness, LINE_AA);
    line(image, q3, q4, lineColor, thickness, LINE_AA);
    line(image, q5, q6, lineColor, thickness, LINE_AA);
    line(image, q7, q8, lineColor, thickness, LINE_AA);

    ellipse(image, p1 + Point(cornerRadius, cornerRadius), Size(cornerRadius, cornerRadius), 180.0, 0, 90, lineColor,
            thickness, lineType);
    ellipse(image, p2 + Point(-cornerRadius, cornerRadius), Size(cornerRadius, cornerRadius), 270.0, 0, 90, lineColor,
            thickness, lineType);

    if (lineType == FILLED) {
        cv::fillConvexPoly(image, std::vector<Point>{ q1, q2, q3, q4, q5, q6, q7, q8 }, lineColor, LINE_AA);
    }
}

void imtabbed_text(cv::Mat& image, const char* text, const cv::Point& ptOrg, uint32_t textColor, uint32_t bgColor,
                   int thickness)
{
    constexpr auto xpad = 4;

    // create cairo surface and context
    auto* surface = cairo_image_surface_create_for_data(
            image.data, CAIRO_FORMAT_ARGB32, image.cols, image.rows, image.step);
    auto* cr = cairo_create(surface);

    // create pango layout
    auto* layout = pango_cairo_create_layout(cr);

    // Set font description
    auto* desc = pango_font_description_from_string("Sans 8");
    pango_layout_set_font_description(layout, desc);
    pango_font_description_free(desc);

    // set layout text
    pango_layout_set_text(layout, text, -1);

    // get pixel size
    cv::Size textSize;
    pango_layout_get_pixel_size(layout, &textSize.width, &textSize.height);
    textSize.width += xpad;

    Point ptStart(ptOrg.x - (thickness / 2) + 1, ptOrg.y - (thickness / 2));
    Point ptEnd(ptStart.x + textSize.width + xpad, ptStart.y - textSize.height);

    auto x = ptStart.x + xpad;
    auto y = ptStart.y - textSize.height;

    imtabbed_rect(image, ptStart, ptEnd, bgColor, thickness, FILLED);

    auto red = COLOR_REDF(textColor);
    auto green = COLOR_GREENF(textColor);
    auto blue = COLOR_BLUEF(textColor);
    cairo_set_source_rgb(cr, red, green, blue);

    cairo_move_to(cr, x, y);
    pango_cairo_show_layout(cr, layout);

    g_object_unref(layout);
    cairo_surface_destroy(surface);
    cairo_destroy(cr);
}

uint32_t imtextcolor(uint32_t bgColor)
{
    constexpr auto black = 0x000000;
    constexpr auto white = 0xFFFFFF;
    constexpr auto gamma = 2.2f;

    auto r = COLOR_REDF(bgColor);
    auto g = COLOR_GREENF(bgColor);
    auto b = COLOR_BLUEF(bgColor);

    const auto luma = 0.2126f * std::pow(r, gamma) + 0.7152f * std::pow(g, gamma) + 0.0722f * std::pow(b, gamma);

    return luma > 0.2f ? black : white;
}

} // px

