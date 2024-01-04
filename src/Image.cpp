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

#ifdef USE_PANGO

#include <cairo/cairo.h>
#include <pango/pangocairo.h>

#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

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

cv::Mat imreadTiff(const char* path)
{
    return readTIFF(path);
}

Mat imread(const char* path)
{
    boost::filesystem::path filePath(path);

    Mat image;

    auto extension = filePath.extension().string();
    if (extension == ".tiff" || extension == ".tif") {
        image = imreadTiff(path);
    } else {
        image = imread(path, IMREAD_UNCHANGED);
        PX_CHECK(!image.empty(), "Could not open image \"%s\".", path);
    }

    return image;
}

LBMat imread(const char* path, int width, int height)
{
    auto image = imread(path);

    // size image to match width and height
    auto lbmat = imletterbox(image, width, height);

    return lbmat;
}

ImageVec imreadVector(const char* path)
{
    auto image = imreadNormalize(path);

    // convert the image from interleaved to planar
    auto vector = imvector(image);

    ImageVec imageVec;
    imageVec.imagePath = path;
    imageVec.data = std::move(vector);
    imageVec.channels = image.channels();
    imageVec.size = { image.cols, image.rows };
    imageVec.originalSize = imageVec.size;
    imageVec.ax = 1.0f;
    imageVec.ay = 1.0f;
    imageVec.dx = 0.0f;
    imageVec.dy = 0.0f;

    return imageVec;
}

ImageVec imreadVector(const char* path, int width, int height)
{
    auto lbmat = imreadNormalize(path, width, height);

    // convert the image from interleaved to planar
    auto vector = imvector(lbmat.image);

    ImageVec imageVec;
    imageVec.imagePath = path;
    imageVec.data = std::move(vector);
    imageVec.channels = lbmat.image.channels();
    imageVec.size = { lbmat.image.cols, lbmat.image.rows };
    imageVec.originalSize = { lbmat.originalWidth, lbmat.originalHeight };
    imageVec.ax = lbmat.ax;
    imageVec.ay = lbmat.ay;
    imageVec.dx = lbmat.dx;
    imageVec.dy = lbmat.dy;

    return imageVec;
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
    } else if (image.channels() == 4) {
        cv::cvtColor(image, swapped, CV_BGRA2RGB);
    } else {
        swapped = image;
    }

    // convert to float and normalize
    Mat out;
    swapped.convertTo(out, CV_32FC(swapped.channels()));

    out /= 255.0f;

    return out;
}

Mat imdenormalize(const Mat& image)
{
    Mat swapped;

    cvtColor(image, swapped, CV_RGB2BGR);
    swapped.convertTo(swapped, CV_8UC3, 255.0);

    return swapped;
}

// read an image and normalize
Mat imreadNormalize(const char* path)
{
    auto image = imread(path);

    auto normal = imnormalize(image);

    return normal.clone();
}

LBMat imreadNormalize(const char* path, int width, int height)
{
    auto image = imreadNormalize(path);

    auto lbmat = imletterbox(image, width, height);

    return lbmat;
}

void imsave(const char* path, const cv::Mat& image)
{
    auto result = imwrite(path, image);
    PX_CHECK(result, "Could not save image \"%s\".", path);
}

// save an image in normalized float format as TIFF
void imsave(const char* path, ImageVec& image)
{
    cv::Mat mat(image.size.height, image.size.width, CV_MAKETYPE(CV_32F, image.channels));

    auto* pimage = image.data.data();
    auto* pmat = mat.ptr<float>();
    auto planeSize = image.size.height * image.size.width;

    // convert planar image to interleaved mat
    for (auto i = 0; i < planeSize; ++i) {
        for (auto j = 0; j < image.channels; ++j) {
            pmat[i * image.channels + j] = pimage[i + j * planeSize];
        }
    }

    imsaveTiff(path, mat);
}

// save an image in normalized float format as TIFF
void imsaveTiff(const char* path, const cv::Mat& image)
{
    Mat tiffImage(image);
    if (tiffImage.type() != CV_32FC3 && tiffImage.type() != CV_32FC1) {
        tiffImage = imnormalize(image);
    }

    writeTIFF(path, tiffImage);
}

cv::Scalar immidpoint(const cv::Mat& image)
{
    int depth = image.depth();
    double midpoint;

    if (depth == CV_8U) {
        midpoint = 255.0 / 2.0;
    } else if (depth == CV_32F) {
        midpoint = 0.5;
    } else {
        PX_ERROR_THROW("Unsupported image depth.");
    }

    return cv::Scalar::all(midpoint);
}

LBMat imletterbox(const Mat& image, int width, int height)
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

    auto midpoint = immidpoint(image);
    Mat boxed{ height, width, image.type(), midpoint };

    auto x = (width - newWidth) / 2;
    auto y = (height - newHeight) / 2;

    resized.copyTo(boxed(Rect(x, y, resized.cols, resized.rows)));

    LBMat lbmat;
    lbmat.image = boxed;
    lbmat.originalWidth = imageWidth;
    lbmat.originalHeight = imageHeight;
    lbmat.ax = newWidth / (float) width;
    lbmat.ay = newHeight / (float) height;
    lbmat.dx = x / static_cast<float>(width);
    lbmat.dy = y / static_cast<float>(height);

    return lbmat;
}

PxCpuVector imvector(const cv::Mat& image)
{
    PX_CHECK(image.isContinuous(), "Non-continuous mat not supported.");

    auto channels = image.channels();
    auto width = image.cols;
    auto height = image.rows;
    auto depth = image.depth();

    PX_CHECK(depth == CV_32F, "Only 32-bit floating point images supported.");

    auto planeSize = height * width;
    PxCpuVector vector(planeSize * channels);
    auto* pvector = vector.data();
    auto* pimage = image.ptr<float>();

    // convert interleaved mat to planar image
    for (auto i = 0; i < planeSize; ++i) {
        for (auto j = 0; j < channels; ++j) {
            pvector[i + j * planeSize] = pimage[i * channels + j];
        }
    }

    return vector;
}

void calculateROI(int w, int h, int dx, int dy, cv::Rect& roiSrc, cv::Rect& roiDest, const cv::Mat& canvas)
{
    int xStart = std::max(0, dx);
    int yStart = std::max(0, dy);
    int xEnd = std::min(canvas.cols, dx + w);
    int yEnd = std::min(canvas.rows, dy + h);

    roiSrc = { xStart - dx, yStart - dy, xEnd - xStart, yEnd - yStart };
    roiDest = { xStart, yStart, xEnd - xStart, yEnd - yStart };
}

void implace(const cv::Mat& image, int w, int h, const cv::Rect& roiSrc, cv::Rect& roiDest, cv::Mat& canvas)
{
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(w, h), cv::INTER_AREA);

    resized(roiSrc).copyTo(canvas(roiDest));
}

void implace(const cv::Mat& image, int w, int h, int dx, int dy, cv::Mat& canvas)
{
    cv::Rect roiSrc, roiDest;
    calculateROI(w, h, dx, dy, roiSrc, roiDest, canvas);

    implace(image, w, h, roiSrc, roiDest, canvas);
}

void imdistort(cv::Mat& image, float hue, float saturation, float exposure)
{
    auto imageType = image.type();

    if (image.channels() != 3) {
        return;
    }

    if (imageType == CV_8UC3) {
        image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    }

    cv::cvtColor(image, image, cv::COLOR_RGB2HSV);

    std::vector<cv::Mat> hsv;
    cv::split(image, hsv);

    hsv[0] = cv::max(0.0f, cv::min(hsv[0] + 360 * hue, 360.0f));
    hsv[1] = cv::max(0.0f, cv::min(hsv[1] * saturation, 1.0f));
    hsv[2] = cv::max(0.0f, cv::min(hsv[2] * exposure, 1.0f));

    cv::Mat merged;
    cv::merge(hsv, merged);

    cv::cvtColor(merged, image, cv::COLOR_HSV2RGB);

    if (imageType == CV_8UC3) {
        image.convertTo(image, CV_8UC3, 255.0);
    }
}

void imrect(cv::Mat& image, const cv::Rect& rect, uint32_t rgb, int thickness, int lineType)
{
    rectangle(image, rect, MAKE_CV_COLOR(rgb), thickness, lineType, 0);
}

void imtabbedRect(cv::Mat& image, const cv::Point& pt1, const cv::Point& pt2, uint32_t rgb, int thickness,
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

#ifdef USE_PANGO

void imtabbedText(cv::Mat& image, const char* text, const cv::Point& ptOrg, uint32_t textColor, uint32_t bgColor,
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
    auto* desc = pango_font_description_from_string("Quicksand Semi-Bold 12, Sans 12");
    pango_layout_set_font_description(layout, desc);
    pango_font_description_free(desc);

    // set layout text
    pango_layout_set_text(layout, text, -1);

    // get pixel size
    cv::Size textSize;
    pango_layout_get_pixel_size(layout, &textSize.width, &textSize.height);
    textSize.width += xpad;

    auto yoffset = thickness / 2;

    auto x0 = std::max(0, std::min(ptOrg.x, image.cols - 1));
    auto y0 = std::max(0, std::min(ptOrg.y - yoffset, image.rows - 1));

    auto x1 = std::max(0, std::min(x0 + textSize.width, image.cols - 1));
    auto y1 = std::max(0, std::min(y0 - textSize.height, image.rows - 1));

    if (x1 - x0 < textSize.width) {
        x0 = std::max(x1 - textSize.width, 0);
    }

    if (y0 - y1 < textSize.height) {
        y0 = std::min(y1 + textSize.height, image.rows - 1);
    }

    Point ptStart(x0, y0);
    Point ptEnd(x1, y1);

    auto x = std::min(ptStart.x + xpad, image.cols - 1);

    imtabbedRect(image, ptStart, ptEnd, bgColor, thickness, FILLED);

    auto red = COLOR_REDF(textColor);
    auto green = COLOR_GREENF(textColor);
    auto blue = COLOR_BLUEF(textColor);
    cairo_set_source_rgb(cr, red, green, blue);

    cairo_move_to(cr, x, y1);
    pango_cairo_show_layout(cr, layout);

    g_object_unref(layout);
    cairo_surface_destroy(surface);
    cairo_destroy(cr);
}

#else

void imtabbedText(cv::Mat& image, const char* text, const cv::Point& ptOrg, uint32_t textColor, uint32_t bgColor,
                  int thickness)
{
    constexpr auto fontFace = FONT_HERSHEY_SIMPLEX;
    constexpr auto fontScale = 0.5f;
    constexpr auto xpad = 4;

    auto baseline = 0;
    auto textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);

    baseline += thickness;
    textSize.width += xpad;
    textSize.height += baseline;

    auto yoffset = thickness / 2;
    Point ptStart(ptOrg.x, ptOrg.y - yoffset);
    Point ptEnd(ptStart.x + textSize.width + xpad, ptStart.y - textSize.height);
    Point ptText(ptStart.x + xpad, ptStart.y - baseline + thickness);

    imtabbedRect(image, ptStart, ptEnd, bgColor, thickness, FILLED);

    putText(image, text, ptText, fontFace, fontScale, MAKE_CV_COLOR(textColor), 1, LINE_AA);
}

#endif // USE_PANGO

uint32_t imtextcolor(uint32_t bgColor)
{
    constexpr auto black = 0x000000;
    constexpr auto white = 0xFFFFFF;
    constexpr auto gamma = 2.2f;

    auto r = COLOR_REDF(bgColor);
    auto g = COLOR_GREENF(bgColor);
    auto b = COLOR_BLUEF(bgColor);

    const auto luma = 0.2126f * std::pow(r, gamma) + 0.7152f * std::pow(g, gamma) + 0.0722f * std::pow(b, gamma);

    return luma > 0.3f ? black : white;
}

} // px

