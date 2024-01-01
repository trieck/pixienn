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

#ifndef PIXIENN_IMAGE_H
#define PIXIENN_IMAGE_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include "PxTensor.h"
#include "ImageVec.h"

namespace px {

struct LBMat
{
    cv::Mat image;
    int originalWidth;
    int originalHeight;
    float ax;
    float ay;
    float dx;
    float dy;
};

cv::Scalar immidpoint(const cv::Mat& image);

// Distort the image with hue, saturation and exposure
void imdistort(cv::Mat& image, float hue, float saturation, float exposure);

// Resize image with letterboxing
LBMat imletterbox(const cv::Mat& image, int width, int height);

// Normalize image pixel values
cv::Mat imnormalize(const cv::Mat& image);

// Denormalize image pixel values
cv::Mat imdenormalize(const cv::Mat& image);

// Place an image on a background
void implace(const cv::Mat& image, int w, int h, int dx, int dy, cv::Mat& canvas);
void implace(const cv::Mat& image, int w, int h, const cv::Rect& roiSrc, cv::Rect& roiDest, cv::Mat& canvas);
void calculateROI(int w, int h, int dx, int dy, cv::Rect& roiSrc, cv::Rect& roiDest, const cv::Mat& canvas);

// Read an image from a file
cv::Mat imread(const char* path);

// Read an image from a file with a specific height and width
LBMat imread(const char* path, int width, int height);

// Read an image and normalize pixel values
cv::Mat imreadNormalize(const char* path);

// Read an image and normalize pixel values with a specific height and width
LBMat imreadNormalize(const char* path, int width, int height);

// Read an image in TIFF format
cv::Mat imreadTiff(const char* path);

// Read an image as vector
ImageVec imreadVector(const char* path);

// Read an image as vector with a specific height and width
ImageVec imreadVector(const char* path, int width, int height);

// Save an image to a file
void imsave(const char* path, const cv::Mat& image);

// Save an image in TIFF format
void imsaveTiff(const char* path, const cv::Mat& image);

// Save an ImageVector in TIFF format
void imsave(const char* path, ImageVec& image);

// Convert image to a PxCpuVector
PxCpuVector imvector(const cv::Mat& image);

// Draw a rectangle on the image
void imrect(cv::Mat& image, const cv::Rect& rect, uint32_t color, int thickness = 1, int lineType = cv::LINE_AA);

// Draw a rectangle with tabbed corners on the image
void imtabbedRect(cv::Mat& img, const cv::Point& pt1, const cv::Point& pt2, uint32_t color,
                  int thickness = 1, int lineType = cv::LINE_AA, int cornerRadius = 2);

// Draw tabbed text on the image
void imtabbedText(cv::Mat& image, const char* text, const cv::Point& ptOrg, uint32_t textColor, uint32_t bgColor,
                  int thickness = 1);

// Get the text color for an image background color
uint32_t imtextcolor(uint32_t color);

} // px

#endif // PIXIENN_IMAGE_H
