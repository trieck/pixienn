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

namespace px {

struct ImageVector
{
    PxCpuVector vector;
    int width, height, channels;
};

// Single-channel extraction from an image
cv::Mat imchannel(const cv::Mat& image, int channel);

// Resize image with letterboxing
cv::Mat imletterbox(const cv::Mat& image, int width, int height);

// Create an image with specified dimensions and channels, optionally filled with a value
cv::Mat immake(int height, int width, int channels, float value = 0.0f);

// Normalize image pixel values
cv::Mat imnormalize(const cv::Mat& image);

// Create a random image with specified dimensions and channels
cv::Mat imrandom(int height, int width, int channels);

// Read an image from a file
cv::Mat imread(const char* path);

// Read an image and resize with normalization
cv::Mat imread_normalize(const char* path);

// Read an image in TIFF format
cv::Mat imread_tiff(const char* path);

// Read an image as vector
ImageVector imread_vector(const char* path);

// Read an image as vector with a specific height and width
ImageVector imread_vector(const char* path, int width, int height);

// Save an image to a file
void imsave(const char* path, const cv::Mat& image);

// Save an image in TIFF format
void imsave_tiff(const char* path, const cv::Mat& image);

// Save an ImageVector in TIFF format
void imsave_tiff(const char* path, ImageVector& image);

// Get the pixel value at a specific position in the image
float imget(const cv::Mat& image, int x, int y, int c);

// Get extended pixel value at a specific position in the image
float imgetextend(const cv::Mat& image, int x, int y, int c);

// Add a value to a pixel in the image
void imadd(cv::Mat& image, int x, int y, int c, float value);

// Set the pixel value at a specific position in the image
void imset(cv::Mat& image, int x, int y, int c, float value);

// Zero out a specific channel in the image
void imzero(const cv::Mat& image, int c);

// Convert image to a PxCpuVector
PxCpuVector imvector(const cv::Mat& image);

// Draw a rectangle on the image
void imrect(cv::Mat& image, const cv::Rect& rect, uint32_t color, int thickness = 1, int lineType = cv::LINE_AA);

// Draw a rectangle with tabbed corners on the image
void imtabbed_rect(cv::Mat& img, const cv::Point& pt1, const cv::Point& pt2, uint32_t color,
                   int thickness = 1, int lineType = cv::LINE_AA, int cornerRadius = 2);

// Draw tabbed text on the image
void imtabbed_text(cv::Mat& image, const char* text, const cv::Point& ptOrg, uint32_t textColor, uint32_t bgColor,
                   int thickness = 1);

// Get the text color for an image background color
uint32_t imtextcolor(uint32_t color);

} // px

#endif // PIXIENN_IMAGE_H
