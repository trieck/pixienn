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
#include "PxTensor.h"

namespace px {

cv::Mat imchannel(const cv::Mat& image, int c);
cv::Mat immake(int height, int width, int channels, float value = 0.0f);
cv::Mat imrandom(int height, int width, int channels);
cv::Mat imletterbox(const cv::Mat& image, int width, int height);
cv::Mat imread_normalize(const char* path);
cv::Mat imnormalize(const cv::Mat& image);
cv::Mat imread(const char* path);
void imsave(const char* path, const cv::Mat& image);
void imsave_normalize(const char* path, const cv::Mat& image);
float imget(const cv::Mat& image, int x, int y, int c);
float imgetextend(const cv::Mat& image, int x, int y, int c);
void im2dconvolve(const cv::Mat& image, int imChannel, const cv::Mat& kernel, int kernelChannel, int stride,
                  cv::Mat& out, int outChannel);
void imadd(cv::Mat& image, int x, int y, int c, float value);
void imconvolve(const cv::Mat& image, const cv::Mat& kernel, int stride, int channel, cv::Mat& out);
void imset(cv::Mat& image, int x, int y, int c, float value);
void imzero(const cv::Mat& image, int c);
PxCpuVector imvector(const cv::Mat& image);
void imrect(cv::Mat& image, const cv::Rect& rect, uint32_t color, int thickness = 1);
void imtext(cv::Mat& image, const char* text, const cv::Point& ptOrg, uint32_t textColor, uint32_t bgColor,
            int thickness = 1);
uint32_t imgetcolor(uint32_t index);
uint32_t imtextcolor(uint32_t color);

} // px

#endif // PIXIENN_IMAGE_H
