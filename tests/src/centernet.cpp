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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <opencv2/opencv.hpp>

#include "Box.h"
#include "CenterNetTargetBuilder.h"

using namespace px;
using namespace testing;

TEST(CenterNetTests, Construction)
{
    CenterNetTargetBuilder builder(20, 1, 128, 128);

    // Center at (64, 64) with size (12.8, 12.8)
    auto targets = builder.buildTargets({ { 0, DarkBox(0.5f, 0.5f, 0.1f, 0.1f) } });
}

TEST(CenterNetTests, HeatmapHasCorrectPeak)
{
    auto classId = 1;
    auto numClasses = 2;
    auto stride = 1;
    auto imageW = 128;
    auto imageH = 128;

    CenterNetTargetBuilder builder(numClasses, stride, imageW, imageH);

    auto targets = builder.buildTargets({ { classId, DarkBox(0.5f, 0.5f, 0.1f, 0.1f) } });
    const auto& heatmap = targets.heatmap;

    auto fmapW = imageW / stride;
    auto fmapH = imageH / stride;

    float maxVal = 0.0f;
    auto maxX = 0, maxY = 0;

    for (auto y = 0; y < fmapH; ++y) {
        for (auto x = 0; x < fmapW; ++x) {
            auto idx = classId * fmapH * fmapW + y * fmapW + x;
            auto val = heatmap[idx];
            if (val > maxVal) {
                maxVal = val;
                maxX = x;
                maxY = y;
            }
        }
    }

    EXPECT_EQ(maxX, imageW / 2);  // center x
    EXPECT_EQ(maxY, imageH / 2);  // center y
    EXPECT_GE(maxVal, 0.99f);  // absolute floor
}


TEST(CenterNetTests, VisualizeHeatmap)
{
    auto classId = 1;

    auto numClasses = 2;
    auto stride = 1;
    auto imageW = 128;
    auto imageH = 128;

    CenterNetTargetBuilder builder(2, stride, imageW, imageH);

    // Center at (64, 64) with size (12.8, 12.8)
    auto targets = builder.buildTargets({ { classId, DarkBox(0.4f, 0.4f, 0.1f, 0.1f) } });

    auto heatmap = targets.heatmap;

    auto fmapW = imageW / stride;
    auto fmapH = imageH / stride;

    cv::Mat heatmapImage(fmapH, fmapW, CV_32FC1);

    for (auto y = 0; y < fmapH; ++y) {
        for (auto x = 0; x < fmapW; ++x) {
            auto idx = classId * fmapH * fmapW + y * fmapW + x;
            heatmapImage.at<float>(y, x) = heatmap[idx];
        }
    }

    cv::Mat heatmapVis;
    cv::normalize(heatmapImage, heatmapVis, 0, 255, cv::NORM_MINMAX);
    heatmapVis.convertTo(heatmapVis, CV_8UC1);  // Convert to 8-bit for visualization

    cv::applyColorMap(heatmapVis, heatmapVis, cv::COLORMAP_JET);
    cv::imshow("Heatmap", heatmapVis);
    cv::waitKey(0);
}
