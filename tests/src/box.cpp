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

#include "Box.h"

using namespace px;
using namespace testing;

TEST(BoxTests, BoxNms)
{
    Detections detects = {
            Detection(cv::Rect2f(100, 100, 50, 50), 0, 1, 0.9f),
            Detection(cv::Rect2f(100, 100, 50, 50), 0, 1, 0.8f),
            Detection(cv::Rect2f(200, 200, 50, 50), 0, 1, 0.7f)
    };

    auto nmsThreshold = 0.5f;
    auto result = nms(detects, nmsThreshold);

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].box(), cv::Rect2f(100, 100, 50, 50));
    EXPECT_EQ(result[1].box(), cv::Rect2f(200, 200, 50, 50));
}

TEST(BoxTests, BoxNmsDoesNotSuppressAcrossBatchImages)
{
    Detections detects = {
            Detection(cv::Rect2f(100, 100, 50, 50), 0, 1, 0.9f),
            Detection(cv::Rect2f(100, 100, 50, 50), 1, 1, 0.8f)
    };

    const auto result = nms(detects, 0.5f);

    EXPECT_EQ(result.size(), 2);
}

TEST(BoxTests, BoxNmsSuppressesNearIdenticalDifferentClassDuplicates)
{
    Detections detects = {
            Detection(cv::Rect2f(100, 100, 50, 50), 0, 16, 0.9f), // dog
            Detection(cv::Rect2f(100, 100, 50, 50), 0, 15, 0.8f)  // cat
    };

    const auto result = nms(detects, 0.3f);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].classIndex(), 16);
    EXPECT_FLOAT_EQ(result[0].prob(), 0.9f);
}

TEST(BoxTests, BoxNmsKeepsOrdinaryDifferentClassOverlaps)
{
    Detections detects = {
            Detection(cv::Rect2f(100, 100, 50, 50), 0, 16, 0.9f),
            Detection(cv::Rect2f(105, 105, 50, 50), 0, 15, 0.8f)
    };

    const auto result = nms(detects, 0.3f);

    EXPECT_EQ(result.size(), 2);
}

TEST(BoxTests, BoxNmsDoesNotLetSuppressedBoxesSuppressOthers)
{
    Detections detects = {
            Detection(cv::Rect2f(0, 0, 100, 100), 0, 1, 0.9f),
            Detection(cv::Rect2f(25, 0, 100, 100), 0, 1, 0.8f),
            Detection(cv::Rect2f(50, 0, 100, 100), 0, 1, 0.7f)
    };

    const auto result = nms(detects, 0.5f);

    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].box(), cv::Rect2f(0, 0, 100, 100));
    EXPECT_EQ(result[1].box(), cv::Rect2f(50, 0, 100, 100));
}
