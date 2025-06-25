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