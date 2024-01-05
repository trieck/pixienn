/********************************************************************************
* Copyright 2023 Thomas A. Rieck, All Rights Reserved
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

#include "DarkBox.h"

using namespace px;
using namespace testing;

TEST(DarkBoxTest, DefaultConstructor)
{
    DarkBox box;

    EXPECT_EQ(box.x(), 0.0f);
    EXPECT_EQ(box.y(), 0.0f);
    EXPECT_EQ(box.w(), 0.0f);
    EXPECT_EQ(box.h(), 0.0f);
}

TEST(DarkBoxTest, ParameterizedConstructor)
{
    DarkBox darkBox(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_EQ(darkBox.x(), 1.0f);
    EXPECT_EQ(darkBox.y(), 2.0f);
    EXPECT_EQ(darkBox.w(), 3.0f);
    EXPECT_EQ(darkBox.h(), 4.0f);
}

TEST(DarkBoxTest, ConstructorFromRect)
{
    cv::Rect2f rect(1.0f, 2.0f, 3.0f, 4.0f);
    DarkBox darkBox(rect);
    EXPECT_EQ(darkBox.x(), 2.5f);
    EXPECT_EQ(darkBox.y(), 4.0f);
    EXPECT_EQ(darkBox.w(), 3.0f);
    EXPECT_EQ(darkBox.h(), 4.0f);
}

TEST(DarkBoxTest, AssignmentOperator)
{
    DarkBox darkBox;
    cv::Rect2f rect(1.0f, 2.0f, 3.0f, 4.0f);
    darkBox = rect;
    EXPECT_EQ(darkBox.x(), 2.5f);
    EXPECT_EQ(darkBox.y(), 4.0f);
    EXPECT_EQ(darkBox.w(), 3.0f);
    EXPECT_EQ(darkBox.h(), 4.0f);
}

TEST(DarkBoxTest, Accessors)
{
    DarkBox darkBox(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_EQ(darkBox.left(), -0.5f);
    EXPECT_EQ(darkBox.right(), 2.5f);
    EXPECT_EQ(darkBox.top(), 0.0f);
    EXPECT_EQ(darkBox.bottom(), 4.0f);
    EXPECT_EQ(darkBox.rect(), cv::Rect2f(-0.5f, 0.0f, 3.0f, 4.0f));
}

TEST(DarkBoxTest, Intersection)
{
    DarkBox box1(1.0f, 2.0f, 3.0f, 4.0f);
    DarkBox box2(2.0f, 3.0f, 3.0f, 4.0f);

    auto area = box1.intersection(box2);

    EXPECT_FLOAT_EQ(area, 6.0f);
}

TEST(DarkBoxTest, UnionArea)
{
    DarkBox box1(1.0f, 2.0f, 3.0f, 4.0f);
    DarkBox box2(2.0f, 3.0f, 3.0f, 4.0f);

    auto area = box1.unionArea(box2);

    EXPECT_FLOAT_EQ(area, 20.0f);
}

TEST(DarkBoxTest, IntersectionOverUnion)
{
    DarkBox box1(1.0f, 2.0f, 3.0f, 4.0f);
    DarkBox box2(2.0f, 3.0f, 3.0f, 4.0f);

    float iou = box1.iou(box2);

    auto expected = box1.intersection(box2) / box1.unionArea(box2);

    EXPECT_FLOAT_EQ(iou, expected);
    EXPECT_FLOAT_EQ(iou, 0.3f);
}