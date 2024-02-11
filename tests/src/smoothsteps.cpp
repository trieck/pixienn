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

#include "PxTensor.h"
#include "SmoothSteppedLRPolicy.h"

using namespace px;
using namespace testing;

TEST(SmoothSteppedLRPolicy, SmokeTest)
{
    SmoothSteppedLRPolicy lrPolicy(0.1, { 100, 200, 300 }, { 0.05, 0.02, 0.01 });

    EXPECT_FLOAT_EQ(lrPolicy.LR(), 0.1);

    lrPolicy.reset();

    EXPECT_FLOAT_EQ(lrPolicy.LR(), 0.1);
}

TEST(SmoothSteppedLRPolicy, Update)
{
    SmoothSteppedLRPolicy lrPolicy(0.1, { 100, 200, 300 }, { 0.05, 0.02, 0.01 });

    lrPolicy.update(0);
    EXPECT_NEAR(lrPolicy.LR(), 0.1, 1e-2);

    lrPolicy.update(50);
    EXPECT_LT(lrPolicy.LR(), 0.1);
    EXPECT_GT(lrPolicy.LR(), 0.05);

    lrPolicy.update(100);
    EXPECT_NEAR(lrPolicy.LR(), 0.05, 1e-2);

    lrPolicy.update(150);
    EXPECT_NEAR(lrPolicy.LR(), 0.035, 1e-2);

    lrPolicy.update(200);
    EXPECT_NEAR(lrPolicy.LR(), 0.02, 1e-2);

    lrPolicy.update(300);
    EXPECT_NEAR(lrPolicy.LR(), 0.01, 1e-2);

    lrPolicy.update(400);
    EXPECT_NEAR(lrPolicy.LR(), 0.01, 1e-2);

    lrPolicy.update(0);
    EXPECT_NEAR(lrPolicy.LR(), 0.1, 1e-1);

    lrPolicy.update(110);
    EXPECT_LT(lrPolicy.LR(), 0.05);
    EXPECT_GT(lrPolicy.LR(), 0.02);

    lrPolicy.update(150);
    EXPECT_NEAR(lrPolicy.LR(), 0.035, 1e-2);

    lrPolicy.update(210);
    EXPECT_LT(lrPolicy.LR(), 0.02);
    EXPECT_GT(lrPolicy.LR(), 0.01);

    lrPolicy.update(300);
    EXPECT_NEAR(lrPolicy.LR(), 0.01, 1e-2);
}

TEST(SmoothSteppedLRPolicy, Before)
{
    SmoothSteppedLRPolicy lrPolicy(0.001, { 10000, 20000, 30000 }, { 0.0003, 0.0002, 0.0001 });

    lrPolicy.update(0);
    EXPECT_NEAR(lrPolicy.LR(), 0.001, 1e-5);

    lrPolicy.update(5000);
    EXPECT_NEAR(lrPolicy.LR(), 0.00065, 1e-5);

    lrPolicy.update(10000);
    EXPECT_NEAR(lrPolicy.LR(), 0.0003, 1e-5);

    lrPolicy.update(15000);
    EXPECT_NEAR(lrPolicy.LR(), 0.00025, 1e-5);

    lrPolicy.update(20000);
    EXPECT_NEAR(lrPolicy.LR(), 0.0002, 1e-5);

    lrPolicy.update(25000);
    EXPECT_NEAR(lrPolicy.LR(), 0.00015, 1e-5);

    lrPolicy.update(30000);
    EXPECT_NEAR(lrPolicy.LR(), 0.0001, 1e-5);

    lrPolicy.update(35000);
    EXPECT_NEAR(lrPolicy.LR(), 0.0001, 1e-5);

    lrPolicy.update(0);
    EXPECT_NEAR(lrPolicy.LR(), 0.001, 1e-5);
}
