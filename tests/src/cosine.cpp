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

#include "CosineLRPolicy.h"

using namespace px;
using namespace testing;

TEST(CosineAnnealingLRPolicy, SmokeTest)
{
    CosineAnnealingLRPolicy lrPolicy(0.1, 0.0001, 1000);

    EXPECT_FLOAT_EQ(lrPolicy.LR(), 0.1);

    lrPolicy.reset();

    EXPECT_FLOAT_EQ(lrPolicy.LR(), 0.1);
}

TEST(CosineAnnealingLRPolicy, Update)
{
    auto initialLR = 0.1f;
    auto minLR = 0.0001f;
    auto batchesPerCycle = 1000;

    CosineAnnealingLRPolicy lrPolicy(initialLR, minLR, batchesPerCycle);

    lrPolicy.update(0);
    EXPECT_NEAR(lrPolicy.LR(), initialLR, 1e-2);

    auto midBatch = batchesPerCycle / 2;
    auto midLR = (initialLR + minLR) / 2;

    lrPolicy.update(midBatch);
    EXPECT_NEAR(lrPolicy.LR(), midLR, 1e-2);

    lrPolicy.update(batchesPerCycle);

    EXPECT_NEAR(lrPolicy.LR(), initialLR, 1e-2);

    lrPolicy.reset();

    EXPECT_NEAR(lrPolicy.LR(), initialLR, 1e-2);
}
