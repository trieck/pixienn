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

#include "SigmoidLRPolicy.h"

using namespace px;
using namespace testing;

TEST(SigmoidLRPolicy, SmokeTest)
{
    SigmoidLRPolicy lrPolicy(0.1, 0.01, 12.0, 5000);

    EXPECT_FLOAT_EQ(lrPolicy.LR(), 0.1);

    lrPolicy.reset();

    EXPECT_FLOAT_EQ(lrPolicy.LR(), 0.1);
}

TEST(SigmoidLRPolicy, Update)
{
    SigmoidLRPolicy lrPolicy(0.1, 0.01, 12.0, 5000);

    lrPolicy.update(0);
    EXPECT_NEAR(lrPolicy.LR(), 0.1, 1e-2);

    lrPolicy.update(2500);
    EXPECT_NEAR(lrPolicy.LR(), 0.055, 1e-2);

    lrPolicy.update(5000);
    EXPECT_NEAR(lrPolicy.LR(), 0.01, 1e-2);

    lrPolicy.reset();
    EXPECT_NEAR(lrPolicy.LR(), 0.1, 1e-2);
}

