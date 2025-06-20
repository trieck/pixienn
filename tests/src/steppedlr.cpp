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
#include "SteppedLRPolicy.h"

using namespace px;
using namespace testing;

TEST(LRPolicy, SteppedLRPolicyTest)
{
    auto lr = 0.0000125f;

    std::vector<int> steps = { 200, 400, 600, 800, 20000, 30000 };
    std::vector<float> scales = { 2.5, 2, 2, 2, 0.1, 0.1 };

    SteppedLRPolicy policy(lr, steps, scales);

    policy.update(0);
    ASSERT_FLOAT_EQ(policy.LR(), lr);

    policy.update(200);
    ASSERT_FLOAT_EQ(policy.LR(), lr * scales[0]);

    policy.update(400);
    ASSERT_FLOAT_EQ(policy.LR(), lr * scales[0] * scales[1]);

    policy.update(200);
    ASSERT_FLOAT_EQ(policy.LR(), lr * scales[0]);

    policy.update(600);
    ASSERT_FLOAT_EQ(policy.LR(), lr * scales[0] * scales[1] * scales[2]);

    policy.update(800);
    ASSERT_FLOAT_EQ(policy.LR(), lr * scales[0] * scales[1] * scales[2] * scales[3]);

    policy.update(20000);
    ASSERT_FLOAT_EQ(policy.LR(), lr * scales[0] * scales[1] * scales[2] * scales[3] * scales[4]);

    policy.update(40000);
    ASSERT_FLOAT_EQ(policy.LR(), lr * scales[0] * scales[1] * scales[2] * scales[3] * scales[4] * scales[5]);

    policy.reset();
    ASSERT_FLOAT_EQ(policy.LR(), lr);
}
