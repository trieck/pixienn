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

#include "SmoothCyclicDecayLRPolicy.h"
#include "Error.h"

using namespace px;
using namespace testing;

TEST(SmoothCyclicDecayLRPolicy, SmokeTest)
{
    auto initialLR = 0.1f;  // Initial learning rate
    auto gamma = 0.01f;     // Decay rate
    auto peakHeight = 0.5f; // 50% above the initial floor
    auto peakWidth = 10;    // 10 batches wide
    auto peakInterval = 20; // 20 batches between peaks

    SmoothCyclicDecayLRPolicy lrPolicy(initialLR, gamma, peakHeight, peakWidth, peakInterval);

    EXPECT_FLOAT_EQ(lrPolicy.LR(), 0.1f);
}

TEST(SmoothCyclicDecayLRPolicy, Update)
{
    auto initialLR = 1.0f;
    auto gamma = 0.01f;
    auto peakHeight = 0.5f;
    auto peakWidth = 10;
    auto peakInterval = 20;

    SmoothCyclicDecayLRPolicy lrPolicy(initialLR, gamma, peakHeight, peakWidth, peakInterval);

    // Check that LR at batch 0 equals initial
    float lr0 = lrPolicy.update(0);
    EXPECT_NEAR(lr0, initialLR, 1e-5);

    auto prevLR = lr0;

    for (auto batch = 1; batch <= 100; ++batch) {
        auto lr = lrPolicy.update(batch);
        auto phase = batch % peakInterval;

        if (phase == peakWidth / 2) {
            // Mid-peak: should be a local maximum
            auto before = lrPolicy.update(batch - 1);
            auto after = lrPolicy.update(batch + 1);
            EXPECT_GE(lr, before) << "Expected LR to rise into midpoint of peak at batch " << batch;
            EXPECT_GE(lr, after) << "Expected LR to fall after midpoint of peak at batch " << batch;
        } else if (phase == 1) {
            // Just past a new peak: LR should jump above the previous (decaying) floor
            EXPECT_GT(lr, prevLR) << "Expected LR to rise at start of peak at batch " << batch;
        }

        prevLR = lr;
    }
}
