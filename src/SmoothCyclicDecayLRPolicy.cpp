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

#include "Error.h"
#include "SmoothCyclicDecayLRPolicy.h"

namespace px {

SmoothCyclicDecayLRPolicy::SmoothCyclicDecayLRPolicy(float initialLR, float gamma, float peakHeight, int peakWidth,
                                                     int peakInterval) :
        initialLR_(initialLR), LR_(initialLR), gamma_(gamma), peakHeight_(peakHeight), peakWidth_(peakWidth),
        peakInterval_(peakInterval)
{
    PX_CHECK(initialLR > 0.0f, "Initial learning rate must be positive");
    PX_CHECK(gamma > 0.0f, "Gamma must be positive");
    PX_CHECK(peakHeight > 0.0f, "Peak height must be positive");
    PX_CHECK(peakWidth > 0, "Peak width must be positive");
    PX_CHECK(peakInterval > 0, "Peak interval must be positive");
}

float SmoothCyclicDecayLRPolicy::update(int batchNum)
{
    auto floor = initialLR_ * std::exp(-gamma_ * batchNum);
    auto lr = floor;

    auto phase = batchNum % peakInterval_;

    if (phase < peakWidth_) {
        auto t = static_cast<float>(phase) / static_cast<float>(peakWidth_);
        auto bump = std::sin(float(M_PI) * t);
        lr += initialLR_ * peakHeight_ * bump;
    }

    LR_ = lr;

    return LR_;
}


float SmoothCyclicDecayLRPolicy::LR() const noexcept
{
    return LR_;
}

void SmoothCyclicDecayLRPolicy::reset()
{
    LR_ = initialLR_;
}

} // px