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
#include "SmoothSteppedLRPolicy.h"

namespace px {

SmoothSteppedLRPolicy::SmoothSteppedLRPolicy(float initialLR, std::vector<int> steps, std::vector<float> targets)
        : initialLR_(initialLR), currentLR_(initialLR), steps_(std::move(steps)), targets_(std::move(targets))
{
    PX_CHECK(initialLR_ > 0, "SmoothSteppedLRPolicy: initial learning rate must be greater than zero");
    PX_CHECK(std::is_sorted(steps_.begin(), steps_.end()), "SmoothSteppedLRPolicy: steps must be sorted");
    PX_CHECK(steps_.size() > 0, "SmoothSteppedLRPolicy: steps must not be empty");
    PX_CHECK(targets_.size() > 0, "SmoothSteppedLRPolicy: targets must not be empty");
    PX_CHECK(steps_.size() == targets_.size(), "SmoothSteppedLRPolicy: steps and targets must have the same size");
}

float SmoothSteppedLRPolicy::update(int batchNum)
{
    // Find the index of the nearest step that is less than or equal to the current batch number
    auto index = 0;
    for (auto i = 0; i < steps_.size(); ++i) {
        if (steps_[i] > batchNum) {
            break;
        }
        index = i;
    }

    auto t = 0.0f;
    auto start = currentLR_;
    auto end = targets_[index];

    if (batchNum < steps_[index]) {
        auto diff = steps_[index] - batchNum;
        t = 1 - static_cast<float>(diff) / steps_[index];
    } else if (batchNum >= steps_[index]) {
        auto diff = batchNum - steps_[index];
        t = static_cast<float>(diff) / (steps_[index + 1] - steps_[index]);
        start = targets_[index];
        end = targets_[index + 1];
    }

    currentLR_ = smoothStepTransition(t, start, end);

    return currentLR_;
}

float SmoothSteppedLRPolicy::smoothStepTransition(float t, float start, float end)
{
    // Use smoothstep function to smoothly interpolate between start and end
    t = std::max(0.0f, std::min(1.0f, t)); // Clamp t to the range [0, 1]

    return start + t * t * (3.0f - 2.0f * t) * (end - start);
}

float SmoothSteppedLRPolicy::LR() const noexcept
{
    return currentLR_;
}

void SmoothSteppedLRPolicy::reset()
{
    currentLR_ = initialLR_;
}

}   // px
