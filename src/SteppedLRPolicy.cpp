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
#include "SteppedLRPolicy.h"

namespace px {

SteppedLRPolicy::SteppedLRPolicy() : lr_(0), origLr_(0)
{
}

SteppedLRPolicy::SteppedLRPolicy(float lr, std::vector<int> steps, const std::vector<float> scales)
{
    set(lr, std::move(steps), std::move(scales));
}

void SteppedLRPolicy::set(float lr, std::vector<int> steps, const std::vector<float> scales)
{
    PX_CHECK(steps.size() == scales.size(), "steps and scales must be the same size.");
    PX_CHECK(steps.size() > 0, "steps and scales must have a non-zero size.");

    lr_ = origLr_ = lr;
    steps_ = std::move(steps);
    scales_ = std::move(scales);
}

float SteppedLRPolicy::update(int batchNum)
{
    lr_ = origLr_;

    for (auto i = 0; i < steps_.size(); ++i) {
        if (steps_[i] > batchNum) {
            break;
        }

        lr_ *= scales_[i];
    }

    return lr_;
}

float SteppedLRPolicy::LR() const noexcept
{
    return lr_;
}

float SteppedLRPolicy::origLR() const noexcept
{
    return origLr_;
}

void SteppedLRPolicy::reset()
{
    lr_ = origLr_;
}

}   // px
