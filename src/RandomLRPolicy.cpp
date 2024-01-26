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
#include "RandomLRPolicy.h"

namespace px {

RandomLRPolicy::RandomLRPolicy(float initialLR, float minLR, std::size_t updateInterval)
        : initialLR_(initialLR), LR_(initialLR), minLR_(minLR), updateInterval_(updateInterval)
{
    PX_CHECK(initialLR > 0.0f, "Initial learning rate must be positive");
    PX_CHECK(minLR > 0.0f, "Minimum learning rate must be positive");
    PX_CHECK(updateInterval > 0, "Update interval must be positive");
    PX_CHECK(initialLR >= minLR, "Initial learning rate must be greater than or equal to minimum learning rate");

    randomEngine_.seed(std::random_device()());
}

float RandomLRPolicy::LR() const noexcept
{
    return LR_;
}

float RandomLRPolicy::update(int batchNum) noexcept
{
    if (batchNum % updateInterval_ == 0) {
        std::uniform_real_distribution<float> distribution(minLR_, initialLR_);
        LR_ = distribution(randomEngine_);
    }

    return LR_;
}

void RandomLRPolicy::reset() noexcept
{
    LR_ = initialLR_;
}

}   // px


