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
#include "CosineLRPolicy.h"

namespace px {

CosineAnnealingLRPolicy::CosineAnnealingLRPolicy(float initialLR, float minLR, int batchesPerCycle)
        : initialLR_(initialLR), currentLR_(initialLR), minLR_(minLR), batchesPerCycle_(batchesPerCycle)
{
    PX_CHECK(initialLR_ > 0, "CosineAnnealingLRPolicy: initial learning rate must be greater than zero");
    PX_CHECK(initialLR_ > minLR_, "CosineAnnealingLRPolicy: initial learning rate must be greater than minLR");
    PX_CHECK(batchesPerCycle_ > 0, "CosineAnnealingLRPolicy: batches per cycle must be greater than zero");
}

float CosineAnnealingLRPolicy::LR() const noexcept
{
    return currentLR_;
}

float CosineAnnealingLRPolicy::update(int batchNum)
{
    // Calculate the position within the current cycle
    auto pos = static_cast<float>(batchNum % batchesPerCycle_) / batchesPerCycle_;

    // Calculate the cosine annealing factor
    auto factor = 0.5f * (1 + std::cos(pos * M_PI));

    // Smoothly anneal between initialLR and lrMin
    currentLR_ = minLR_ + (initialLR_ - minLR_) * factor;

    return currentLR_;
}

void CosineAnnealingLRPolicy::reset()
{
    currentLR_ = initialLR_;
}

} // px
