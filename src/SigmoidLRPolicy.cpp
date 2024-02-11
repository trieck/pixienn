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
#include "SigmoidLRPolicy.h"

namespace px {

SigmoidLRPolicy::SigmoidLRPolicy(float initialLR, float targetLR, float factor, int maxBatches)
        : initialLR_(initialLR), currentLR_(initialLR), targetLR_(targetLR), factor_(factor), maxBatches_(maxBatches)
{
    PX_CHECK(initialLR_ > 0, "SigmoidLRPolicy: initial learning rate must be greater than zero");
    PX_CHECK(targetLR_ > 0, "SigmoidLRPolicy: target learning rate must be greater than zero");
    PX_CHECK(factor_ > 0, "SigmoidLRPolicy: factor must be greater than zero");
    PX_CHECK(maxBatches_ > 0, "SigmoidLRPolicy: max batches must be greater than zero");
}

float SigmoidLRPolicy::LR() const noexcept
{
    return currentLR_;
}

float SigmoidLRPolicy::update(int batchNum)
{
    auto progress = batchNum / static_cast<float>(maxBatches_);

    auto sigmoidInput = factor_ * (progress - 0.5);

    currentLR_ = initialLR_ + (targetLR_ - initialLR_) / (1 + std::exp(-sigmoidInput));

    return currentLR_;
}

void SigmoidLRPolicy::reset()
{
    currentLR_ = initialLR_;
}

}   // px
