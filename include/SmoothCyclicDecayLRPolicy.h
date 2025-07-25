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

#pragma once

#include "Common.h"
#include "LRPolicy.h"

namespace px {

class SmoothCyclicDecayLRPolicy : public LRPolicy
{
public:
    SmoothCyclicDecayLRPolicy(float initialLR, float gamma, float peakHeight, int peakWidth, int peakInterval);

    // Inherited methods
    float update(int batchNum) override;
    float LR() const noexcept override;
    void reset() override;

private:
    float initialLR_;
    float LR_;
    float gamma_;
    float peakHeight_;
    int peakWidth_;
    int peakInterval_;
};

} // px
