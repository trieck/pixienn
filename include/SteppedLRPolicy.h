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

#ifndef PIXIENN_STEPPEDLRPOLICY_H__
#define PIXIENN_STEPPEDLRPOLICY_H__

#include "Common.h"

namespace px {

class SteppedLRPolicy
{
public:
    SteppedLRPolicy();
    SteppedLRPolicy(float lr, std::vector<int> steps, const std::vector<float> scales);

    float LR() const noexcept;
    float origLR() const noexcept;
    float scale() const noexcept;
    float update(int batchNum);
    int step() const noexcept;

    void set(float lr, std::vector<int> steps, const std::vector<float> scales);
    void reset();

private:
    int step_ = 0;
    float lr_ = 0, origLr_ = 0;

    std::vector<int> steps_;
    std::vector<float> scales_;
};

}   // px

#endif // PIXIENN_STEPPEDLRPOLICY_H__