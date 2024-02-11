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
#include "ConstantLRPolicy.h"

namespace px {

ConstantLRPolicy::ConstantLRPolicy(float initialLR) : lr_(initialLR)
{
    PX_CHECK(initialLR > 0, "ConstantLRPolicy: initial learning rate must be greater than zero");
}

float ConstantLRPolicy::update(int batchNum)
{
    return lr_;
}

float ConstantLRPolicy::LR() const noexcept
{
    return lr_;
}

void ConstantLRPolicy::reset()
{
}

}   // px
