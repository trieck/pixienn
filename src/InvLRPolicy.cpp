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
#include "InvLRPolicy.h"

namespace px {

InvLRPolicy::InvLRPolicy() : origLr_(0), lr_(0), gamma_(0), power_(0)
{
}

InvLRPolicy::InvLRPolicy(float lr, float gamma, float power) : origLr_(lr), lr_(lr),
                                                               gamma_(gamma), power_(power)
{
}

float InvLRPolicy::update(int batchNum)
{
    lr_ = origLr_ / std::pow(1 + gamma_ * batchNum, power_);

    return lr_;
}

float InvLRPolicy::LR() const noexcept
{
    return lr_;
}

float InvLRPolicy::origLR() const noexcept
{
    return origLr_;
}

void InvLRPolicy::reset()
{
    lr_ = origLr_;
}

}   // px
