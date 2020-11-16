/********************************************************************************
* Copyright 2020 Thomas A. Rieck, All Rights Reserved
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

#ifndef PIXIENN_ACTIVATION_H
#define PIXIENN_ACTIVATION_H

#include "common.h"
#include "xtensor/xcontainer.hpp"

namespace px {

class Activation
{
public:
    using Ptr = std::shared_ptr<Activation>;

    static Activation::Ptr get(const std::string& s);

    virtual void apply(float* begin, float* end) const = 0;

    template<typename T>
    void apply(xt::xcontainer<T>&) const;
};

template<typename T>
void Activation::apply(xt::xcontainer<T>& container) const
{
    apply(container.begin(), container.end());
}

}   // px

#endif // PIXIENN_ACTIVATION_H
