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

#ifndef PIXIENN_MATH_H
#define PIXIENN_MATH_H

#include "xtensor/xmath.hpp"

namespace px {

template<typename T>
auto logsumexp(T&& t)
{
    auto max = xt::amax(std::forward<T>(t))();
    return xt::log(xt::sum(xt::exp(std::forward<T>(t) - max)));
}

template<typename T>
auto softmax(T&& t)
{
    // compute in log space for numerical stability
    return xt::exp(std::forward<T>(t) - logsumexp(std::forward<T>(t)));
}

}   // px

#endif // PIXIENN_MATH_H
