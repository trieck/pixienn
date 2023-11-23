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

#include <cmath>
#include "Common.h"
#include "Math.h"

namespace px {

auto sum(const PxCpuVector& input) -> PxCpuVector::value_type
{
    PxCpuVector::value_type sum = 0;

    for (auto v: input) {
        sum += v;
    }

    return sum;
}

PxCpuVector exp(const PxCpuVector& input)
{
    PxCpuVector output(input.size());

    for (auto i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i]);
    }

    return output;
}

PxCpuVector log(const PxCpuVector& input)
{
    PxCpuVector output(input.size());

    for (auto i = 0; i < input.size(); ++i) {
        output[i] = std::log(input[i]);
    }

    return output;
}

auto logsumexp(const PxCpuVector& input) -> PxCpuVector::value_type
{
    using VT = std::iterator_traits<PxCpuVector::const_iterator>::value_type;

    auto max = *std::max_element(input.begin(), input.end());
    auto sum = std::accumulate(input.begin(), input.end(), VT{},
                               [max](VT a, VT b) { return a + std::exp(b - max); });

    auto lse = max + std::log(sum);

    return lse;
}

PxCpuVector softmax(const PxCpuVector& input)
{
    // compute in log space for numerical stability
    auto softmax = exp(input - logsumexp(input));

    return softmax;
}

}   // px
