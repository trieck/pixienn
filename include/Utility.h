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

#ifndef UTILITY_H
#define UTILITY_H

#include <opencv2/core/types.hpp>

#include "Common.h"

namespace px {

std::string fmtInt(int number);

template<typename T>
T randomUniform(T min = 0, T max = 1)
{
    if (max < min) {
        std::swap(min, max);
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> distribution(min, max);
        return distribution(gen);
    } else {
        std::uniform_real_distribution<T> distribution(min, max);
        return distribution(gen);
    }
}

template<typename T>
T randomScale(T s)
{
    auto scale = randomUniform<T>(1, s);
    if (randomUniform<T>() > 0.5) {
        return scale;
    } else {
        return 1 / scale;
    }
}

}   // px

#endif // UTILITY_H


