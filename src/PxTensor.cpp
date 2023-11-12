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

#include <xtensor/xtensor.hpp>
#include "PxTensor.h"

using namespace px;

template<typename T=float, std::size_t N = 1>
using cpu_tensor_t = xt::xtensor_container<xt::uvector<T>, N>;

template<typename T=float, std::size_t N = 1>
using cuda_tensor_t = xt::xtensor_container<cuda_vector_t<T>, N>;

template<std::size_t N = 1>
using cpu_tensor = cpu_tensor_t<float, N>;

template<std::size_t N = 1>
using cuda_tensor = cuda_tensor_t<float, N>;

///////////////////////////////////////////////////////////////////////////////
void foobar()
{
    xt::xtensor<float, 4>::shape_type shape{ 1, 2, 2, 2 };

    using cpu_1d = cpu_tensor<1>;
    auto S = xtl::make_sequence<cpu_1d>({1, 2, 3});
    for (auto x: S) {
        printf("%.2f\n", x);
    }

    printf("\n");

    using cuda_1d = cuda_tensor<1>;
    auto T = xtl::make_sequence<cuda_1d>({1, 2, 3});
    for (auto x: T) {
        printf("%.2f\n", (float)x);
    }
}

