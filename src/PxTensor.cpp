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

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include "PxTensor.h"

///////////////////////////////////////////////////////////////////////////////
using namespace px;

template<typename T=float, std::size_t N = 1>
using xt_cpu_tensor_t = xt::xtensor_container<xt::uvector<T>, N>;

template<typename T=float, std::size_t N = 1>
using xt_cuda_tensor_t = xt::xtensor_container<cuda_vector_t<T>, N>;

template<std::size_t N = 1>
using cpu_tensor = xt_cpu_tensor_t<float, N>;

template<std::size_t N = 1>
using cuda_tensor = xt_cuda_tensor_t<float, N>;

void foobar()
{
    xt::xtensor<float, 4>::shape_type shape{1, 2, 2, 2 };

    cpu_tensor<4> image = xt::ones<float>(shape);
    for (auto x: image) {
        printf("%.2f\n", x);
    }

    printf("\n");

    cuda_tensor<4> cuda_image(shape);
    for (auto x: cuda_image) {
        printf("%.2f\n", x);
    }
}

