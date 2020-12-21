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

#include "Error.h"
#include "Tensor.cuh"

namespace px {

template<typename T, Device D, typename B>
static void inline print(tensor<T, D, B>&& t)
{
    int i = 0;
    for (const auto& v: t) {
        std::cout << ++i << "    " << v << std::endl;
    }
}

void foobar()
{
    print(cpu_array::fill({10, 100, 10}, 1));
    print(cpu_tensor<float, 3>::fill({10, 100, 10}, 2));

    print(cuda_array::fill({10, 100, 10}, 3));
    print(cuda_tensor<float, 3>::fill({10, 100, 10}, 4));

//    //auto a = cuda_tensor<float, 3>::fill({1000, 1000, 1000}, 2.71828182);
//
//    auto a = cuda_tensor<float, 1>::random({100000000});
//
//    auto p = a.data();
//    std::cout << *p << std::endl;
//
//    *p = 3.14159;
//    std::cout << *p << std::endl;
}

}   // px


