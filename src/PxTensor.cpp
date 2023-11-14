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

#include "PxTensor.h"

using namespace px;

using CpuVector = PxCpuVectorT<float>;
using CudaVector = PxCudaVectorT<float>;

///////////////////////////////////////////////////////////////////////////////
void foobar()
{
    PxCudaTensor Z0{ 23.0f, 91.0f, 113.7f };
    auto q = Z0.asVector();
    std::copy(q.begin(), q.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << std::endl;

    PxCpuTensor Z1{ 23.0f, 91.0f, 113.7f };
    std::copy(Z1.begin(), Z1.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << std::endl;

    CpuVector s(100, 7.0f);
    assert(s.size() == 100);

    CudaVector u(100, 7.0f);
    assert(u.size() == 100);

    CudaVector v(u);
    assert(v.size() == 100);

    CudaVector w(std::move(v));
    assert(w.size() == 100);
    assert(v.size() == 0);

    CudaVector x{ 1.0f, 2.0f, 3.0f };
    assert(x.size() == 3);

    auto y = x.asVector();
    std::copy(y.begin(), y.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << std::endl;

    auto z0 = cpuTensor(std::initializer_list<float>{ 1, 2, 3 });

    printf("CPU tensor has size: %zu, data is %p.\n", z0->size(), z0->data());
    auto V = z0->asVector();
    std::copy(V.begin(), V.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << std::endl;

    auto z1 = cudaTensor(std::initializer_list<float>{ 4, 5, 6 });

    printf("CUDA tensor has size: %zu, data is %p.\n", z1->size(), z1->data());
    V = z1->asVector();
    std::copy(V.begin(), V.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << std::endl;

    auto z2 = cudaTensor(100);
    printf("CUDA tensor has size: %zu, data is %p.\n", z2->size(), z2->data());

    auto z3 = cpuTensor(100);
    printf("CPU tensor has size: %zu, data is %p.\n", z3->size(), z3->data());


}

