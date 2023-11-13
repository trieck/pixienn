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

using CudaVector = PxCudaVectorT<float>;

///////////////////////////////////////////////////////////////////////////////
void foobar()
{
    CudaVector u(100, 7.0f);
    assert(u.size() == 100);

    CudaVector v(u);
    assert(v.size() == 100);

    CudaVector w(std::move(v));
    assert(w.size() == 100);
    assert(v.size() == 0);

    CudaVector x{ 1.0f, 2.0f, 3.0f };
    assert(x.size() == 3);

    auto y = x.toHost();
    std::copy(y.begin(), y.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << std::endl;

    PxCudaTensor Z{ 23.0f, 91.0f, 113.7f };

    auto q = Z.toHost();
    std::copy(q.begin(), q.end(), std::ostream_iterator<float>(std::cout, ", "));
}
