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

///////////////////////////////////////////////////////////////////////////////
using namespace px;

void foobar()
{
    cuda_vector u{ 1.0f, 2.0f, 3.0f };
    assert(u.size() == 3);

    for (auto x: u) {
        printf("%.2f\n", x);
    }

    printf("\n");

    cuda_vector v(100);
    assert(v.size() == 100);

    v.randomize();

    for (auto x: v) {
        printf("%.2f\n", x);
    }

    printf("\n");

    v.randomize();

    for (auto i = 0; i < v.size(); ++i) {
        auto x = v[i];
        printf("%.2f\n", x);
    }

    cuda_vector w(v);
    assert(w.size() == 100);

    for (auto x: w) {
        printf("%.2f\n", x);
    }

    v.resize(0);
    assert(v.empty());
}

