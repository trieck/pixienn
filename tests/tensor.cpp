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

#include <gtest/gtest.h>

#include "PxTensor.h"

using namespace px;

GTEST_TEST(TensorSuite, TensorCpuVector)
{
    constexpr std::size_t size(100);

    PxCpuVector u(size, 7.0f);
    ASSERT_EQ(u.size(), size);

    const auto* ptr = u.data();
    for (auto i = 0; i < size; ++i) {
        const auto& value = *ptr++;
        ASSERT_FLOAT_EQ(value, 7.0f);
    }
}

GTEST_TEST(TensorSuite, TensorCudaVector)
{
    constexpr std::size_t size(100);

    PxCudaVector u(size, 7.0f);
    ASSERT_EQ(u.size(), size);

    auto v = u.asVector();
    for (auto value: v) {
        ASSERT_FLOAT_EQ(value, 7.0f);
    }
}