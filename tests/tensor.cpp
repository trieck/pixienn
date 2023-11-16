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
#include <gmock/gmock.h>

#include "Strides.h"
#include "PxTensor.h"

using namespace px;
using namespace testing;

GTEST_TEST(TensorSuite, CpuVector)
{
    constexpr std::size_t size(100);

    PxCpuVector u(size, 7.0f);
    ASSERT_EQ(u.size(), size);

    for (auto value: u) {
        ASSERT_FLOAT_EQ(value, 7.0f);
    }
}

GTEST_TEST(TensorSuite, CudaVector)
{
    constexpr std::size_t size(100);

    PxCudaVector u(size, 7.0f);
    ASSERT_EQ(u.size(), size);

    auto v = u.asVector();
    for (auto value: v) {
        ASSERT_FLOAT_EQ(value, 7.0f);
    }
}

GTEST_TEST(TensorSuite, CpuTensor)
{
    constexpr std::size_t size(100);

    PxCpuTensor u(size, 7.0f);
    ASSERT_EQ(u.size(), size);

    for (auto value: u) {
        ASSERT_FLOAT_EQ(value, 7.0f);
    }
}

GTEST_TEST(TensorSuite, CudaTensor)
{
    constexpr std::size_t size(100);

    PxCudaTensor u(size, 7.0f);
    ASSERT_EQ(u.size(), size);

    auto v = u.asVector();
    for (auto value: v) {
        ASSERT_FLOAT_EQ(value, 7.0f);
    }
}

GTEST_TEST(TensorSuite, MakeCpuTensor)
{
    constexpr std::size_t size(100);
    auto u = cpuTensor(size, 7.0f);

    ASSERT_EQ(u->size(), size);
}

GTEST_TEST(TensorSuite, MakeCudaTensor)
{
    constexpr std::size_t size(100);
    auto u = cudaTensor(size, 7.0f);

    ASSERT_EQ(u->size(), size);
}

GTEST_TEST(TensorSuite, Strides2D)
{
    constexpr size_t NDIMS = 2;

    std::array<std::size_t, NDIMS> shape{ 3, 4 };
    std::array<std::size_t, NDIMS> strides;

    px::compute_strides(shape, strides);

    ASSERT_THAT(strides, ElementsAre(4, 1));
}

GTEST_TEST(TensorSuite, Strides3D)
{
    constexpr size_t NDIMS = 3;

    std::array<std::size_t, NDIMS> shape{ 1, 2, 3 };
    std::array<std::size_t, NDIMS> strides;

    px::compute_strides(shape, strides);

    ASSERT_THAT(strides, ElementsAre(0, 3, 1));
}

GTEST_TEST(TensorSuite, Strides4D)
{
    constexpr size_t NDIMS = 4;

    std::array<std::size_t, NDIMS> shape{ 1, 3, 224, 224 };
    std::array<std::size_t, NDIMS> strides;

    px::compute_strides(shape, strides);

    ASSERT_THAT(strides, ElementsAre(0, 50176, 224, 1));
}
