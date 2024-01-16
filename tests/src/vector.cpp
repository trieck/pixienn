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

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "PxTensor.h"

using namespace px;
using namespace testing;

GTEST_TEST(VectorSuite, CpuVector)
{
    constexpr std::size_t size(100);

    PxCpuVector u(size, 7.0f);
    ASSERT_EQ(u.size(), size);
    EXPECT_THAT(u, Each(FloatEq(7.0f)));
}

GTEST_TEST(VectorSuite, MakeCpuVector)
{
    constexpr std::size_t size(100);

    auto vector = cpuVector(size);

    ASSERT_EQ(vector->device(), Device::CPU);
    ASSERT_EQ(vector->size(), size);
    EXPECT_THAT(vector->asVector(), Each(FloatEq(0.0f)));
}

GTEST_TEST(VectorSuite, MakeCpuVectorVal)
{
    constexpr std::size_t size(100);

    auto vector = cpuVector(size, 7.0f);

    ASSERT_EQ(vector->device(), Device::CPU);
    ASSERT_EQ(vector->size(), size);
    EXPECT_THAT(vector->asVector(), Each(FloatEq(7.0f)));
}

GTEST_TEST(VectorSuite, MakeCpuVectorInit)
{
    auto vector = cpuVector(std::initializer_list<float>{ 1.0f, 2.0f, 3.0f, 4.0f });

    ASSERT_EQ(vector->device(), Device::CPU);
    ASSERT_EQ(vector->size(), 4);
    ASSERT_THAT(vector->asVector(), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
}

#ifdef USE_CUDA

GTEST_TEST(VectorSuite, CudaVector)
{
    constexpr std::size_t size(100);

    PxCudaVector u(size, 7.0f);
    ASSERT_EQ(u.size(), size);
    EXPECT_THAT(u.asVector(), Each(FloatEq(7.0f)));
}

GTEST_TEST(VectorSuite, MakeCudaVector)
{
    constexpr std::size_t size(100);

    auto vector = cudaVector(size, 0.0f);

    ASSERT_EQ(vector->device(), Device::CUDA);
    ASSERT_EQ(vector->size(), size);
    EXPECT_THAT(vector->asVector(), Each(FloatEq(0.0f)));
}

GTEST_TEST(VectorSuite, MakeCudaVectorVal)
{
    constexpr std::size_t size(100);

    auto vector = cudaVector(size, 7.0f);

    ASSERT_EQ(vector->device(), Device::CUDA);
    ASSERT_EQ(vector->size(), size);
    EXPECT_THAT(vector->asVector(), Each(FloatEq(7.0f)));
}

GTEST_TEST(VectorSuite, MakeCudaVectorInit)
{
    auto vector = cudaVector(std::initializer_list<float>{ 1.0f, 2.0f, 3.0f, 4.0f });

    ASSERT_EQ(vector->device(), Device::CUDA);
    ASSERT_EQ(vector->size(), 4);
    ASSERT_THAT(vector->asVector(), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
}

#endif // USE_CUDA
