/********************************************************************************
* Copyright 2020-2023 Thomas A. Rieck, All Rights Reserved
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

GTEST_TEST(TensorSuite, CpuTensor)
{
    constexpr std::size_t size(100);

    PxCpuTensor<1> tensor({ size }, 7.0f);
    ASSERT_EQ(tensor.size(), size);
    EXPECT_THAT(tensor.asVector(), Each(FloatEq(7.0f)));
    EXPECT_THAT(tensor.shape(), ElementsAre(100));
    EXPECT_THAT(tensor.strides(), ElementsAre(1));
}

GTEST_TEST(TensorSuite, CpuTensor3D)
{
    PxCpuTensor<3> tensor({ 2, 2, 2 }, 7.0f);

    ASSERT_EQ(tensor.size(), 8);
    EXPECT_THAT(tensor.asVector(), Each(FloatEq(7.0f)));
    EXPECT_THAT(tensor.shape(), ElementsAre(2, 2, 2));
    EXPECT_THAT(tensor.strides(), ElementsAre(4, 2, 1));
}

GTEST_TEST(TensorSuite, MakeCpuTensor3D)
{
    std::array<std::size_t, 3> shape{ 2, 2, 2 };

    auto tensor = cpuTensor<3>(shape);

    ASSERT_EQ(tensor->device(), Device::CPU);
    ASSERT_EQ(tensor->size(), 8);
    EXPECT_THAT(tensor->asVector(), Each(FloatEq(0.0f)));
    EXPECT_THAT(tensor->shape(), ElementsAre(2, 2, 2));
    EXPECT_THAT(tensor->strides(), ElementsAre(4, 2, 1));
}

#ifdef USE_CUDA

GTEST_TEST(TensorSuite, CudaTensor)
{
    constexpr std::size_t size(100);

    PxCudaTensor<1> tensor({ size }, 7.0f);
    ASSERT_EQ(tensor.size(), size);
    EXPECT_THAT(tensor.asVector(), Each(FloatEq(7.0f)));
    EXPECT_THAT(tensor.shape(), ElementsAre(100));
    EXPECT_THAT(tensor.strides(), ElementsAre(1));
}

GTEST_TEST(TensorSuite, CudaTensor3D)
{
    PxCudaTensor<3> tensor({ 2, 2, 2 }, 7.0f);
    ASSERT_EQ(tensor.size(), 8);
    EXPECT_THAT(tensor.asVector(), Each(FloatEq(7.0f)));
    EXPECT_THAT(tensor.shape(), ElementsAre(2, 2, 2));
    EXPECT_THAT(tensor.strides(), ElementsAre(4, 2, 1));
}

GTEST_TEST(TensorSuite, MakeCudaTensor3D)
{
    std::array<std::size_t, 3> shape{ 2, 2, 2 };

    auto tensor = cudaTensor<3>(shape, 0.0f);

    ASSERT_EQ(tensor->device(), Device::CUDA);
    ASSERT_EQ(tensor->size(), 8);
    EXPECT_THAT(tensor->asVector(), Each(FloatEq(0.0f)));
    EXPECT_THAT(tensor->shape(), ElementsAre(2, 2, 2));
    EXPECT_THAT(tensor->strides(), ElementsAre(4, 2, 1));
}

#endif // USE_CUDA

GTEST_TEST(TensorSuite, Strides2D)
{
    constexpr size_t NDIMS = 2;

    std::array<std::size_t, NDIMS> shape{ 3, 4 };
    std::array<std::size_t, NDIMS> strides;

    compute_strides(shape, strides);

    ASSERT_THAT(strides, ElementsAre(4, 1));
}

GTEST_TEST(TensorSuite, Strides3D)
{
    constexpr size_t NDIMS = 3;

    std::array<std::size_t, NDIMS> shape{ 1, 2, 3 };
    std::array<std::size_t, NDIMS> strides;

    compute_strides(shape, strides);

    ASSERT_THAT(strides, ElementsAre(0, 3, 1));
}

GTEST_TEST(TensorSuite, Strides4D)
{
    constexpr size_t NDIMS = 4;

    std::array<std::size_t, NDIMS> shape{ 1, 3, 224, 224 };
    std::array<std::size_t, NDIMS> strides;

    compute_strides(shape, strides);

    ASSERT_THAT(strides, ElementsAre(0, 50176, 224, 1));
}

