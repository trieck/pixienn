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

#include "CpuUtil.h"
#include "PxTensor.h"

using namespace px;
using namespace testing;

TEST(BlasTests, AddBias)
{
    constexpr auto batch = 2;
    constexpr auto n = 3;
    constexpr auto size = 4;

    PxCpuTensor<3> output({ batch, n, size }, 1.0f);
    PxCpuTensor<1> bias({ n }, 0.5f);

    addBias(output.data(), bias.data(), batch, n, size);

    EXPECT_THAT(output.asVector(), Each(FloatEq(1.5f)));
}

TEST(BlasTests, MeanCpu)
{
    constexpr auto batch = 2;
    constexpr auto filters = 1;
    constexpr auto spatial = 3;

    PxCpuTensor<3> input({ batch, filters, spatial }, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
    PxCpuTensor<1> mean({ filters });

    meanCpu(input.data(), batch, filters, spatial, mean.data());

    EXPECT_THAT(mean.asVector(), Each(FloatEq(3.5f)));
}

TEST(BlasTests, VarianceCpu)
{
    constexpr auto batch = 2;
    constexpr auto filters = 3;
    constexpr auto spatial = 4;

    PxCpuTensor<3> input({ batch, filters, spatial },
                         { 1.0f, 2.0f, 3.0f, 4.0f,
                           5.0f, 6.0f, 7.0f, 8.0f,
                           9.0f, 10.0f, 11.0f, 12.0f,
                           13.0f, 14.0f, 15.0f, 16.0f,
                           17.0f, 18.0f, 19.0f, 20.0f,
                           21.0f, 22.0f, 23.0f, 24.0f });

    PxCpuTensor<1> mean({ filters }, { 2.0f, 4.0f, 6.0f });

    PxCpuTensor<1> expectedVariance({ filters }, { 90.85714, 125.14286, 168.57144 });

    PxCpuTensor<1> variance({ filters });
    varianceCpu(input.data(), mean.data(), batch, filters, spatial, variance.data());

    EXPECT_THAT(variance.asVector(), Pointwise(FloatNear(1e-4), expectedVariance.asVector()));
}