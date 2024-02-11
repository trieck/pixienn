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

TEST(BlasTests, MagArray)
{
    constexpr float array[] = { 1, 2, 3, 4, 5 };

    const auto expected = std::sqrt(55.0f);

    auto result = magArray(array, sizeof(array) / sizeof(float));

    EXPECT_FLOAT_EQ(result, expected);
}

TEST(BlasTests, SumArray)
{
    constexpr float array[] = {
            1, 2, 4, 8, 16, 31, 62, 124, 248
    };

    constexpr float expected = 496.0;

    auto result = sumArray(array, sizeof(array) / sizeof(float));

    EXPECT_FLOAT_EQ(result, expected);
}

TEST(BlasTests, BackwardBias)
{
    constexpr auto batch = 1;
    constexpr auto n = 2;
    constexpr auto size = 3;

    PxCpuTensor<1> biasUpdates({ n }, 1.0f);
    PxCpuTensor<3> delta({ batch, n, size }, 0.5f);

    backwardBias(biasUpdates.data(), delta.data(), batch, n, size);

    EXPECT_THAT(biasUpdates.asVector(), ElementsAre(2.5, 2.5));
}

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

TEST(BlasTests, ScaleBias)
{
    constexpr auto batch = 2;
    constexpr auto n = 3;
    constexpr auto size = 4;

    PxCpuTensor<3> output({ batch, n, size }, 2.0f);
    PxCpuTensor<3> bias({ batch, n, size }, 0.2f);

    scaleBias(output.data(), bias.data(), batch, n, size);

    EXPECT_THAT(output.asVector(), Each(FloatEq(0.4f)));
}

TEST(BlasTests, BackwardScaleCpu)
{
    constexpr auto batch = 2;
    constexpr auto n = 3;
    constexpr auto size = 4;

    PxCpuTensor<3> xNorm({ batch, n, size }, 1.0f);
    PxCpuTensor<3> delta({ batch, n, size }, 2.0f);
    PxCpuTensor<1> scaleUpdates({ n }, 0.0f);

    backwardScaleCpu(xNorm.data(), delta.data(), batch, n, size, scaleUpdates.data());

    EXPECT_THAT(scaleUpdates.asVector(), Each(FloatEq(16.0f)));
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

    PxCpuTensor<1> expected({ filters }, { 90.85714, 125.14286, 168.57144 });

    PxCpuTensor<1> variance({ filters });
    varianceCpu(input.data(), mean.data(), batch, filters, spatial, variance.data());

    EXPECT_THAT(variance.asVector(), Pointwise(FloatNear(1e-4), expected.asVector()));
}

TEST(BlasTests, MeanDeltaCpu)
{
    constexpr int batch = 2;
    constexpr int filters = 3;
    constexpr int spatial = 4;

    PxCpuTensor<3> delta({ batch, filters, spatial },
                         { 1.0f, 2.0, 3.0, 4.0,
                           5.0, 6.0, 7.0, 8.0,
                           9.0, 10.0, 11.0, 12.0,
                           13.0, 14.0, 15.0, 16.0,
                           17.0, 18.0, 19.0, 20.0,
                           21.0, 22.0, 23.0, 24.0 });

    PxCpuTensor<1> variance({ 3 }, { 2.0f, 3.0f, 4.0f });
    PxCpuTensor<1> expected({ 3 }, { -48.0831, -57.7349, -65.9999 });
    PxCpuTensor<1> output({ 3 });

    meanDeltaCpu(delta.data(), variance.data(), batch, filters, spatial, output.data());

    EXPECT_THAT(output.asVector(), Pointwise(FloatNear(1e-4), expected.asVector()));
}

TEST(BlasTests, VarianceDeltaCpu)
{
    constexpr int batch = 1;
    constexpr int filters = 2;
    constexpr int spatial = 3;

    PxCpuTensor<3> x({ batch, filters, spatial }, { 1, 2, 3, 4, 5, 6 });
    PxCpuTensor<3> delta({ batch, filters, spatial }, { 2, 3, 4, 5, 6, 7 });
    PxCpuTensor<1> mean({ filters }, { 0.5, 0.6 });
    PxCpuTensor<1> variance({ filters }, { 0.7, 0.8 });
    PxCpuTensor<1> expected({ filters }, { -13.2326, -56.7392 });
    PxCpuTensor<1> varianceDelta({ filters });

    varianceDeltaCpu(x.data(), delta.data(), mean.data(), variance.data(), batch, filters, spatial,
                     varianceDelta.data());

    EXPECT_THAT(varianceDelta.asVector(), Pointwise(FloatNear(1e-4), expected.asVector()));
}

TEST(BlasTests, NormalizeDeltaCpu)
{
    constexpr int batch = 2;
    constexpr int filters = 2;
    constexpr int spatial = 2;

    PxCpuTensor<3> x({ batch, filters, spatial }, { 1.0f, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
    PxCpuTensor<1> mean({ filters }, { 10.0, 12.0 });
    PxCpuTensor<1> variance({ filters }, { 4.0f, 6.0f });
    PxCpuTensor<1> meanDelta({ filters }, { -1.0, 0.5 });
    PxCpuTensor<1> varianceDelta({ filters }, { 0.1, -0.23 });
    PxCpuTensor<3> delta({ batch, filters, spatial }, 0.0f);

    PxCpuTensor<3> expected({ batch, filters, spatial }, { -0.7, -0.65, 1.16, 1.045, -0.5, -0.45, 0.7, 0.585 });

    normalizeDeltaCpu(x.data(), mean.data(), variance.data(), meanDelta.data(),
                      varianceDelta.data(), batch, filters, spatial, delta.data());

    EXPECT_THAT(delta.asVector(), Pointwise(FloatNear(1e-4), expected.asVector()));
}

TEST(BlasTests, Im2ColCpu)
{
    constexpr int channels = 2;
    constexpr int height = 3;
    constexpr int width = 3;
    constexpr int ksize = 2;
    constexpr int stride = 1;
    constexpr int pad = 0;

    PxCpuTensor<3> input({ channels, height, width }, {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f
    });

    PxCpuTensor<3> expected({ channels * ksize * ksize,
                              (height - ksize + 2 * pad) / stride + 1,
                              (width - ksize + 2 * pad) / stride + 1 },
                            { 1.0f, 2.0f, 4.0f, 5.0f,
                              2.0f, 3.0f, 5.0f, 6.0f,
                              4.0f, 5.0f, 7.0f, 8.0f,
                              5.0f, 6.0f, 8.0f, 9.0f,
                              10.0f, 11.0f, 13.0f, 14.0f,
                              11.0f, 12.0f, 14.0f, 15.0f,
                              13.0f, 14.0f, 16.0f, 17.0f,
                              14.0f, 15.0f, 17.0f, 18.0f });

    PxCpuTensor<3> result(expected.shape(), 0.0f);

    im2ColCpu(input.data(), channels, height, width, ksize, stride, pad, result.data());

    EXPECT_THAT(result.asVector(), Pointwise(FloatNear(1e-4), expected.asVector()));
}

TEST(BlasTests, Col2ImCpu)
{
    constexpr int channels = 2;
    constexpr int height = 2;
    constexpr int width = 2;
    constexpr int ksize = 2;
    constexpr int stride = 1;
    constexpr int pad = 0;

    PxCpuTensor<3> input({ channels, height, width }, {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    });

    PxCpuTensor<3> colResult({ channels * ksize * ksize,
                               (height - ksize + 2 * pad) / stride + 1,
                               (width - ksize + 2 * pad) / stride + 1 }, 0.0f);

    im2ColCpu(input.data(), channels, height, width, ksize, stride, pad, colResult.data());

    PxCpuTensor<3> col2ImResult(input.shape(), 0.0f);
    col2ImCpu(colResult.data(), channels, height, width, ksize, stride, pad, col2ImResult.data());

    EXPECT_THAT(col2ImResult.asVector(), Pointwise(FloatNear(1e-4), input.asVector()));
}
