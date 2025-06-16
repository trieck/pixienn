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

#include "Math.h"
#include "PxTensor.h"

using namespace px;
using namespace testing;

TEST(MathSuite, Softmax)
{
    PxCpuVector input = { 1.0f, 2.0f, 3.0f };
    PxCpuVector result = softmax(input);

    PxCpuVector expected = {
            std::exp(1.0f - 3.0f) / (std::exp(1.0f - 3.0f) + std::exp(2.0f - 3.0f) + 1.0f),
            std::exp(2.0f - 3.0f) / (std::exp(1.0f - 3.0f) + std::exp(2.0f - 3.0f) + 1.0f),
            std::exp(3.0f - 3.0f) / (std::exp(1.0f - 3.0f) + std::exp(2.0f - 3.0f) + 1.0f)
    };

    EXPECT_THAT(result, ElementsAre(FloatNear(expected[0], 1e-6),
                                    FloatNear(expected[1], 1e-6),
                                    FloatNear(expected[2], 1e-6)));
}

TEST(MathSuite, Sigmoid)
{
    PxCpuVector input = { -2.0f, 0.0f, 2.0f };
    PxCpuVector result = sigmoid(input);

    PxCpuVector expected = {
            1.0f / (1.0f + std::exp(2.0f)),   // sigmoid(-2)
            0.5f,                             // sigmoid(0)
            1.0f / (1.0f + std::exp(-2.0f))   // sigmoid(2)
    };

    EXPECT_THAT(result, ElementsAre(FloatNear(expected[0], 1e-6),
                                    FloatNear(expected[1], 1e-6),
                                    FloatNear(expected[2], 1e-6)));
}
