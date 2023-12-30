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

#include "Activation.h"

using namespace px;
using namespace testing;

class LeakyActivationTest : public ::testing::Test
{
protected:
    void SetUp() override;

    Activations::Ptr activation;
};

void LeakyActivationTest::SetUp()
{
    activation = Activations::get("leaky");
}

TEST_F(LeakyActivationTest, Apply)
{
    PxCpuVector input = { 1.0f, -2.0f, 3.0f, -4.0f };

    activation->apply(input);

    ASSERT_FLOAT_EQ(input[0], 1.0f);
    ASSERT_FLOAT_EQ(input[1], -0.2f);
    ASSERT_FLOAT_EQ(input[2], 3.0f);
    ASSERT_FLOAT_EQ(input[3], -0.4f);
}

TEST_F(LeakyActivationTest, Gradient)
{
    PxCpuVector input = { 1.0f, -2.0f, 3.0f, -4.0f };
    PxCpuVector delta = { 0.1f, -0.1f, 0.2f, -0.2f };

    activation->gradient(input, delta);

    ASSERT_FLOAT_EQ(delta[0], 0.1f);
    ASSERT_FLOAT_EQ(delta[1], -0.01f);
    ASSERT_FLOAT_EQ(delta[2], 0.2f);
    ASSERT_FLOAT_EQ(delta[3], -0.02f);
}

class LoggyActivationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations::get("loggy");
    }

    Activations::Ptr activation;
};

TEST_F(LoggyActivationTest, Apply)
{
    PxCpuVector input = { 1.0f, -2.0f, 3.0f, -4.0f };

    activation->apply(input);

    ASSERT_FLOAT_EQ(input[0], 0.4621172);
    ASSERT_FLOAT_EQ(input[1], -0.76159418);
    ASSERT_FLOAT_EQ(input[2], 0.90514827);
    ASSERT_FLOAT_EQ(input[3], -0.9640276);
}

TEST_F(LoggyActivationTest, Gradient)
{
    PxCpuVector input = { 1.0f, -2.0f, 3.0f, -4.0f };
    PxCpuVector delta = { 0.1f, -0.1f, 0.2f, -0.2f };

    activation->gradient(input, delta);

    ASSERT_FLOAT_EQ(delta[0], 0.0);
    ASSERT_FLOAT_EQ(delta[1], 0.15);
    ASSERT_FLOAT_EQ(delta[2], -0.8);
    ASSERT_FLOAT_EQ(delta[3], 1.5);
}

class LogisticActivationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations::get("logistic");
    }

    Activations::Ptr activation;
};

TEST_F(LogisticActivationTest, Apply)
{
    PxCpuVector input = { 1.0f, -2.0f, 3.0f, -4.0f };

    activation->apply(input);

    ASSERT_FLOAT_EQ(input[0], 0.7310586);
    ASSERT_FLOAT_EQ(input[1], 0.11920292);
    ASSERT_FLOAT_EQ(input[2], 0.95257413);
    ASSERT_FLOAT_EQ(input[3], 0.01798621);
}

TEST_F(LogisticActivationTest, Gradient)
{
    PxCpuVector input = { 1.0f, -2.0f, 3.0f, -4.0f };
    PxCpuVector delta = { 0.1f, -0.1f, 0.2f, -0.2f };

    activation->gradient(input, delta);

    ASSERT_FLOAT_EQ(delta[0], 0.019661194);
    ASSERT_FLOAT_EQ(delta[1], -0.010499358);
    ASSERT_FLOAT_EQ(delta[2], 0.0090353312);
    ASSERT_FLOAT_EQ(delta[3], -0.0035325412);
}

class MishActivationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations::get("mish");
    }

    Activations::Ptr activation;
};

TEST_F(MishActivationTest, Apply)
{
    PxCpuVector input = { 1.0f, -2.0f, 3.0f, -4.0f };

    activation->apply(input);

    ASSERT_FLOAT_EQ(input[0], 0.86509836f);
    ASSERT_FLOAT_EQ(input[1], -0.25250155f);
    ASSERT_FLOAT_EQ(input[2], 2.9865351f);
    ASSERT_FLOAT_EQ(input[3], -0.072591871f);
}

TEST_F(MishActivationTest, Gradient)
{
    PxCpuVector input = { 1.0f, -2.0f, 3.0f, -4.0f };
    PxCpuVector delta = { 0.1f, -0.1f, 0.2f, -0.2f };

    activation->gradient(input, delta);

    ASSERT_FLOAT_EQ(delta[0], 0.10490363f);
    ASSERT_FLOAT_EQ(delta[1], 0.010835516f);
    ASSERT_FLOAT_EQ(delta[2], 0.2042214f);
    ASSERT_FLOAT_EQ(delta[3], 0.010754657f);
}

class ReluActivationTest : public ::testing::Test
{
protected:
    void SetUp() override;

    Activations::Ptr activation;
};

void ReluActivationTest::SetUp()
{
    activation = Activations::get("relu");
}

TEST_F(ReluActivationTest, Apply)
{
    PxCpuVector input = { 1.0f, -2.0f, 3.0f, -4.0f };

    activation->apply(input);

    ASSERT_FLOAT_EQ(input[0], 1.0f);
    ASSERT_FLOAT_EQ(input[1], 0.0f);
    ASSERT_FLOAT_EQ(input[2], 3.0f);
    ASSERT_FLOAT_EQ(input[3], 0.0f);
}

TEST_F(ReluActivationTest, Gradient)
{
    PxCpuVector input = { 1.0f, -2.0f, 3.0f, -4.0f };
    PxCpuVector delta = { 0.1f, -0.1f, 0.2f, -0.2f };

    activation->gradient(input, delta);

    ASSERT_FLOAT_EQ(delta[0], 0.1f);
    ASSERT_FLOAT_EQ(delta[1], 0.0f);
    ASSERT_FLOAT_EQ(delta[2], 0.2f);
    ASSERT_FLOAT_EQ(delta[3], 0.0f);
}

class SoftplusActivationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations::get("softplus");
    }

    Activations::Ptr activation;
};

TEST_F(SoftplusActivationTest, Apply)
{
    PxCpuVector input = { 1.0f, 2.0f, 3.0f, 4.0f, 21.0f, -21.0f };

    activation->apply(input);

    ASSERT_FLOAT_EQ(input[0], 1.3132616f);
    ASSERT_FLOAT_EQ(input[1], 2.1269281f);
    ASSERT_FLOAT_EQ(input[2], 3.0485873f);
    ASSERT_FLOAT_EQ(input[3], 4.0181499f);
    ASSERT_FLOAT_EQ(input[4], 21.0f);
    ASSERT_FLOAT_EQ(input[5], 7.5825607e-10f);
}

TEST_F(SoftplusActivationTest, Gradient)
{
    PxCpuVector input = { 1.0f, 2.0f, 3.0f, 4.0f, 21.0f, -21.0f };
    PxCpuVector delta = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, -0.5f };

    activation->gradient(input, delta);

    ASSERT_FLOAT_EQ(delta[0], 0.073105864);
    ASSERT_FLOAT_EQ(delta[1], 0.17615943);
    ASSERT_FLOAT_EQ(delta[2], 0.28577226);
    ASSERT_FLOAT_EQ(delta[3], 0.39280552);
    ASSERT_FLOAT_EQ(delta[4], 0.5f);
    ASSERT_FLOAT_EQ(delta[5], -3.7912801e-10f);
}

class SwishActivationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations::get("swish");
    }

    Activations::Ptr activation;
};

TEST_F(SwishActivationTest, Apply)
{
    PxCpuVector input = { 1.0f, 2.0f, 3.0f, 4.0f };

    activation->apply(input);

    ASSERT_FLOAT_EQ(input[0], 0.7310586f);
    ASSERT_FLOAT_EQ(input[1], 1.7615942f);
    ASSERT_FLOAT_EQ(input[2], 2.857722f);
    ASSERT_FLOAT_EQ(input[3], 3.928055f);
}

TEST_F(SwishActivationTest, Gradient)
{
    PxCpuVector input = { 1.0f, 2.0f, 3.0f, 4.0f };
    PxCpuVector delta = { 0.1f, 0.2f, 0.3f, 0.4f };

    activation->gradient(input, delta);

    ASSERT_FLOAT_EQ(delta[0], 0.092767052f);
    ASSERT_FLOAT_EQ(delta[1], 0.21815686f);
    ASSERT_FLOAT_EQ(delta[2], 0.32643124f);
    ASSERT_FLOAT_EQ(delta[3], 0.4210659f);
}

class TanhActivationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations::get("tanh");
    }

    Activations::Ptr activation;
};

TEST_F(TanhActivationTest, Apply)
{
    PxCpuVector input = { 1.0f, 2.0f, 3.0f, 4.0f };

    activation->apply(input);

    ASSERT_FLOAT_EQ(input[0], 0.7615942f);
    ASSERT_FLOAT_EQ(input[1], 0.9640276f);
    ASSERT_FLOAT_EQ(input[2], 0.9950547f);
    ASSERT_FLOAT_EQ(input[3], 0.9993293f);
}

TEST_F(TanhActivationTest, Gradient)
{
    PxCpuVector input = { 1.0f, 2.0f, 3.0f, 4.0f };
    PxCpuVector delta = { 0.1f, 0.2f, 0.3f, 0.4f };

    activation->gradient(input, delta);

    ASSERT_FLOAT_EQ(delta[0], 0.041997429f);
    ASSERT_FLOAT_EQ(delta[1], 0.014130163f);
    ASSERT_FLOAT_EQ(delta[2], 0.0029597946f);
    ASSERT_FLOAT_EQ(delta[3], 0.00053635723f);
}
