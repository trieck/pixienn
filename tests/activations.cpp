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

    Activation::Ptr activation;
};

void LeakyActivationTest::SetUp()
{
    activation = Activation::get("leaky");
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
        activation = Activation::get("loggy");
    }

    Activation::Ptr activation;
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
        activation = Activation::get("logistic");
    }

    Activation::Ptr activation;
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

    ASSERT_FLOAT_EQ(delta[0], 0);
    ASSERT_FLOAT_EQ(delta[1], 0.6);
    ASSERT_FLOAT_EQ(delta[2], -1.2);
    ASSERT_FLOAT_EQ(delta[3], 4);
}

class ReluActivationTest : public ::testing::Test
{
protected:
    void SetUp() override;

    Activation::Ptr activation;
};

void ReluActivationTest::SetUp()
{
    activation = Activation::get("relu");
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
