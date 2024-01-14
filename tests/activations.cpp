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
#include "Device.h"

using namespace px;
using namespace testing;

template<Device D>
struct DeviceTrait;

template<>
struct DeviceTrait<Device::CPU>
{
    static constexpr auto D = Device::CPU;

    using V = PxCpuVector;
};

template<>
struct DeviceTrait<Device::CUDA>
{
    static constexpr auto D = Device::CUDA;

    using V = PxCudaVector;
};

#ifdef USE_CUDA
using DeviceTypes = ::testing::Types<DeviceTrait<Device::CPU>, DeviceTrait<Device::CUDA>>;
#else
using DeviceTypes = ::testing::Types<DeviceTrait<Device::CPU>>;
#endif

/////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class LeakyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations<T::D>::get("leaky");
    }

    Activations<T::D>::Ptr activation;
};

TYPED_TEST_SUITE(LeakyTest, DeviceTypes);

TYPED_TEST(LeakyTest, Apply)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };

    this->activation->apply(input);

    std::vector<float> results = input.asVector();

    ASSERT_FLOAT_EQ(results[0], 1.0f);
    ASSERT_FLOAT_EQ(results[1], -0.2f);
    ASSERT_FLOAT_EQ(results[2], 3.0f);
    ASSERT_FLOAT_EQ(results[3], -0.4f);
}

TYPED_TEST(LeakyTest, Gradient)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };
    typename TypeParam::V delta{ 0.1f, -0.1f, 0.2f, -0.2f };

    this->activation->gradient(input, delta);

    std::vector<float> results = delta.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.1f);
    ASSERT_FLOAT_EQ(results[1], -0.01f);
    ASSERT_FLOAT_EQ(results[2], 0.2f);
    ASSERT_FLOAT_EQ(results[3], -0.02f);
}

/////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class LinearTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations<T::D>::get("linear");
    }

    Activations<T::D>::Ptr activation;
};

TYPED_TEST_SUITE(LinearTest, DeviceTypes);

TYPED_TEST(LinearTest, Apply)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };

    this->activation->apply(input);

    std::vector<float> results = input.asVector();

    ASSERT_FLOAT_EQ(results[0], 1.0f);
    ASSERT_FLOAT_EQ(results[1], -2.0f);
    ASSERT_FLOAT_EQ(results[2], 3.0f);
    ASSERT_FLOAT_EQ(results[3], -4.0f);
}

TYPED_TEST(LinearTest, Gradient)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };
    typename TypeParam::V delta{ 0.1f, -0.1f, 0.2f, -0.2f };

    this->activation->gradient(input, delta);

    std::vector<float> results = delta.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.1f);
    ASSERT_FLOAT_EQ(results[1], -0.1f);
    ASSERT_FLOAT_EQ(results[2], 0.2f);
    ASSERT_FLOAT_EQ(results[3], -0.2f);
}

/////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class LoggyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations<T::D>::get("loggy");
    }

    Activations<T::D>::Ptr activation;
};

TYPED_TEST_SUITE(LoggyTest, DeviceTypes);

TYPED_TEST(LoggyTest, Apply)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };

    this->activation->apply(input);

    std::vector<float> results = input.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.4621172);
    ASSERT_FLOAT_EQ(results[1], -0.76159418);
    ASSERT_FLOAT_EQ(results[2], 0.90514827);
    ASSERT_FLOAT_EQ(results[3], -0.9640276);
}

TYPED_TEST(LoggyTest, Gradient)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };
    typename TypeParam::V delta{ 0.1f, -0.1f, 0.2f, -0.2f };

    this->activation->gradient(input, delta);

    std::vector<float> results = delta.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.0);
    ASSERT_FLOAT_EQ(results[1], 0.15);
    ASSERT_FLOAT_EQ(results[2], -0.8);
    ASSERT_FLOAT_EQ(results[3], 1.5);
}

/////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class LogisticTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations<T::D>::get("logistic");
    }

    Activations<T::D>::Ptr activation;
};

class LogisticActivationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations<>::get("logistic");
    }

    Activations<>::Ptr activation;
};

TYPED_TEST_SUITE(LogisticTest, DeviceTypes);

TYPED_TEST(LogisticTest, Apply)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };

    this->activation->apply(input);

    std::vector<float> results = input.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.7310586);
    ASSERT_FLOAT_EQ(results[1], 0.11920292);
    ASSERT_FLOAT_EQ(results[2], 0.95257413);
    ASSERT_FLOAT_EQ(results[3], 0.017986209);
}

TYPED_TEST(LogisticTest, Gradient)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };
    typename TypeParam::V delta{ 0.1f, -0.1f, 0.2f, -0.2f };

    this->activation->gradient(input, delta);

    std::vector<float> results = delta.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.019661194);
    ASSERT_FLOAT_EQ(results[1], -0.010499358);
    ASSERT_FLOAT_EQ(results[2], 0.0090353312);
    ASSERT_FLOAT_EQ(results[3], -0.0035325412);
}

/////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class MishTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations<T::D>::get("mish");
    }

    Activations<T::D>::Ptr activation;
};

TYPED_TEST_SUITE(MishTest, DeviceTypes);

TYPED_TEST(MishTest, Apply)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };

    this->activation->apply(input);

    std::vector<float> results = input.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.86509836f);
    ASSERT_FLOAT_EQ(results[1], -0.25250155f);
    ASSERT_FLOAT_EQ(results[2], 2.9865351f);
    ASSERT_FLOAT_EQ(results[3], -0.072591871f);
}

TYPED_TEST(MishTest, Gradient)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };
    typename TypeParam::V delta{ 0.1f, -0.1f, 0.2f, -0.2f };

    this->activation->gradient(input, delta);

    std::vector<float> results = delta.asVector();

    EXPECT_NEAR(results[0], 0.10490363f, 1e-5);
    EXPECT_NEAR(results[0], 0.10490363f, 1e-5);
    EXPECT_NEAR(results[1], 0.010835516f, 1e-5);
    EXPECT_NEAR(results[2], 0.2042214f, 1e-5);
    EXPECT_NEAR(results[3], 0.010754657f, 1e-5);
}

/////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class ReluTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations<T::D>::get("relu");
    }

    Activations<T::D>::Ptr activation;
};

TYPED_TEST_SUITE(ReluTest, DeviceTypes);

TYPED_TEST(ReluTest, Apply)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };

    this->activation->apply(input);

    std::vector<float> results = input.asVector();

    ASSERT_FLOAT_EQ(results[0], 1.0f);
    ASSERT_FLOAT_EQ(results[1], 0.0f);
    ASSERT_FLOAT_EQ(results[2], 3.0f);
    ASSERT_FLOAT_EQ(results[3], 0.0f);
}

TYPED_TEST(ReluTest, Gradient)
{
    typename TypeParam::V input{ 1.0f, -2.0f, 3.0f, -4.0f };
    typename TypeParam::V delta{ 0.1f, -0.1f, 0.2f, -0.2f };

    this->activation->gradient(input, delta);

    std::vector<float> results = delta.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.1f);
    ASSERT_FLOAT_EQ(results[1], 0.0f);
    ASSERT_FLOAT_EQ(results[2], 0.2f);
    ASSERT_FLOAT_EQ(results[3], 0.0f);
}

/////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class SoftplusTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations<T::D>::get("softplus");
    }

    Activations<T::D>::Ptr activation;
};

TYPED_TEST_SUITE(SoftplusTest, DeviceTypes);

TYPED_TEST(SoftplusTest, Apply)
{
    typename TypeParam::V input{ 1.0f, 2.0f, 3.0f, 4.0f, 21.0f, -21.0f };

    this->activation->apply(input);

    std::vector<float> results = input.asVector();

    ASSERT_FLOAT_EQ(results[0], 1.3132616f);
    ASSERT_FLOAT_EQ(results[1], 2.1269281f);
    ASSERT_FLOAT_EQ(results[2], 3.0485873f);
    ASSERT_FLOAT_EQ(results[3], 4.0181499f);
    ASSERT_FLOAT_EQ(results[4], 21.0f);
    ASSERT_FLOAT_EQ(results[5], 7.5825607e-10f);
}

TYPED_TEST(SoftplusTest, Gradient)
{
    typename TypeParam::V input{ 1.0f, 2.0f, 3.0f, 4.0f, 21.0f, -21.0f };
    typename TypeParam::V delta{ 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, -0.5f };

    this->activation->gradient(input, delta);

    std::vector<float> results = delta.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.073105864);
    ASSERT_FLOAT_EQ(results[1], 0.17615943);
    ASSERT_FLOAT_EQ(results[2], 0.28577226);
    ASSERT_FLOAT_EQ(results[3], 0.39280552);
    ASSERT_FLOAT_EQ(results[4], 0.5f);
    ASSERT_FLOAT_EQ(results[5], -3.7912801e-10f);
}

/////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class SwishTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations<T::D>::get("swish");
    }

    Activations<T::D>::Ptr activation;
};

TYPED_TEST_SUITE(SwishTest, DeviceTypes);

TYPED_TEST(SwishTest, Apply)
{
    typename TypeParam::V input{ 1.0f, 2.0f, 3.0f, 4.0f };

    this->activation->apply(input);

    std::vector<float> results = input.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.7310586f);
    ASSERT_FLOAT_EQ(results[1], 1.7615942f);
    ASSERT_FLOAT_EQ(results[2], 2.857722f);
    ASSERT_FLOAT_EQ(results[3], 3.928055f);
}

TYPED_TEST(SwishTest, Gradient)
{
    typename TypeParam::V input{ 1.0f, 2.0f, 3.0f, 4.0f };
    typename TypeParam::V delta{ 0.1f, 0.2f, 0.3f, 0.4f };

    this->activation->gradient(input, delta);

    std::vector<float> results = delta.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.092767052f);
    ASSERT_FLOAT_EQ(results[1], 0.21815686f);
    ASSERT_FLOAT_EQ(results[2], 0.32643124f);
    ASSERT_FLOAT_EQ(results[3], 0.4210659f);
}

/////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class TanhTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        activation = Activations<T::D>::get("tanh");
    }

    Activations<T::D>::Ptr activation;
};

TYPED_TEST_SUITE(TanhTest, DeviceTypes);

TYPED_TEST(TanhTest, Apply)
{
    typename TypeParam::V input{ 1.0f, 2.0f, 3.0f, 4.0f };

    this->activation->apply(input);

    std::vector<float> results = input.asVector();

    ASSERT_FLOAT_EQ(results[0], 0.7615942f);
    ASSERT_FLOAT_EQ(results[1], 0.9640276f);
    ASSERT_FLOAT_EQ(results[2], 0.9950547f);
    ASSERT_FLOAT_EQ(results[3], 0.9993293f);
}

TYPED_TEST(TanhTest, Gradient)
{
    typename TypeParam::V input{ 1.0f, 2.0f, 3.0f, 4.0f };
    typename TypeParam::V delta{ 0.1f, 0.2f, 0.3f, 0.4f };

    this->activation->gradient(input, delta);

    std::vector<float> results = delta.asVector();

    EXPECT_NEAR(results[0], 0.041997429f, 1e-5);
    EXPECT_NEAR(results[1], 0.014130163f, 1e-5);
    EXPECT_NEAR(results[2], 0.0029597946f, 1e-5);
    EXPECT_NEAR(results[3], 0.00053635723f, 1e-5);
}
