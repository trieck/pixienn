/********************************************************************************
* Copyright 2026 Thomas A. Rieck, All Rights Reserved
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

#include <cmath>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Model.h"
#include "PositionalEncoding.h"

using namespace px;
using namespace testing;

namespace {

YAML::Node definition()
{
    return YAML::Load(R"(
        type: positional-encoding
        batch: 1
        channels: 4
        height: 2
        width: 2
    )");
}

} // namespace

TEST(PositionalEncodingSuite, AddsDistinctSpatialInformation)
{
    const PxCpuVector input(16, 0.0f);
    Model<> model;
    CpuPositionalEncoding layer(model, definition());

    layer.forward(input);

    EXPECT_FLOAT_EQ(layer.output()[0], 0.0f);
    EXPECT_FLOAT_EQ(layer.output()[2], 1.0f);
    EXPECT_FLOAT_EQ(layer.output()[4], 0.0f);
    EXPECT_FLOAT_EQ(layer.output()[6], 1.0f);
    EXPECT_NE(layer.output()[0], layer.output()[8]);
    EXPECT_NE(layer.output()[4], layer.output()[5]);
}

TEST(PositionalEncodingSuite, BackwardPassesGradientUnchanged)
{
    const PxCpuVector input(16, 0.0f);
    const PxCpuVector upstream(16, 1.0f);
    Model<> model;
    CpuPositionalEncoding layer(model, definition());
    layer.forward(input);
    layer.delta().copy(upstream);

    PxCpuVector gradient(16, 0.0f);
    layer.backward(input, &gradient);

    EXPECT_THAT(gradient, Each(FloatEq(1.0f)));
}

TEST(PositionalEncodingSuite, IsConstructedThroughModelConfiguration)
{
    const auto modelDefinition = YAML::Load(R"(
        model:
          batch: 1
          channels: 4
          height: 2
          width: 2
          layers:
            - type: positional-encoding
    )");
    Model<> model;
    model.parseModel(modelDefinition);

    ASSERT_EQ(model.layerSize(), 1);
    const auto layer = std::dynamic_pointer_cast<CpuPositionalEncoding>(model.layerAt(0));
    ASSERT_NE(layer, nullptr);
}
