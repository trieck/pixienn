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
#include <functional>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Model.h"
#include "LayerFactory.h"
#include "LayerNorm.h"

using namespace px;
using namespace testing;

namespace {

YAML::Node definition()
{
    return YAML::Load(R"(
        type: layernorm
        batch: 1
        channels: 3
        height: 1
        width: 2
        epsilon: 0.00001
    )");
}

float loss(const PxCpuVector& input,
           const PxCpuVector& scales,
           const PxCpuVector& biases,
           const PxCpuVector& delta)
{
    Model<> model;
    CpuLayerNorm layer(model, definition());
    layer.copyScales(scales);
    layer.copyBiases(biases);
    layer.forward(input);

    float result = 0.0f;
    for (std::size_t index = 0; index < delta.size(); ++index) {
        result += layer.output()[index] * delta[index];
    }
    return result;
}

float numericalGradient(const std::function<float(float)>& evaluate, float value)
{
    constexpr float step = 1e-3f;
    return (evaluate(value + step) - evaluate(value - step)) / (2.0f * step);
}

} // namespace

TEST(LayerNormSuite, NormalizesChannelsAtEachSpatialLocation)
{
    const PxCpuVector input = { 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f };
    const PxCpuVector expected = { -1.2247357f, -1.2247447f,
                                   0.0f, 0.0f,
                                   1.2247357f, 1.2247447f };
    Model<> model;
    CpuLayerNorm layer(model, definition());

    layer.forward(input);

    EXPECT_THAT(layer.output(), ElementsAre(FloatNear(expected[0], 1e-5f),
                                             FloatNear(expected[1], 1e-5f),
                                             FloatNear(expected[2], 1e-6f),
                                             FloatNear(expected[3], 1e-6f),
                                             FloatNear(expected[4], 1e-5f),
                                             FloatNear(expected[5], 1e-5f)));
}

TEST(LayerNormSuite, AppliesTrainableScaleAndBias)
{
    const PxCpuVector input = { 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f };
    const PxCpuVector scales = { 2.0f, 0.5f, -1.5f };
    const PxCpuVector biases = { 0.25f, -0.5f, 2.0f };
    Model<> model;
    CpuLayerNorm layer(model, definition());
    layer.copyScales(scales);
    layer.copyBiases(biases);

    layer.forward(input);

    EXPECT_THAT(layer.output(), ElementsAre(FloatNear(-2.1994714f, 1e-5f),
                                             FloatNear(-2.1994850f, 1e-5f),
                                             FloatNear(-0.5f, 1e-6f),
                                             FloatNear(-0.5f, 1e-6f),
                                             FloatNear(0.1628964f, 1e-5f),
                                             FloatNear(0.1628830f, 1e-5f)));
}

TEST(LayerNormSuite, ConstantLocationRemainsFinite)
{
    const PxCpuVector input = { 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f };
    Model<> model;
    CpuLayerNorm layer(model, definition());

    layer.forward(input);

    for (const auto value : layer.output()) EXPECT_TRUE(std::isfinite(value));
}

TEST(LayerNormSuite, BackwardMatchesFiniteDifferences)
{
    const PxCpuVector input = { 0.2f, 1.4f, -1.1f, 0.7f, 2.3f, -0.4f };
    const PxCpuVector scales = { 1.2f, 0.7f, 1.5f };
    const PxCpuVector biases = { 0.1f, -0.2f, 0.4f };
    const PxCpuVector upstream = { 0.3f, -0.2f, -0.8f, 0.4f, 0.5f, 0.9f };
    Model<> model;
    CpuLayerNorm layer(model, definition());
    layer.copyScales(scales);
    layer.copyBiases(biases);
    layer.forward(input);
    layer.delta().copy(upstream);
    layer.backward(input, nullptr);

    for (std::size_t index = 0; index < input.size(); ++index) {
        auto evaluate = [&](float value) {
            auto perturbed = input;
            perturbed[index] = value;
            return loss(perturbed, scales, biases, upstream);
        };
        EXPECT_NEAR(layer.delta()[index], numericalGradient(evaluate, input[index]), 3e-3f);
    }
}

TEST(LayerNormSuite, SerializesAndRestoresTrainableParameters)
{
    const PxCpuVector input = { 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f };
    const PxCpuVector scales = { 1.5f, 0.75f, 2.0f };
    const PxCpuVector biases = { -0.25f, 0.5f, 1.25f };
    Model<> sourceModel;
    CpuLayerNorm source(sourceModel, definition());
    source.copyScales(scales);
    source.copyBiases(biases);
    source.forward(input);

    std::stringstream weights(std::ios::in | std::ios::out | std::ios::binary);
    source.saveWeights(weights);

    Model<> restoredModel;
    CpuLayerNorm restored(restoredModel, definition());
    restored.loadWeights(weights);
    restored.forward(input);

    EXPECT_THAT(restored.output(),
                ElementsAre(FloatNear(source.output()[0], 1e-6f),
                            FloatNear(source.output()[1], 1e-6f),
                            FloatNear(source.output()[2], 1e-6f),
                            FloatNear(source.output()[3], 1e-6f),
                            FloatNear(source.output()[4], 1e-6f),
                            FloatNear(source.output()[5], 1e-6f)));
}

TEST(LayerNormSuite, IsConstructedAndExecutedThroughModelConfiguration)
{
    const auto input = PxCpuVector{ 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f };
    const auto expected = PxCpuVector{ -1.2247357f, -1.2247447f,
                                       0.0f, 0.0f,
                                       1.2247357f, 1.2247447f };
    const auto modelDefinition = YAML::Load(R"(
        model:
          batch: 1
          channels: 3
          height: 1
          width: 2
          layers:
            - type: layernorm
              epsilon: 0.00001
    )");

    Model<> model;
    model.parseModel(modelDefinition);

    ASSERT_EQ(model.layerSize(), 1);
    const auto layer = std::dynamic_pointer_cast<CpuLayerNorm>(model.layerAt(0));
    ASSERT_NE(layer, nullptr);

    model.forward(input);
    EXPECT_THAT(layer->output(), ElementsAre(FloatNear(expected[0], 1e-5f),
                                              FloatNear(expected[1], 1e-5f),
                                              FloatNear(expected[2], 1e-6f),
                                              FloatNear(expected[3], 1e-6f),
                                              FloatNear(expected[4], 1e-5f),
                                              FloatNear(expected[5], 1e-5f)));

    layer->delta().fill(1.0f);
    model.backward(input);
    for (const auto value : layer->delta()) {
        EXPECT_TRUE(std::isfinite(value));
    }
}
