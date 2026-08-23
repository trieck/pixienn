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
#include "SelfAttention.h"

using namespace px;
using namespace testing;

namespace {

YAML::Node definition()
{
    return YAML::Load(R"(
        type: self-attention
        batch: 1
        channels: 2
        height: 1
        width: 2
    )");
}

YAML::Node multiHeadDefinition()
{
    return YAML::Load(R"(
        type: self-attention
        batch: 1
        channels: 4
        height: 1
        width: 2
        heads: 2
    )");
}

float loss(const PxCpuVector& input, const PxCpuVector& delta)
{
    Model<> model;
    CpuSelfAttention layer(model, definition());
    layer.forward(input);

    auto result = 0.0f;
    for (std::size_t i = 0; i < delta.size(); ++i) {
        result += layer.output()[i] * delta[i];
    }
    return result;
}

float multiHeadLoss(const PxCpuVector& input, const PxCpuVector& delta)
{
    Model<> model;
    CpuSelfAttention layer(model, multiHeadDefinition());
    layer.forward(input);

    auto result = 0.0f;
    for (std::size_t i = 0; i < delta.size(); ++i) {
        result += layer.output()[i] * delta[i];
    }
    return result;
}

float numericalGradient(const std::function<float(float)>& evaluate, float value)
{
    constexpr auto step = 1e-3f;
    return (evaluate(value + step) - evaluate(value - step)) / (2.0f * step);
}

} // namespace

TEST(SelfAttentionSuite, AttendsAcrossSpatialTokens)
{
    const PxCpuVector input = { 1.0f, 0.0f, 0.0f, 1.0f };
    Model<> model;
    CpuSelfAttention layer(model, definition());

    layer.forward(input);

    const auto diagonal = std::exp(1.0f / std::sqrt(2.0f))
            / (std::exp(1.0f / std::sqrt(2.0f)) + 1.0f);
    const auto offDiagonal = 1.0f - diagonal;
    EXPECT_THAT(layer.output(), ElementsAre(FloatNear(diagonal, 1e-5f),
                                             FloatNear(offDiagonal, 1e-5f),
                                             FloatNear(offDiagonal, 1e-5f),
                                             FloatNear(diagonal, 1e-5f)));
}

TEST(SelfAttentionSuite, BackwardMatchesFiniteDifferences)
{
    const PxCpuVector input = { 0.2f, 1.4f, -1.1f, 0.7f };
    const PxCpuVector upstream = { 0.3f, -0.2f, 0.4f, 0.9f };
    Model<> model;
    CpuSelfAttention layer(model, definition());
    layer.forward(input);
    layer.delta().copy(upstream);
    layer.backward(input, nullptr);

    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto evaluate = [&](float value) {
            auto perturbed = input;
            perturbed[i] = value;
            return loss(perturbed, upstream);
        };
        EXPECT_NEAR(layer.delta()[i], numericalGradient(evaluate, input[i]), 3e-3f);
    }
}

TEST(SelfAttentionSuite, MultiHeadBackwardMatchesFiniteDifferences)
{
    const PxCpuVector input = { 0.2f, 1.4f, -1.1f, 0.7f,
                                2.3f, -0.4f, 0.5f, 1.1f };
    const PxCpuVector upstream = { 0.3f, -0.2f, 0.4f, 0.9f,
                                  -0.6f, 0.1f, 0.8f, -0.5f };
    Model<> model;
    CpuSelfAttention layer(model, multiHeadDefinition());
    layer.forward(input);
    layer.delta().copy(upstream);
    layer.backward(input, nullptr);

    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto evaluate = [&](float value) {
            auto perturbed = input;
            perturbed[i] = value;
            return multiHeadLoss(perturbed, upstream);
        };
        EXPECT_NEAR(layer.delta()[i], numericalGradient(evaluate, input[i]), 3e-3f);
    }
}

TEST(SelfAttentionSuite, IsConstructedThroughModelConfiguration)
{
    const auto modelDefinition = YAML::Load(R"(
        model:
          batch: 1
          channels: 2
          height: 1
          width: 2
          layers:
            - type: self-attention
    )");
    const PxCpuVector input = { 1.0f, 0.0f, 0.0f, 1.0f };
    Model<> model;
    model.parseModel(modelDefinition);

    ASSERT_EQ(model.layerSize(), 1);
    const auto layer = std::dynamic_pointer_cast<CpuSelfAttention>(model.layerAt(0));
    ASSERT_NE(layer, nullptr);

    model.forward(input);
    layer->delta().fill(1.0f);
    model.backward(input);
    for (const auto value : layer->delta()) {
        EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(SelfAttentionSuite, SerializesNonIdentityProjectionWeights)
{
    const PxCpuVector input = { 0.2f, 1.4f, -1.1f, 0.7f };
    PxCpuVector queryWeights = { 1.2f, 0.1f, -0.3f, 0.8f };
    Model<> sourceModel;
    CpuSelfAttention source(sourceModel, definition());
    source.copyQueryWeights(queryWeights);
    source.forward(input);

    std::stringstream weights(std::ios::in | std::ios::out | std::ios::binary);
    source.saveWeights(weights);

    Model<> restoredModel;
    CpuSelfAttention restored(restoredModel, definition());
    restored.loadWeights(weights);
    restored.forward(input);

    EXPECT_THAT(restored.output(),
                ElementsAre(FloatNear(source.output()[0], 1e-6f),
                            FloatNear(source.output()[1], 1e-6f),
                            FloatNear(source.output()[2], 1e-6f),
                            FloatNear(source.output()[3], 1e-6f)));
}

TEST(SelfAttentionSuite, ProjectionWeightsUpdateDuringTraining)
{
    const auto modelDefinition = YAML::Load(R"(
        model:
          batch: 1
          channels: 2
          height: 1
          width: 2
          max_batches: 1
          momentum: 0.0
          learning_rate:
            initial_learning_rate: 0.1
            policy: constant
          layers:
            - type: self-attention
    )");
    const PxCpuVector input = { 0.2f, 1.4f, -1.1f, 0.7f };
    const PxCpuVector upstream = { 0.3f, -0.2f, 0.4f, 0.9f };
    Model<> model;
    model.setMode(Mode::TRAINING);
    model.parseModel(modelDefinition);
    const auto layer = std::dynamic_pointer_cast<CpuSelfAttention>(model.layerAt(0));
    ASSERT_NE(layer, nullptr);

    const auto before = PxCpuVector(layer->queryWeights());
    layer->forward(input);
    layer->delta().copy(upstream);
    layer->backward(input, nullptr);
    layer->update();

    auto changed = false;
    for (std::size_t i = 0; i < before.size(); ++i) {
        changed = changed || std::abs(layer->queryWeights()[i] - before[i]) > 1e-7f;
    }
    EXPECT_TRUE(changed);
}
