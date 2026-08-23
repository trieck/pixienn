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

#include <gtest/gtest.h>

#include "Model.h"

using namespace px;

TEST(TransformerSuite, ComposesAnEncoderBlockFromExistingLayers)
{
    const auto modelDefinition = YAML::Load(R"(
        model:
          batch: 1
          channels: 4
          height: 2
          width: 2
          layers:
            # Add spatial identity before attention.
            - type: positional-encoding
            # Global token mixing with learned projections.
            - type: self-attention
              heads: 2
            # Attention residual: positional input + attention output.
            - type: shortcut
              from: -2
              activation: linear
            # Normalize before the feed-forward sublayer.
            - type: layernorm
            # Expand and contract the embedding at each spatial location.
            - type: conv
              filters: 16
              kernel: 1
              stride: 1
              pad: True
              activation: mish
            - type: conv
              filters: 4
              kernel: 1
              stride: 1
              pad: True
              activation: linear
            # Feed-forward residual: attention result + feed-forward output.
            - type: shortcut
              from: -4
              activation: linear
    )");
    const PxCpuVector input = { 0.2f, 1.4f, -1.1f, 0.7f,
                                2.3f, -0.4f, 0.5f, 1.1f,
                                -0.8f, 0.6f, 1.7f, -1.3f,
                                0.9f, -0.2f, 0.4f, 1.8f };
    Model<> model;
    model.parseModel(modelDefinition);

    ASSERT_EQ(model.layerSize(), 7);
    model.forward(input);

    const auto& output = model.layerAt(model.layerSize() - 1)->output();
    ASSERT_EQ(output.size(), input.size());
    for (const auto value : output) {
        EXPECT_TRUE(std::isfinite(value));
    }

    model.layerAt(model.layerSize() - 1)->delta().fill(1.0f);
    model.backward(input);
    for (const auto& layer : model.layers()) {
        for (const auto value : layer->delta()) {
            EXPECT_TRUE(std::isfinite(value));
        }
    }
}
