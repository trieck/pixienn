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

#ifdef USE_CUDA
#include "PxTensor.h"
#endif

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

#ifdef USE_CUDA

TEST(TransformerSuite, CUDAForwardMatchesCPU)
{
    const auto definition = YAML::Load(R"(
        model:
          batch: 1
          channels: 4
          height: 2
          width: 2
          layers:
            - type: positional-encoding
            - type: self-attention
              heads: 2
            - type: shortcut
              from: -2
              activation: linear
            - type: layernorm
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
            - type: shortcut
              from: -4
              activation: linear
    )");
    CpuModel cpu;
    CudaModel gpu;
    cpu.parseModel(definition);
    gpu.parseModel(definition);
    PxCpuVector input = { 0.2f, 1.4f, -1.1f, 0.7f,
                          2.3f, -0.4f, 0.5f, 1.1f,
                          -0.8f, 0.6f, 1.7f, -1.3f,
                          0.9f, -0.2f, 0.4f, 1.8f };
    PxCudaVector cudaInput(input.data(), input.data() + input.size());
    cpu.forward(input);
    gpu.forward(cudaInput);
    // The convolutional layers are independently initialized in the two
    // models. Compare the shared deterministic Transformer path only.
    for (auto layerIndex = 0; layerIndex < 4; ++layerIndex) {
        const auto cpuLayerOutput = cpu.layerAt(layerIndex)->output().asVector();
        const auto gpuLayerOutput = gpu.layerAt(layerIndex)->output().asVector();
        ASSERT_EQ(cpuLayerOutput.size(), gpuLayerOutput.size());
        for (std::size_t i = 0; i < cpuLayerOutput.size(); ++i) {
            EXPECT_NEAR(cpuLayerOutput[i], gpuLayerOutput[i], 2e-4f)
                    << "layer " << layerIndex << " output index " << i;
        }
    }
}

TEST(TransformerSuite, CUDAAttentionBackwardMatchesCPU)
{
    const auto definition = YAML::Load(R"(
        type: self-attention
        batch: 1
        channels: 4
        height: 1
        width: 2
        heads: 2
    )");
    Model<> cpuModel;
    Model<Device::CUDA> gpuModel;
    CpuSelfAttention cpu(cpuModel, definition);
    SelfAttention<Device::CUDA> gpu(gpuModel, definition);
    const PxCpuVector input = { 0.2f, 1.4f, -1.1f, 0.7f,
                                2.3f, -0.4f, 0.5f, 1.1f };
    const PxCpuVector upstream = { 0.3f, -0.2f, 0.4f, 0.9f,
                                   -0.6f, 0.1f, 0.8f, -0.5f };
    PxCudaVector cudaInput(input.data(), input.data() + input.size());
    PxCudaVector cudaUpstream(upstream.data(), upstream.data() + upstream.size());
    cpu.forward(input);
    gpu.forward(cudaInput);
    cpu.delta().copy(upstream);
    gpu.delta().copy(cudaUpstream);
    cpu.backward(input, nullptr);
    gpu.backward(cudaInput, nullptr);
    const auto expected = cpu.delta().asVector();
    const auto actual = gpu.delta().asVector();
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(expected[i], actual[i], 2e-3f) << "delta index " << i;
    }
}

TEST(TransformerSuite, CUDACenterNetMatchesCPU)
{
    const auto definition = YAML::Load(R"(
        model:
          batch: 1
          channels: 84
          height: 80
          width: 80
          max_batches: 1
          layers:
            - type: centernet
              heatmap_bias: -2.19
    )");
    CpuModel cpu;
    CudaModel gpu;
    std::vector<std::string> labels(80, "class");
    cpu.setLabels(labels);
    gpu.setLabels(labels);
    cpu.setMode(Mode::VALIDATING);
    gpu.setMode(Mode::VALIDATING);
    cpu.parseModel(definition);
    gpu.parseModel(definition);
    MiniBatch cpuBatch(1, 84, 80, 80);
    MiniBatch gpuBatch(1, 84, 80, 80);
    GroundTruthVec truth{ { 16, DarkBox(0.55f, 0.52f, 0.42f, 0.58f) } };
    cpuBatch.setGroundTruth(0, GroundTruthVec(truth));
    gpuBatch.setGroundTruth(0, std::move(truth));
    cpu.setTrainBatch(std::move(cpuBatch));
    gpu.setTrainBatch(std::move(gpuBatch));
    PxCpuVector input(84 * 80 * 80);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.01f;
    }
    PxCudaVector cudaInput(input.data(), input.data() + input.size());
    cpu.forward(input);
    gpu.forward(cudaInput);
    const auto cpuOutput = cpu.layerAt(0)->output().asVector();
    const auto gpuOutput = gpu.layerAt(0)->output().asVector();
    const auto cpuDelta = cpu.layerAt(0)->delta().asVector();
    const auto gpuDelta = gpu.layerAt(0)->delta().asVector();
    ASSERT_EQ(cpuOutput.size(), gpuOutput.size());
    ASSERT_EQ(cpuDelta.size(), gpuDelta.size());
    for (std::size_t i = 0; i < cpuOutput.size(); ++i) {
        EXPECT_NEAR(cpuOutput[i], gpuOutput[i], 2e-5f) << "output index " << i;
        EXPECT_NEAR(cpuDelta[i], gpuDelta[i], 2e-4f) << "delta index " << i;
    }
    EXPECT_NEAR(cpu.cost(), gpu.cost(), 1.0f);
}

#endif
