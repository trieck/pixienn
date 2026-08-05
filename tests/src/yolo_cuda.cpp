#include <gtest/gtest.h>

#ifdef USE_CUDA

#include "PxTensor.h"
#include "Model.h"
#include "YoloKernels.cuh"
#include "YoloTargetBuilder.h"

using namespace px;

TEST(YoloTargetBuilderTest, ResolvesSameCellCollisionOntoFreeMaskedAnchor)
{
    const YoloTargetBuilder builder({ 10, 10, 20, 20 }, { 0, 1 }, 4, 4, 100, 100);
    const auto targets = builder.build({
            { 0, DarkBox(0.30f, 0.30f, 0.10f, 0.10f) },
            { 1, DarkBox(0.31f, 0.31f, 0.11f, 0.11f) }
    });
    const auto cell = 1 * 4 + 1;
    EXPECT_EQ(targets.assigned, 2);
    EXPECT_EQ(targets.classes[cell], 0);
    EXPECT_EQ(targets.anchors[cell], 0);
    EXPECT_EQ(targets.classes[16 + cell], 1);
    EXPECT_EQ(targets.anchors[16 + cell], 1);
}

TEST(YoloTargetBuilderTest, LeavesTruthForAnotherHeadUnassigned)
{
    const YoloTargetBuilder builder({ 10, 10, 80, 80 }, { 0 }, 4, 4, 100, 100);
    const auto targets = builder.build({ { 0, DarkBox(0.5f, 0.5f, 0.8f, 0.8f) } });
    EXPECT_EQ(targets.assigned, 0);
    EXPECT_TRUE(std::all_of(targets.classes.begin(), targets.classes.end(),
                            [](int value) { return value == -1; }));
}

TEST(YoloCudaTest, MatchesCpuInferenceDecoding)
{
    const auto definition = YAML::Load(R"(
model:
  batch: 1
  subdivisions: 1
  channels: 6
  height: 1
  width: 1
  layers:
    - type: yolo
      mask: [0]
      anchors: [1, 1]
      num: 1
      log_interval: 0
)" );
    CpuModel cpu;
    CudaModel gpu;
    cpu.setLabels({ "object" });
    gpu.setLabels({ "object" });
    cpu.setThreshold(0.1f);
    gpu.setThreshold(0.1f);
    cpu.parseModel(definition);
    gpu.parseModel(definition);
    PxCpuVector input{ 0.2f, -0.4f, 0.1f, -0.2f, 2.0f, 1.5f };
    PxCudaVector cudaInput(input.data(), input.data() + input.size());
    cpu.forward(input);
    gpu.forward(cudaInput);
    const auto cpuDetections = cpu.detections();
    const auto gpuDetections = gpu.detections();
    ASSERT_EQ(gpuDetections.size(), cpuDetections.size());
    ASSERT_EQ(gpuDetections.size(), 1);
    EXPECT_NEAR(gpuDetections[0].prob(), cpuDetections[0].prob(), 1e-6f);
    EXPECT_NEAR(gpuDetections[0].box().x, cpuDetections[0].box().x, 1e-6f);
    EXPECT_NEAR(gpuDetections[0].box().y, cpuDetections[0].box().y, 1e-6f);
    EXPECT_NEAR(gpuDetections[0].box().width, cpuDetections[0].box().width, 1e-6f);
    EXPECT_NEAR(gpuDetections[0].box().height, cpuDetections[0].box().height, 1e-6f);
}

TEST(YoloCudaTest, MatchesCpuTrainingOutputDeltaAndCost)
{
    const auto definition = YAML::Load(R"(
model:
  batch: 1
  subdivisions: 1
  channels: 14
  height: 2
  width: 2
  layers:
    - type: yolo
      mask: [0, 1]
      anchors: [1, 1, 2, 2]
      num: 2
      ignore_thresh: 0.3
      truth_thresh: 1.0
      coord_scale: 1.3
      object_scale: 1.7
      noobject_scale: 0.4
      class_scale: 0.8
)" );
    CpuModel cpu;
    CudaModel gpu;
    cpu.setLabels({ "a", "b" });
    gpu.setLabels({ "a", "b" });
    cpu.parseModel(definition);
    gpu.parseModel(definition);
    cpu.setMode(Mode::VALIDATING);
    gpu.setMode(Mode::VALIDATING);
    MiniBatch cpuBatch(1, 14, 2, 2);
    MiniBatch gpuBatch(1, 14, 2, 2);
    GroundTruthVec truths{
            { 0, DarkBox(0.26f, 0.26f, 0.50f, 0.50f) },
            { 1, DarkBox(0.28f, 0.28f, 0.56f, 0.54f) },
            { 1, DarkBox(0.76f, 0.74f, 0.90f, 0.85f) }
    };
    cpuBatch.setGroundTruth(0, GroundTruthVec(truths));
    gpuBatch.setGroundTruth(0, std::move(truths));
    cpu.setTrainBatch(std::move(cpuBatch));
    gpu.setTrainBatch(std::move(gpuBatch));

    PxCpuVector input(56);
    for (auto i = 0u; i < input.size(); ++i) {
        input[i] = static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.11f;
    }
    PxCudaVector cudaInput(input.data(), input.data() + input.size());
    cpu.forward(input);
    gpu.forward(cudaInput);
    const auto cpuOutput = cpu.layerAt(0)->output().asVector();
    const auto gpuOutput = gpu.layerAt(0)->output().asVector();
    const auto cpuDelta = cpu.layerAt(0)->delta().asVector();
    const auto gpuDelta = gpu.layerAt(0)->delta().asVector();
    ASSERT_EQ(gpuOutput.size(), cpuOutput.size());
    ASSERT_EQ(gpuDelta.size(), cpuDelta.size());
    for (auto i = 0u; i < cpuOutput.size(); ++i) {
        EXPECT_NEAR(gpuOutput[i], cpuOutput[i], 2e-6f) << "output index " << i;
        EXPECT_NEAR(gpuDelta[i], cpuDelta[i], 2e-5f) << "delta index " << i;
    }
    EXPECT_NEAR(gpu.cost(), cpu.cost(), 2e-4f);
}

TEST(YoloCudaTest, MatchesReferenceAssignedTargetLossAndGradients)
{
    constexpr int classes = 2;
    constexpr int attributes = classes + 5;
    PxCpuVector raw{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f };
    PxCudaVector input(raw.data(), raw.data() + raw.size());
    PxCudaVector output(attributes, 0.0f);
    PxCudaVector delta(attributes, 0.0f);
    yoloActivateGpu(input.data(), output.data(), 1, 1, classes, 1);

    PxCpuVector truth{ 0.5f, 0.5f, 0.3125f, 0.3125f, 1.0f };
    PxCpuVector assignedBox{ 0.5f, 0.5f, 0.3125f, 0.3125f };
    PxCpuVector anchors{ 10.0f, 10.0f };
    PxCpuVectorT<int> truthCounts{ 1 };
    PxCpuVectorT<int> assignedClasses{ 1 };
    PxCpuVectorT<int> assignedAnchors{ 0 };
    PxCpuVectorT<int> masks{ 0 };
    PxCudaVector truthGpu(truth.data(), truth.data() + truth.size());
    PxCudaVector assignedBoxGpu(assignedBox.data(), assignedBox.data() + assignedBox.size());
    PxCudaVector anchorsGpu(anchors.data(), anchors.data() + anchors.size());
    PxCudaVectorT<int> truthCountsGpu(truthCounts.data(), truthCounts.data() + truthCounts.size());
    PxCudaVectorT<int> assignedClassesGpu(assignedClasses.data(), assignedClasses.data() + assignedClasses.size());
    PxCudaVectorT<int> assignedAnchorsGpu(assignedAnchors.data(), assignedAnchors.data() + assignedAnchors.size());
    PxCudaVectorT<int> masksGpu(masks.data(), masks.data() + masks.size());
    PxCudaVector stats(YOLO_STATS_SIZE, 0.0f);
    PxCudaVector cost(1, 0.0f);

    yoloLossGpu(output.data(), delta.data(), truthGpu.data(), truthCountsGpu.data(), 1,
                assignedClassesGpu.data(), assignedAnchorsGpu.data(), assignedBoxGpu.data(),
                masksGpu.data(), anchorsGpu.data(), stats.data(), cost.data(), 1, 1, 1,
                classes, 1, 1, 32, 32, 0.5f, 1.0f, 1.0f, 2.0f, 0.25f, 3.0f);

    const auto actualOutput = output.asVector();
    const auto actualDelta = delta.asVector();
    const auto actualStats = stats.asVector();
    const auto actualCost = cost.asVector();
    const auto sigmoidNegOne = 1.0f / (1.0f + std::exp(1.0f));
    const auto sigmoidOne = 1.0f / (1.0f + std::exp(-1.0f));
    EXPECT_FLOAT_EQ(actualOutput[0], 0.5f);
    EXPECT_FLOAT_EQ(actualOutput[1], 0.5f);
    EXPECT_FLOAT_EQ(actualOutput[2], 0.0f);
    EXPECT_FLOAT_EQ(actualOutput[3], 0.0f);
    EXPECT_FLOAT_EQ(actualOutput[4], 0.5f);
    EXPECT_NEAR(actualOutput[5], sigmoidNegOne, 1e-6f);
    EXPECT_NEAR(actualOutput[6], sigmoidOne, 1e-6f);

    EXPECT_FLOAT_EQ(actualDelta[0], 0.0f);
    EXPECT_FLOAT_EQ(actualDelta[1], 0.0f);
    EXPECT_FLOAT_EQ(actualDelta[2], 0.0f);
    EXPECT_FLOAT_EQ(actualDelta[3], 0.0f);
    EXPECT_FLOAT_EQ(actualDelta[4], 1.0f);
    EXPECT_NEAR(actualDelta[5], -3.0f * sigmoidNegOne, 1e-6f);
    EXPECT_NEAR(actualDelta[6], 3.0f * (1.0f - sigmoidOne), 1e-6f);
    const auto expectedCost = actualDelta[4] * actualDelta[4]
                              + actualDelta[5] * actualDelta[5]
                              + actualDelta[6] * actualDelta[6];
    EXPECT_NEAR(actualCost[0], expectedCost, 1e-5f);
    EXPECT_NEAR(actualStats[0], 1.0f, 1e-6f);
    EXPECT_FLOAT_EQ(actualStats[1], 1.0f);
    EXPECT_FLOAT_EQ(actualStats[2], 1.0f);
    EXPECT_NEAR(actualStats[3], sigmoidOne, 1e-6f);
    EXPECT_FLOAT_EQ(actualStats[4], 0.5f);
    EXPECT_FLOAT_EQ(actualStats[5], 0.5f);
    EXPECT_FLOAT_EQ(actualStats[6], 1.0f);
    EXPECT_FLOAT_EQ(actualStats[7], 1.0f);
}

TEST(YoloCudaTest, LeavesSparseEntriesZeroForBackground)
{
    constexpr int classes = 1;
    PxCpuVector raw(6, 0.0f);
    PxCudaVector input(raw.data(), raw.data() + raw.size());
    PxCudaVector output(6, 0.0f);
    PxCudaVector delta(6, 7.0f);
    yoloActivateGpu(input.data(), output.data(), 1, 1, classes, 1);
    delta.fill(0.0f);

    PxCpuVector truth(5, 0.0f), assignedBox(4, 0.0f), anchors{ 10.0f, 10.0f };
    PxCpuVectorT<int> truthCounts{ 0 }, assignedClasses{ -1 }, assignedAnchors{ -1 }, masks{ 0 };
    PxCudaVector truthGpu(truth.data(), truth.data() + truth.size());
    PxCudaVector assignedBoxGpu(assignedBox.data(), assignedBox.data() + assignedBox.size());
    PxCudaVector anchorsGpu(anchors.data(), anchors.data() + anchors.size());
    PxCudaVectorT<int> truthCountsGpu(truthCounts.data(), truthCounts.data() + truthCounts.size());
    PxCudaVectorT<int> assignedClassesGpu(assignedClasses.data(), assignedClasses.data() + assignedClasses.size());
    PxCudaVectorT<int> assignedAnchorsGpu(assignedAnchors.data(), assignedAnchors.data() + assignedAnchors.size());
    PxCudaVectorT<int> masksGpu(masks.data(), masks.data() + masks.size());
    PxCudaVector stats(YOLO_STATS_SIZE, 0.0f), cost(1, 0.0f);
    yoloLossGpu(output.data(), delta.data(), truthGpu.data(), truthCountsGpu.data(), 1,
                assignedClassesGpu.data(), assignedAnchorsGpu.data(), assignedBoxGpu.data(),
                masksGpu.data(), anchorsGpu.data(), stats.data(), cost.data(), 1, 1, 1,
                classes, 1, 1, 32, 32, 0.5f, 1.0f, 1.0f, 1.0f, 0.25f, 1.0f);
    const auto actual = delta.asVector();
    EXPECT_FLOAT_EQ(actual[0], 0.0f);
    EXPECT_FLOAT_EQ(actual[1], 0.0f);
    EXPECT_FLOAT_EQ(actual[2], 0.0f);
    EXPECT_FLOAT_EQ(actual[3], 0.0f);
    EXPECT_FLOAT_EQ(actual[4], -0.125f);
    EXPECT_FLOAT_EQ(actual[5], 0.0f);
}

TEST(YoloCudaTest, TruthThresholdOverridesNoObjectPenalty)
{
    constexpr int classes = 1;
    constexpr int attributes = classes + 5;
    PxCpuVector raw{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    PxCudaVector input(raw.data(), raw.data() + raw.size());
    PxCudaVector output(attributes, 0.0f);
    PxCudaVector delta(attributes, 0.0f);
    yoloActivateGpu(input.data(), output.data(), 1, 1, classes, 1);

    // The decoded prediction is a 10x10 box at the image center.  This truth
    // produces an IoU between truth_thresh (0.5) and ignore_thresh (0.7).
    PxCpuVector truth{ 0.5f, 0.5f, 0.8f, 0.8f, 0.0f };
    PxCpuVector assignedBox(4, 0.0f);
    PxCpuVector anchors{ 10.0f, 10.0f };
    PxCpuVectorT<int> truthCounts{ 1 };
    PxCpuVectorT<int> assignedClasses{ -1 };
    PxCpuVectorT<int> assignedAnchors{ -1 };
    PxCpuVectorT<int> masks{ 0 };
    PxCudaVector truthGpu(truth.data(), truth.data() + truth.size());
    PxCudaVector assignedBoxGpu(assignedBox.data(), assignedBox.data() + assignedBox.size());
    PxCudaVector anchorsGpu(anchors.data(), anchors.data() + anchors.size());
    PxCudaVectorT<int> truthCountsGpu(truthCounts.data(), truthCounts.data() + truthCounts.size());
    PxCudaVectorT<int> assignedClassesGpu(assignedClasses.data(), assignedClasses.data() + assignedClasses.size());
    PxCudaVectorT<int> assignedAnchorsGpu(assignedAnchors.data(), assignedAnchors.data() + assignedAnchors.size());
    PxCudaVectorT<int> masksGpu(masks.data(), masks.data() + masks.size());
    PxCudaVector stats(YOLO_STATS_SIZE, 0.0f), cost(1, 0.0f);

    yoloLossGpu(output.data(), delta.data(), truthGpu.data(), truthCountsGpu.data(), 1,
                assignedClassesGpu.data(), assignedAnchorsGpu.data(), assignedBoxGpu.data(),
                masksGpu.data(), anchorsGpu.data(), stats.data(), cost.data(), 1, 1, 1,
                classes, 1, 1, 10, 10, 0.7f, 0.5f, 1.0f, 2.0f, 0.25f, 1.0f);

    // Positive objectness is 2*(1 - sigmoid(0)), not the no-object value
    // -0.25*sigmoid(0).
    EXPECT_FLOAT_EQ(delta.asVector()[4], 1.0f);
}

#endif
