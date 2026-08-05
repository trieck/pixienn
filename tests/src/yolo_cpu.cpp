#include <gtest/gtest.h>

#include "Model.h"

using namespace px;

TEST(YoloCpuTest, TruthThresholdOverridesNoObjectPenalty)
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
      anchors: [10, 10]
      num: 1
      ignore_thresh: 0.7
      truth_thresh: 0.5
      object_scale: 2.0
      noobject_scale: 0.25
)");
    CpuModel cpu;
    cpu.setLabels({ "object" });
    cpu.parseModel(definition);
    cpu.setMode(Mode::VALIDATING);

    MiniBatch batch(1, 6, 1, 1);
    batch.setGroundTruth(0, { { 0, DarkBox(0.5f, 0.5f, 0.8f, 0.8f) } });
    cpu.setTrainBatch(std::move(batch));

    cpu.forward(PxCpuVector(6, 0.0f));
    const auto delta = cpu.layerAt(0)->delta().asVector();
    ASSERT_EQ(delta.size(), 6);
    EXPECT_FLOAT_EQ(delta[4], 1.0f);
}
