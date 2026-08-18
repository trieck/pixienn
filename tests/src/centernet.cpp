#include <gtest/gtest.h>

#include "CenterNetTargetBuilder.h"
#include "Model.h"

using namespace px;

TEST(CenterNetTargetBuilderTest, BuildsHeatmapSizeAndOffsetTargets)
{
    CenterNetTargetBuilder builder(2, 16, 8);
    const auto targets = builder.build({ { 1, DarkBox(0.40f, 0.60f, 0.25f, 0.50f) } });

    const auto x = 6;
    const auto y = 4;
    const auto area = 16 * 8;
    const auto index = y * 16 + x;

    EXPECT_FLOAT_EQ(targets.heatmap[area + index], 1.0f);
    EXPECT_FLOAT_EQ(targets.size[index], 4.0f);
    EXPECT_FLOAT_EQ(targets.size[area + index], 4.0f);
    EXPECT_NEAR(targets.offset[index], 0.40f, 1e-6f);
    EXPECT_NEAR(targets.offset[area + index], 0.80f, 1e-6f);
    EXPECT_FLOAT_EQ(targets.mask[index], 1.0f);
    EXPECT_EQ(targets.objects, 1);
    EXPECT_EQ(targets.collisions, 0);
}

TEST(CenterNetTargetBuilderTest, GaussianSupervisesPixelsAroundLargeObjects)
{
    CenterNetTargetBuilder builder(1, 32, 32);
    const auto targets = builder.build({ { 0, DarkBox(0.5f, 0.5f, 0.5f, 0.5f) } });
    const auto center = 16 * 32 + 16;

    EXPECT_EQ(CenterNetTargetBuilder::gaussianRadius(16, 16), 4);
    EXPECT_FLOAT_EQ(targets.heatmap[center], 1.0f);
    EXPECT_GT(targets.heatmap[center + 1], 0.0f);
    EXPECT_LT(targets.heatmap[center + 1], 1.0f);
}

TEST(CenterNetTargetBuilderTest, ReportsRegressionCollisionsAndKeepsLargerBox)
{
    CenterNetTargetBuilder builder(2, 8, 8);
    const auto targets = builder.build({
            { 0, DarkBox(0.51f, 0.51f, 0.10f, 0.10f) },
            { 1, DarkBox(0.52f, 0.52f, 0.30f, 0.20f) }
    });
    const auto index = 4 * 8 + 4;

    EXPECT_EQ(targets.objects, 2);
    EXPECT_EQ(targets.collisions, 1);
    EXPECT_FLOAT_EQ(targets.heatmap[index], 1.0f);
    EXPECT_FLOAT_EQ(targets.heatmap[8 * 8 + index], 1.0f);
    EXPECT_FLOAT_EQ(targets.size[index], 2.40f);
    EXPECT_FLOAT_EQ(targets.size[8 * 8 + index], 1.60f);
}

TEST(CenterNetTargetBuilderTest, RejectsOutOfRangeClasses)
{
    CenterNetTargetBuilder builder(2, 8, 8);
    EXPECT_ANY_THROW(builder.build({ { 2, DarkBox(0.5f, 0.5f, 0.2f, 0.2f) } }));
}

TEST(CenterNetLayerTest, DecodesLocalHeatmapPeakIntoDetection)
{
    CpuModel model;
    model.setLabels({ "object" });
    model.setThreshold(0.5f);
    model.parseModel(YAML::Load(R"(
model:
  batch: 1
  subdivisions: 1
  channels: 5
  height: 4
  width: 4
  layers:
    - type: centernet
      max_detections: 10
      heatmap_bias: 0.0
)"));

    constexpr auto area = 16;
    PxCpuVector input(5 * area, 0.0f);
    std::fill(input.begin(), input.begin() + area, -10.0f);
    const auto index = 2 * 4 + 1;
    input[index] = 4.0f;                    // center confidence
    input[area + index] = 2.0f;             // width in feature-map cells
    input[2 * area + index] = 1.0f;         // height in feature-map cells
    input[3 * area + index] = 0.5f;         // x offset
    input[4 * area + index] = 0.0f;         // y offset

    model.forward(input);
    const auto detections = model.detections();

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].classIndex(), 0);
    EXPECT_NEAR(detections[0].prob(), 0.982014f, 1e-5f);
    EXPECT_NEAR(detections[0].box().x, 0.125f, 1e-6f);
    EXPECT_NEAR(detections[0].box().y, 0.375f, 1e-6f);
    EXPECT_NEAR(detections[0].box().width, 0.5f, 1e-6f);
    EXPECT_NEAR(detections[0].box().height, 0.25f, 1e-6f);
}
