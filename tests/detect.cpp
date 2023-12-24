/********************************************************************************
* Copyright 2023 Thomas A. Rieck, All Rights Reserved
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

#include "Box.h"
#include "Common.h"
#include "DetectAlgo.h"
#include "Utility.h"

using namespace px;
using namespace testing;

float darkIoU(int gridSize, cv::Rect2f box1, cv::Rect2f box2)
{
    cv::Rect2f dbox1(std::move(box1));
    cv::Rect2f dbox2(std::move(box2));

    dbox1.x /= gridSize;
    dbox1.y /= gridSize;

    dbox2.x /= gridSize;
    dbox2.y /= gridSize;

    auto iou = boxIou(dbox1, dbox2);

    return iou;
}



cv::Rect2f darkBox(int gridSize, int row, int col)
{
    auto minCol = float(col) / gridSize;
    auto maxCol = float(col + 0.999) / gridSize;
    auto minRow = float(row) / gridSize;
    auto maxRow = float(row + 0.999) / gridSize;

    auto x = randomUniform(minCol, maxCol);
    auto y = randomUniform(minRow, maxRow);
    auto width = randomUniform<float>();
    auto height = randomUniform<float>();

    auto ncol = int(x * gridSize);
    EXPECT_TRUE(ncol == col);

    auto nrow = int(y * gridSize);
    EXPECT_TRUE(ncol == col);

    return { x, y, width, height };
}

std::pair<cv::Rect2f, cv::Rect2f> makeBoxes(int gridSize, int gridRow, int gridCol, float minIoU)
{
    float iou = 0;

    cv::Rect2f box1, box2;
    box1 = darkBox(gridSize, gridRow, gridCol);

    constexpr auto maxIterations = 100000;
    auto iterations = 0;

    do {
        box2 = darkBox(gridSize, gridRow, gridCol);
        iou = boxIou(box1, box2);

        if (++iterations == maxIterations) {
            PX_ERROR_THROW("Unable to generate bounding boxes in %d iterations.", iterations);
        }

    } while (iou < minIoU);

    return { box1, box2 };
}

class DetectionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        input = PxCpuVector(TENSOR_SIZE, 0.0f);
        output = PxCpuVector(TENSOR_SIZE, 0.0f);
        delta = PxCpuVector(TENSOR_SIZE, 0.0f);

        groundTruths.resize(BATCH_SIZE);

        ctxt.batch = 1;
        ctxt.input = &input;
        ctxt.output = &output;
        ctxt.delta = &delta;
        ctxt.cost = &cost;
        ctxt.classes = C;
        ctxt.coords = COORDS;
        ctxt.groundTruths = &groundTruths;
        ctxt.inputs = TENSOR_SIZE;
        ctxt.outputs = TENSOR_SIZE;
        ctxt.num = B;
        ctxt.side = S;
        ctxt.training = true;
    }

    void AddGroundTruth(int batch, GroundTruth&& gt)
    {
        groundTruths[batch].emplace_back(std::move(gt));
    }

    void CheckDetect(int clazz, float prob, int row, int col, const cv::Rect2f& pred)
    {
        Detections dets;

        PredictContext pctxt{};
        pctxt.detections = &dets;
        pctxt.predictions = output.data();
        pctxt.threshold = 0.8f;
        pctxt.classes = ctxt.classes;
        pctxt.width = IMAGE_SIZE.width;
        pctxt.height = IMAGE_SIZE.height;
        pctxt.coords = ctxt.coords;
        pctxt.num = ctxt.num;
        pctxt.side = ctxt.side;

        detectAddPredicts(pctxt);

        EXPECT_EQ(dets.size(), 1);
        const auto& detect = dets[0];

        EXPECT_FLOAT_EQ(detect.classIndex(), clazz);
        EXPECT_FLOAT_EQ(detect.prob(), prob);

        auto bbox = cv::Rect(detect.box());

        auto x = (pred.x + col) / ctxt.side * IMAGE_SIZE.width;
        auto y = (pred.y + row) / ctxt.side * IMAGE_SIZE.height;
        auto w = std::pow(pred.width, (ctxt.sqrt ? 2 : 1)) * IMAGE_SIZE.width;
        auto h = std::pow(pred.height, (ctxt.sqrt ? 2 : 1)) * IMAGE_SIZE.height;

        auto left = std::max<int>(0, (x - w / 2));
        auto right = std::min<int>(IMAGE_SIZE.width - 1, (x + w / 2));
        auto top = std::max<int>(0, (y - h / 2));
        auto bottom = std::min<int>(IMAGE_SIZE.height - 1, (y + h / 2));
        cv::Rect predBox{ left, top, right - left, bottom - top };

        EXPECT_EQ(bbox, predBox);
    }

    void TearDown() override
    {
    }

    static constexpr auto BATCH_SIZE = 1;
    static constexpr auto B = 2;                                            // number of bounding boxes predicted
    static constexpr auto C = 20;                                           // number of classes
    static constexpr auto COORDS = 4;                                       // number of coordinates
    static constexpr auto S = 7;                                            // size of the side of a grid cell
    static constexpr auto LOCATIONS = S * S;                                // number of grid cells
    static constexpr auto TENSOR_SIZE = LOCATIONS * (B * (COORDS + 1) + C); // the size of the output tensor
    const cv::Size IMAGE_SIZE{ S * 64, S * 64 };

    PxCpuVector input, output, delta;
    DetectContext ctxt{};

    GroundTruths groundTruths;
    float cost = 0;
};

TEST_F(DetectionTest, BoxConversion)
{
    // create two boxes in row#2 and col#3 on an SxS grid
    // with an IoU of at least minIoU.
    // This test may raise an exception, if you set this too high.
    constexpr auto minIoU = 0.3f;

    auto boxes = makeBoxes(S, 2, 3, minIoU);
    auto box1 = boxes.first;
    auto box2 = boxes.second;

    auto col1 = int(box1.x * S);
    auto row1 = int(box1.y * S);
    EXPECT_EQ(col1, 3);
    EXPECT_EQ(row1, 2);

    // ensure the IoU is within bounds
    auto iou = boxIou(box1, box2);
    EXPECT_TRUE(minIoU <= iou);
}

TEST_F(DetectionTest, Forward)
{
    auto row = 2, col = 3;
    int clazz = 1;
    int predIndex = row * S + col;

    auto boxes = makeBoxes(S, row, col, 0.2f);
    auto predBox = boxes.first;
    auto gtBox = boxes.second;
    auto iou = darkIoU(S, predBox, gtBox);
    auto bboxIndex = 0; // in the range 0.B-1

    AddGroundTruth(0, { clazz, gtBox });

    auto objOffset = LOCATIONS * C + predIndex * B + bboxIndex;
    auto classOffset = predIndex * C + clazz;
    auto bboxOffset = LOCATIONS * (ctxt.classes + ctxt.num) + (predIndex * ctxt.num + bboxIndex) * ctxt.coords;
    auto objectness = 0.8;
    auto probability = 1.2;

    input[objOffset] = objectness;          // objectness
    input[bboxOffset + 0] = predBox.x;      // x
    input[bboxOffset + 1] = predBox.y;      // y
    input[bboxOffset + 2] = predBox.width;  // width
    input[bboxOffset + 3] = predBox.height; // height
    input[classOffset] = 1.2f;              // class probability

    detectForward(ctxt);

    EXPECT_EQ(ctxt.count, 1); // make sure we found one prediction
    EXPECT_FLOAT_EQ(ctxt.avgAllCat / (ctxt.count * ctxt.classes), probability / ctxt.classes);
    EXPECT_FLOAT_EQ(ctxt.avgAnyObj / ctxt.count, objectness);
    EXPECT_FLOAT_EQ(ctxt.avgCat / ctxt.count, probability);
    EXPECT_FLOAT_EQ(ctxt.avgObj / ctxt.count, objectness);
    EXPECT_NEAR(ctxt.avgIoU / ctxt.count, iou, 1e-2);

    EXPECT_FLOAT_EQ(delta[objOffset], ctxt.objectScale * (1 - objectness));
    EXPECT_FLOAT_EQ(delta[bboxOffset + 0], ctxt.coordScale * (gtBox.x - predBox.x));
    EXPECT_FLOAT_EQ(delta[bboxOffset + 1], ctxt.coordScale * (gtBox.y - predBox.y));
    EXPECT_FLOAT_EQ(delta[bboxOffset + 2], ctxt.coordScale * (gtBox.width - predBox.width));
    EXPECT_FLOAT_EQ(delta[bboxOffset + 3], ctxt.coordScale * (gtBox.height - predBox.height));
    EXPECT_FLOAT_EQ(delta[classOffset], ctxt.classScale * (1 - probability));

    CheckDetect(clazz, objectness * probability, row, col, predBox);
}
