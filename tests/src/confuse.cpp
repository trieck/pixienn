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

#include "ConfusionMatrix.h"

using namespace px;
using namespace testing;

class ConfusionMatrixTest : public ::testing::Test
{
protected:
    ConfusionMatrixTest() : matrix(CLASSES)
    {

    }
    void SetUp() override
    {
        matrix.reset();
    }

    void TearDown() override
    {
    }

    static constexpr auto CLASSES = 4;

    ConfusionMatrix matrix;
};

TEST_F(ConfusionMatrixTest, Update)
{
    const auto trueClass = 1;
    const auto predictedClass = 2;

    matrix.update(trueClass, predictedClass);

    ASSERT_EQ(matrix.TP(trueClass), 0);
    ASSERT_EQ(matrix.FP(trueClass), 1);
    ASSERT_EQ(matrix.FN(predictedClass), 1);
}

TEST_F(ConfusionMatrixTest, Reset)
{
    const auto trueClass = 1;
    const auto predictedClass = 2;

    matrix.update(trueClass, predictedClass);

    ASSERT_EQ(matrix.TP(trueClass), 0);
    ASSERT_EQ(matrix.FP(trueClass), 1);
    ASSERT_EQ(matrix.FN(predictedClass), 1);

    matrix.reset();

    ASSERT_EQ(matrix.TP(trueClass), 0);
    ASSERT_EQ(matrix.FP(predictedClass), 0);
    ASSERT_EQ(matrix.FN(trueClass), 0);
}

TEST_F(ConfusionMatrixTest, TruePositives)
{
    const auto trueClass = 1;
    const auto predictedClass = 1;

    matrix.update(trueClass, predictedClass);
    matrix.update(trueClass, predictedClass);

    const auto result = matrix.TP(trueClass);

    ASSERT_EQ(result, 2);
}

TEST_F(ConfusionMatrixTest, FalsePositives)
{
    const auto trueClass = 1;
    const auto predictedClass = 2;

    matrix.update(trueClass, predictedClass);
    matrix.update(trueClass, predictedClass);
    matrix.update(trueClass, predictedClass);

    const auto result = matrix.FP(trueClass);

    ASSERT_EQ(result, 3);
}

TEST_F(ConfusionMatrixTest, GhostPrediction)
{
    const auto trueClass = -1;
    const auto predictedClass = 1;

    matrix.update(trueClass, predictedClass);
    matrix.update(trueClass, predictedClass);
    matrix.update(trueClass, predictedClass);

    auto result = matrix.ghosts(predictedClass);
    ASSERT_EQ(result, 3);
}

TEST_F(ConfusionMatrixTest, UndetectedObject)
{
    const auto trueClass = 1;
    const auto predictedClass = -1;

    matrix.update(trueClass, predictedClass);
    matrix.update(trueClass, predictedClass);
    matrix.update(trueClass, predictedClass);

    auto result = matrix.undetected(trueClass);
    ASSERT_EQ(result, 3);
}

TEST_F(ConfusionMatrixTest, FalseNegatives)
{
    const auto trueClass = 1;
    const auto predictedClass = 2;

    matrix.update(trueClass, predictedClass);
    matrix.update(trueClass, predictedClass);
    matrix.update(trueClass, predictedClass);

    const auto result = matrix.FN(predictedClass);

    ASSERT_EQ(result, 3);
}


TEST_F(ConfusionMatrixTest, Precision)
{
    const int trueClass = 1;
    const int predictedClass = 1;
    matrix.update(trueClass, predictedClass);

    float precision = matrix.precision(trueClass);

    ASSERT_FLOAT_EQ(precision, 1.0f);
}

TEST_F(ConfusionMatrixTest, Recall)
{
    const int trueClass = 2;
    const int predictedClass = 2;
    matrix.update(trueClass, predictedClass);

    float recall = matrix.recall(trueClass);

    ASSERT_FLOAT_EQ(recall, 1.0f);
}

TEST_F(ConfusionMatrixTest, mAP)
{
    const auto class1 = 1;
    const auto class2 = 2;

    matrix.update(class1, class1);
    matrix.update(class1, class1);
    matrix.update(class1, class2);

    matrix.update(class2, class2);
    matrix.update(class2, class1);

    EXPECT_NEAR(matrix.precision(1), 0.66f, 1e-2);
    EXPECT_NEAR(matrix.recall(1), 0.66f, 1e-2);

    EXPECT_NEAR(matrix.precision(2), 0.5f, 1e-2);
    EXPECT_NEAR(matrix.recall(2), 0.5f, 1e-2);

    const auto result = matrix.mAP();

    const auto expectedMAP = 7.0f / 24;

    EXPECT_NEAR(result, expectedMAP, 1e-2);
}

TEST_F(ConfusionMatrixTest, F1Score)
{
    const auto classIndex = 1;

    matrix.update(classIndex, 0);
    matrix.update(classIndex, 1);
    matrix.update(classIndex, 2);

    const auto expected = 0.5f;
    const auto f1Score = matrix.F1(classIndex);

    EXPECT_NEAR(f1Score, expected, 1e-2);
}

TEST_F(ConfusionMatrixTest, MicroAvgF1Score)
{
    matrix.update(0, 0);
    matrix.update(0, 1);
    matrix.update(1, 1);
    matrix.update(1, 2);
    matrix.update(2, 2);
    matrix.update(2, 3);
    matrix.update(3, 3);

    const auto microAvgF1Score = matrix.microAvgF1();
    const auto expectedMicroAvgF1 = 0.5714f;

    EXPECT_NEAR(microAvgF1Score, expectedMicroAvgF1, 1e-2);
}
