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

#include "TrainBatch.h"

using namespace px;
using namespace testing;

GTEST_TEST(ImageBatchTest, ConstructorAndGetters)
{
    const std::uint32_t batchSize = 2;
    const std::uint32_t channels = 3;
    const std::uint32_t height = 32;
    const std::uint32_t width = 32;

    TrainBatch imageBatch(batchSize, channels, height, width);

    EXPECT_EQ(imageBatch.batchSize(), batchSize);
    EXPECT_EQ(imageBatch.channels(), channels);
    EXPECT_EQ(imageBatch.height(), height);
    EXPECT_EQ(imageBatch.width(), width);
}

GTEST_TEST(ImageBatchTest, CopyConstructor)
{
    const std::uint32_t batchSize = 2;
    const std::uint32_t channels = 3;
    const std::uint32_t height = 32;
    const std::uint32_t width = 32;

    TrainBatch original(batchSize, channels, height, width);
    TrainBatch copy(original);

    // Test that the two objects are independent (deep copy)
    EXPECT_NE(original.imageData().data(), copy.imageData().data());
    EXPECT_NE(original.groundTruth().data(), copy.groundTruth().data());
}

GTEST_TEST(ImageBatchTest, MoveConstructor)
{
    const std::uint32_t batchSize = 2;
    const std::uint32_t channels = 3;
    const std::uint32_t height = 32;
    const std::uint32_t width = 32;

    TrainBatch original(batchSize, channels, height, width);
    TrainBatch moved(std::move(original));

    // Test that the moved object took ownership of resources
    EXPECT_EQ(original.imageData().size(), 0);
    EXPECT_EQ(original.groundTruth().size(), 0);

    // Test that the moved object has the correct dimensions
    EXPECT_EQ(moved.batchSize(), batchSize);
    EXPECT_EQ(moved.channels(), channels);
    EXPECT_EQ(moved.height(), height);
    EXPECT_EQ(moved.width(), width);
}

GTEST_TEST(ImageBatchTest, SetImageDataAndGetters)
{
    const std::uint32_t batchSize = 2;
    const std::uint32_t channels = 3;
    const std::uint32_t height = 32;
    const std::uint32_t width = 32;

    TrainBatch imageBatch(batchSize, channels, height, width);

    // Set image data for index 0
    PxCpuVector imageData0(height * channels * width, 1.0f);
    imageBatch.setImageData(0, imageData0);

    // Set image data for index 1
    PxCpuVector imageData1(height * channels * width, 2.0f);
    imageBatch.setImageData(1, imageData1);

    // Expect an exception for out-of-bounds condition
    EXPECT_THROW(imageBatch.setImageData(2, imageData1), px::Error);

    // Test slice for index 0
    PxCpuVector::const_pointer slice0 = imageBatch.slice(0);
    for (std::size_t i = 0; i < imageData0.size(); ++i) {
        EXPECT_EQ(slice0[i], imageData0[i]);
    }

    // Test slice for index 1
    PxCpuVector::const_pointer slice1 = imageBatch.slice(1);
    for (std::size_t i = 0; i < imageData1.size(); ++i) {
        EXPECT_EQ(slice1[i], imageData1[i]);
    }
}

TEST(ImageBatchTest, AddGroundTruth)
{
    TrainBatch imageBatch(2, 3, 4, 5);

    GroundTruth groundTruth;
    groundTruth.classId = 1;
    groundTruth.box = cv::Rect2f(1.0f, 2.0f, 3.0f, 4.0f);

    // Add ground truth to index 0
    imageBatch.addGroundTruth(0, std::move(groundTruth));

    // Retrieve the ground truths for index 0 and check if it matches
    const auto& groundTruths = imageBatch.groundTruth(0);
    ASSERT_EQ(groundTruths.size(), 1);
    EXPECT_EQ(groundTruths[0].classId, 1);
    EXPECT_EQ(groundTruths[0].box, cv::Rect2f(1.0f, 2.0f, 3.0f, 4.0f));
}

TEST(ImageBatchTest, SetGroundTruth)
{
    TrainBatch imageBatch(2, 3, 4, 5);

    GroundTruthVec groundTruthVec = {
            { 1, cv::Rect2f(1.0f, 2.0f, 3.0f, 4.0f) },
            { 2, cv::Rect2f(5.0f, 6.0f, 7.0f, 8.0f) }
    };

    // Set ground truth for index 0
    imageBatch.setGroundTruth(0, std::move(groundTruthVec));

    // It should have been moved
    EXPECT_TRUE(groundTruthVec.empty());

    // Retrieve the ground truths for index 0 and check if they match
    const auto& groundTruths = imageBatch.groundTruth(0);
    ASSERT_EQ(groundTruths.size(), 2);

    EXPECT_EQ(groundTruths[0].classId, 1);
    EXPECT_EQ(groundTruths[0].box, cv::Rect2f(1.0f, 2.0f, 3.0f, 4.0f));

    EXPECT_EQ(groundTruths[1].classId, 2);
    EXPECT_EQ(groundTruths[1].box, cv::Rect2f(5.0f, 6.0f, 7.0f, 8.0f));
}

TEST(ImageBatchTest, AllocateMethod)
{
    TrainBatch imageBatch;
    imageBatch.allocate(2, 3, 32, 32);

    EXPECT_EQ(imageBatch.batchSize(), 2u);
    EXPECT_EQ(imageBatch.channels(), 3u);
    EXPECT_EQ(imageBatch.height(), 32u);
    EXPECT_EQ(imageBatch.width(), 32u);
    EXPECT_EQ(imageBatch.imageData().size(), 2 * 3 * 32 * 32u);
    EXPECT_TRUE(imageBatch.groundTruth().empty());
}