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

#include <boost/filesystem.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <opencv2/core.hpp>

#include "Error.h"
#include "TiffIO.h"

using namespace boost::filesystem;
using namespace px;
using namespace testing;

namespace px {

class TIFFIOTest : public ::testing::Test
{
protected:
    path testImagePath;

    void SetUp() override
    {
        testImagePath = temp_directory_path() / "test_image.tif";
    }

    void TearDown() override
    {
        remove(testImagePath);
    }
};

TEST_F(TIFFIOTest, OpenNonExistingFile)
{
    auto path = temp_directory_path() / "non_existing_file.tif";
    EXPECT_THROW(readTIFF(path.string().c_str()), px::Error);
}

TEST_F(TIFFIOTest, ReadWriteTIFF)
{
    cv::Mat sampleImage(100, 100, CV_32FC3, cv::Scalar(0.5, 0.3, 0.2));
    writeTIFF(testImagePath.string().c_str(), sampleImage);

    auto readImage = readTIFF(testImagePath.string().c_str());

    EXPECT_EQ(sampleImage.rows, readImage.rows);
    EXPECT_EQ(sampleImage.cols, readImage.cols);
    EXPECT_EQ(sampleImage.channels(), readImage.channels());

    std::vector<cv::Mat> sampleChannels, readChannels;
    cv::split(sampleImage, sampleChannels);
    cv::split(readImage, readChannels);

    for (int i = 0; i < sampleChannels.size(); ++i) {
        cv::Mat diff;
        cv::compare(sampleChannels[i], readChannels[i], diff, cv::CMP_NE);
        EXPECT_TRUE(cv::countNonZero(diff) == 0);
    }
}

TEST_F(TIFFIOTest, WriteUnsupportedImageType)
{
    cv::Mat unsupportedImage(100, 100, CV_16UC1, cv::Scalar(1000));
    EXPECT_THROW(writeTIFF(testImagePath.string().c_str(), unsupportedImage), px::Error);
}

}  // namespace px

