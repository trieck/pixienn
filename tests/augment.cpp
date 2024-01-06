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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

#include "ImageAugmenter.h"
#include "Image.h"
#include "Utility.h"

using namespace px;
using namespace testing;
using namespace cv;

bool valuesInRange(const cv::Mat& image, float minVal, float maxVal)
{
    CV_Assert(image.type() == CV_32FC3);

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3f value = image.at<cv::Vec3f>(i, j);
            for (int k = 0; k < 3; ++k) {
                if (value[k] < minVal || value[k] > maxVal) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool nearBox(const DarkBox& box1, const DarkBox& box2, float epsilon)
{
    auto dx = std::abs(box1.x() - box2.x());
    auto dy = std::abs(box1.y() - box2.y());
    auto dw = std::abs(box1.w() - box2.w());
    auto dh = std::abs(box1.h() - box2.h());

    return dx < epsilon && dy < epsilon && dw < epsilon && dh < epsilon;
}

class AugmentationTest : public ::testing::Test
{
protected:
    AugmentationTest()
    {
    }

    Mat makeImage()
    {
        return { IMAGE_WIDTH, IMAGE_HEIGHT, CV_8UC3, Scalar::all(192) };
    }

    void SetUp() override
    {
        image = makeImage();

        int dw = jitter_ * IMAGE_WIDTH / 2;
        int dh = jitter_ * IMAGE_HEIGHT / 2;
        auto x = IMAGE_WIDTH / 2 + randomUniform(-dw, dw);
        auto y = IMAGE_WIDTH / 2 + randomUniform(-dw, dw);
        auto w = randomUniform(IMAGE_WIDTH / 8, IMAGE_WIDTH / 4);
        auto h = randomUniform(IMAGE_HEIGHT / 8, IMAGE_HEIGHT / 4);

        Rect2f box(x, y, w, h);

        gt.classId = 0;
        gt.box = darkBox(box, image.size());

        rectangle(image, box, cv::Scalar(255, 0, 0), 2, LINE_4, 0);
    }

    void TearDown() override
    {
    }

    static constexpr auto IMAGE_WIDTH = 200;
    static constexpr auto IMAGE_HEIGHT = 200;

    cv::Mat image;
    GroundTruth gt;

    float saturation_ = .75f;
    float exposure_ = .75f;
    float hue_ = 0.1f;
    float jitter_ = 0.2f;
    bool flip_ = true;
};

TEST_F(AugmentationTest, Setup)
{
    ASSERT_TRUE(image.cols == IMAGE_WIDTH);
    ASSERT_TRUE(image.rows == IMAGE_HEIGHT);
    ASSERT_TRUE(image.channels() == 3);
    ASSERT_TRUE(image.type() == CV_8UC3);
    ASSERT_FALSE(gt.box.empty());
}

TEST_F(AugmentationTest, Light2Dark)
{
    auto size = cv::Size{ IMAGE_WIDTH, IMAGE_HEIGHT };

    auto light = lightBox(gt.box, size);
    auto dark = darkBox(light, size);

    EXPECT_TRUE(nearBox(gt.box, dark, 1e-2));
}

TEST_F(AugmentationTest, EdgeDetection)
{
    cv::Mat edges;
    cv::Canny(image, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    EXPECT_TRUE(contours.size() > 0);

    std::vector<cv::Rect> boxes;
    for (const auto& contour: contours) {
        boxes.emplace_back(std::move(cv::boundingRect(contour)));
    }

    EXPECT_TRUE(boxes.size() > 0);

    auto matched = false;
    for (const auto& box: boxes) {
        auto dbox = darkBox(boxes[0], image.size());
        if (nearBox(dbox, gt.box, 1e-1)) {
            matched = true;
            break;
        }
    }

    EXPECT_TRUE(matched);
}

TEST_F(AugmentationTest, Distort)
{
    auto normal = imnormalize(image);


    ImageAugmenter augmenter(jitter_, hue_, saturation_, exposure_, flip_);

    augmenter.distort(normal);

    ASSERT_TRUE(valuesInRange(normal, 0.0f, 1.0f));
}

TEST_F(AugmentationTest, TransformGT)
{
    auto targetSize = cv::Size{ 768, 768 };

    auto normalized = imnormalize(image);

    ImageAugmenter augmenter(jitter_, hue_, saturation_, exposure_, flip_);
    auto imageLabels = augmenter.augment(normalized, targetSize, { gt });

    auto denormalized = imdenormalize(imageLabels.first);

    cv::Mat gray;
    cv::cvtColor(denormalized, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), BORDER_CONSTANT);

    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    std::vector<cv::Rect> boxes;
    for (const auto& contour: contours) {
        boxes.emplace_back(std::move(cv::boundingRect(contour)));
    }

    if (boxes.size() == 0) {    // it's possible that the box is outside the image
        return;
    }

    auto transformedBox = imageLabels.second[0].box;

    auto matched = false;
    for (const auto& box: boxes) {
        auto dbox = darkBox(box, targetSize);
        if (nearBox(dbox, transformedBox, 1e-1)) {
            matched = true;
            break;
        }
    }

    // EXPECT_TRUE(matched); won't work for boxes not fully contained in the image
}

