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

#include "ImageAugmenter.h"
#include "Image.h"
#include "Utility.h"

using namespace cv;

namespace px {

ImageAugmenter::ImageAugmenter(float jitter, float hue, float saturation, float exposure, bool flip,
                               bool mosaic, float mosaicProbability)
        : jitter_(jitter), hue_(hue), saturation_(saturation), exposure_(exposure), flip_(flip),
          mosaic_(mosaic), mosaicProbability_(constrain(0.0f, 1.0f, mosaicProbability))
{
}

void ImageAugmenter::distort(Mat& image) const
{
    auto hue = randomUniform<float>(-hue_, hue_);
    auto saturation = randomScale(saturation_);
    auto exposure = randomScale(exposure_);

    imdistort(image, hue, saturation, exposure);
}

Augmentation ImageAugmenter::augment(Mat& image, const cv::Size& targetSize) const
{
    const auto origSize = image.size();

    auto dw = jitter_ * origSize.width;
    auto dh = jitter_ * origSize.height;

    auto newWidth = origSize.width + randomUniform(-dw, dw);
    auto newHeight = origSize.height + randomUniform(-dh, dh);

    auto newAR = newWidth / newHeight;

    float nw, nh;
    if (newAR < 1) {
        nh = targetSize.height;
        nw = nh * newAR;
    } else {
        nw = targetSize.width;
        nh = nw / newAR;
    }

    auto dx = randomUniform(0.0f, targetSize.width - nw);
    auto dy = randomUniform(0.0f, targetSize.height - nh);

    auto flipped = flip_ && randomUniform(0.0f, 1.0f) > 0.5f;
    if (flipped) {
        cv::flip(image, image, 1);
    }

    auto midpoint = immidpoint(image);
    Mat canvas{ targetSize.height, targetSize.width, image.type(), midpoint };

    cv::Rect roiSrc, roiDst;
    calculateROI(nw, nh, dx, dy, roiSrc, roiDst, canvas);

    implace(image, nw, nh, roiSrc, roiDst, canvas);

    distort(canvas);

    auto w = targetSize.width;
    auto h = targetSize.height;

    BoxTransform transform = [dx, dy, nw, nh, w, h, flipped](const DarkBox& box) -> DarkBox {

        auto ddx = -dx / w;
        auto ddy = -dy / h;
        auto sx = nw / w;
        auto sy = nh / h;

        auto x = (flipped ? (1.0f - box.x()) : box.x()) * sx - ddx;
        auto y = box.y() * sy - ddy;
        auto width = box.w() * sx;
        auto height = box.h() * sy;

        x = constrain(0, 1, x);
        y = constrain(0, 1, y);
        width = constrain(0, 1, width);
        height = constrain(0, 1, height);

        return { x, y, width, height };
    };

    return { canvas, transform };
}

ImageLabel ImageAugmenter::augment(Mat& image, const cv::Size& targetSize, const GroundTruthVec& labels) const
{
    auto augmentation = augment(image, targetSize);

    GroundTruthVec transformed(labels.size());

    std::transform(labels.begin(), labels.end(), transformed.begin(),
                   [&augmentation](const GroundTruth& label) -> GroundTruth {
                       auto box = augmentation.second(label.box);
                       return { label.classId, box };
                   });

    return { augmentation.first, transformed };
}

bool ImageAugmenter::useMosaic() const
{
    return mosaic_ && randomUniform<float>() < mosaicProbability_;
}

ImageLabel ImageAugmenter::augmentMosaic(const std::array<ImageLabel, 4>& images,
                                         const cv::Size& targetSize) const
{
    const auto canvasSize = targetSize;
    const auto tileWidth = targetSize.width / 2;
    const auto tileHeight = targetSize.height / 2;
    cv::Mat canvas{ canvasSize.height, canvasSize.width, images[0].first.type(),
                    immidpoint(images[0].first) };
    GroundTruthVec transformed;

    for (auto tile = 0; tile < 4; ++tile) {
        const auto& source = images[tile].first;
        const auto& labels = images[tile].second;
        const auto scale = std::min(static_cast<float>(tileWidth) / source.cols,
                                    static_cast<float>(tileHeight) / source.rows);
        const auto resizedWidth = std::max(1, static_cast<int>(std::round(source.cols * scale)));
        const auto resizedHeight = std::max(1, static_cast<int>(std::round(source.rows * scale)));
        cv::Mat resized;
        cv::resize(source, resized, { resizedWidth, resizedHeight });

        const auto quadrantX = (tile % 2) * tileWidth;
        const auto quadrantY = (tile / 2) * tileHeight;
        const auto imageLeft = static_cast<float>(quadrantX + (tileWidth - resizedWidth) / 2);
        const auto imageTop = static_cast<float>(quadrantY + (tileHeight - resizedHeight) / 2);
        const auto destination = cv::Rect{ static_cast<int>(imageLeft), static_cast<int>(imageTop),
                                           resized.cols, resized.rows };
        resized.copyTo(canvas(destination));

        for (const auto& label : labels) {
            const auto sourceBox = label.box;
            const auto boxLeft = imageLeft + (sourceBox.x() - sourceBox.w() / 2.0f) * resized.cols;
            const auto boxTop = imageTop + (sourceBox.y() - sourceBox.h() / 2.0f) * resized.rows;
            const auto boxRight = imageLeft + (sourceBox.x() + sourceBox.w() / 2.0f) * resized.cols;
            const auto boxBottom = imageTop + (sourceBox.y() + sourceBox.h() / 2.0f) * resized.rows;
            const auto x = (boxLeft + boxRight) / (2.0f * targetSize.width);
            const auto y = (boxTop + boxBottom) / (2.0f * targetSize.height);
            const auto width = (boxRight - boxLeft) / targetSize.width;
            const auto height = (boxBottom - boxTop) / targetSize.height;
            transformed.push_back({ label.classId, { x, y, width, height } });
        }
    }

    auto result = canvas;
    distort(result);
    return { std::move(result), std::move(transformed) };
}

}   // px
