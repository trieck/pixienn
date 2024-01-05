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

ImageAugmenter::ImageAugmenter(float jitter, float hue, float saturation, float exposure)
        : jitter_(jitter), hue_(hue), saturation_(saturation), exposure_(exposure)
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
    auto scale = randomUniform(minScale_, maxScale_);

    float nw, nh;
    if (newAR < 1) {
        nh = scale * targetSize.height;
        nw = nh * newAR;
    } else {
        nw = scale * targetSize.width;
        nh = nw / newAR;
    }

    auto dx = randomUniform(0.0f, targetSize.width - nw);
    auto dy = randomUniform(0.0f, targetSize.height - nh);

    bool flip = randomUniform(0.0f, 1.0f) > 0.5f;
    if (flip) {
        cv::flip(image, image, 1);
    }

    auto midpoint = immidpoint(image);

    Mat canvas{ targetSize.height, targetSize.width, image.type(), midpoint };

    cv::Rect roiSrc, roiDst;
    calculateROI(nw, nh, dx, dy, roiSrc, roiDst, canvas);

    implace(image, nw, nh, roiSrc, roiDst, canvas);

    auto w = targetSize.width;
    auto h = targetSize.height;

    BoxTransform transform = [dx, dy, nw, nh, w, h, flip](const DarkBox& box) -> DarkBox {

        auto ddx = -dx / w;
        auto ddy = -dy / h;
        auto sx = nw / w;
        auto sy = nh / h;

        auto x = (flip ? (1.0f - box.x()) : box.x()) * sx - ddx;
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

}   // px
