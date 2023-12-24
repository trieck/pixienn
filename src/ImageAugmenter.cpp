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

    Mat canvas{ targetSize.height, targetSize.width, CV_32FC(image.channels()), Scalar_<float>::all(0.5f) };

    cv::Rect roiSrc, roiDst;
    calculateROI(nw, nh, dx, dy, roiSrc, roiDst, canvas);
    implace(image, nw, nh, roiSrc, roiDst, canvas);

    cv::Size srcSize = image.size();

    BoxTransform transform = [nw, nh, roiSrc, roiDst, srcSize, targetSize, flip](const cv::Rect2f& box) -> cv::Rect2f {

        auto lbox = lightBox(box, srcSize);

        auto xr = nw / srcSize.width;
        auto yr = nh / srcSize.height;

        float x;
        auto y = lbox.y * yr + roiDst.y - roiSrc.y;
        auto w = lbox.width * xr;
        auto h = lbox.height * yr;

        if (flip) {
            x = (srcSize.width - lbox.x - lbox.width) * xr + roiDst.x - roiSrc.x;
        } else {
            x = lbox.x * xr + roiDst.x - roiSrc.x;
        }

        auto dbox = darkBox(Rect2f{ x, y, w, h }, targetSize);

        return dbox;
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
