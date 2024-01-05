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

#ifndef PIXIENN_IMAGE_AUGMENTER_H
#define PIXIENN_IMAGE_AUGMENTER_H

#include <opencv2/core/types.hpp>
#include "Common.h"
#include "DarkBox.h"
#include "GroundTruth.h"

namespace px {

using BoxTransform = std::function<DarkBox(const DarkBox&)>;
using Augmentation = std::pair<cv::Mat, BoxTransform>;
using ImageLabel = std::pair<cv::Mat, GroundTruthVec>;

class ImageAugmenter
{
public:
    ImageAugmenter(float jitter, float hue, float saturation, float exposure);
    virtual ~ImageAugmenter() = default;

    Augmentation augment(cv::Mat& image, const cv::Size& targetSize) const;
    ImageLabel augment(cv::Mat& image, const cv::Size& targetSize, const GroundTruthVec& labels) const;

    void distort(cv::Mat& image) const;

private:
    float jitter_, hue_, saturation_, exposure_;
    float minScale_ = 1.0f, maxScale_ = 1.0f;
};

}   // px

#endif  // PIXIENN_IMAGE_AUGMENTER_H
