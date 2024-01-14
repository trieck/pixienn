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

#pragma once

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
    using Ptr = std::shared_ptr<ImageAugmenter>;

    ImageAugmenter(float jitter, float hue, float saturation, float exposure, bool flip);

    virtual ~ImageAugmenter() = default;

    Augmentation augment(cv::Mat& image, const cv::Size& targetSize) const;
    ImageLabel augment(cv::Mat& image, const cv::Size& targetSize, const GroundTruthVec& labels) const;

    void distort(cv::Mat& image) const;

private:
    float jitter_, hue_, saturation_, exposure_;
    bool flip_;
};

}   // px

