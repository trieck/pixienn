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

#ifndef PIXIENN_DETECTION_H
#define PIXIENN_DETECTION_H

#include <opencv2/core/types.hpp>

namespace px {

class Detection
{
public:
    Detection(cv::Rect2f box, int classIndex, float prob);

    float prob() const noexcept;
    const cv::Rect2f& box() const noexcept;
    int classIndex() const noexcept;

private:
    cv::Rect2f box_;
    float prob_;
    int classIndex_ = 0;
};

using Detections = std::vector<Detection>;

struct Detector
{
    virtual void addDetects(Detections& detects, float threshold) = 0;
    virtual void addDetects(Detections& detects, int width, int height, float threshold) = 0;

#ifdef USE_CUDA
    virtual void addDetectsGpu(Detections& detects, int width, int height, float threshold) = 0;
#endif // USE_CUDA
};

}   // px

#endif // PIXIENN_DETECTION_H
