/********************************************************************************
* Copyright 2020 Thomas A. Rieck, All Rights Reserved
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

#include "opencv2/core/types.hpp"

namespace px {

class Detection
{
public:
    Detection(int classes, cv::Rect box, float objectness);

    float& operator[](int clazz);
    const float& operator[](int clazz) const;

    const std::vector<float>& prob() const noexcept;

    int size() const noexcept;
    const cv::Rect& box() const noexcept;

    float max() const noexcept;
    int maxClass() const noexcept;
    void setMaxClass(int max);

private:
    cv::Rect box_;
    std::vector<float> prob_;
    float objectness_;
    int maxClass_ = 0;
};

using Detections = std::vector<Detection>;

struct Detector
{
    virtual void addDetects(Detections& detects, int width, int height, float threshold) = 0;
#ifdef USE_CUDA
    virtual void addDetectsGpu(Detections& detects, int width, int height, float threshold) = 0;
#endif // USE_CUDA
};

}   // px

#endif // PIXIENN_DETECTION_H
