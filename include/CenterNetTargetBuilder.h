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

#pragma once

#include "GroundTruth.h"
#include "PxTensor.h"

namespace px {

struct CenterNetTargets
{
    PxCpuVector heatmap;       // [C, H, W]
    PxCpuVector size;          // [2, H, W]
    PxCpuVector offset;        // [2, H, W]
    PxCpuVector mask;          // [1, H, W] or list of coords
};

class CenterNetTargetBuilder
{
public:
    CenterNetTargetBuilder(int numClasses, int stride, int imageW, int imageH);
    CenterNetTargets buildTargets(const GroundTruthVec& truth);

private:
    void drawGaussian(PxCpuVector& heatmap, int classId, int cx, int cy, float radius);
    float gaussianRadius(float width, float height) const;

    int numClasses_;
    int stride_;
    int fmapW_, fmapH_;
    int imageW_, imageH_;
};

} // namespace px
