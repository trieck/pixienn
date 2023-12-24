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

#ifndef PIXIENN_GROUNDTRUTH_H
#define PIXIENN_GROUNDTRUTH_H

#include <opencv2/core/types.hpp>
#include "Common.h"

namespace px {

struct GroundTruth
{
    int classId;
    cv::Rect2f box;
};

using GroundTruthVec = std::vector<GroundTruth>;
using GroundTruths = std::vector<GroundTruthVec>;

}   // px

#endif  // PIXIENN_GROUNDTRUTH_H
