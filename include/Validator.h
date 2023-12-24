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

#ifndef PIXIENN_VALIDATOR_H
#define PIXIENN_VALIDATOR_H

#include "Common.h"
#include "ConfusionMatrix.h"

namespace px {

class Model;    // forward reference

class Validator
{
public:
    Validator(Model& model);

    void validate(TrainBatch&& batch);

    float avgRecall() const noexcept;
    float mAP() const noexcept;
    float microAvgF1() const noexcept;

private:
    void processDetects(const GroundTruthVec& gts);
    GroundTruthVec::size_type findGroundTruth(const Detection& detection, const GroundTruthVec& gts);
    float iou(const Detection& detection, const GroundTruth& truth);

    ConfusionMatrix matrix_;
    std::unordered_set<int> classesSeen_;

    Model& model_;
};

}

#endif  // PIXIENN_VALIDATOR_H
