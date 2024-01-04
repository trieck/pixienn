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

#include "Box.h"
#include "Image.h"
#include "Model.h"
#include "TrainBatch.h"
#include "Validator.h"
#include "Timer.h"
#include "Utility.h"

namespace px {

Validator::Validator(Model& model) : model_(model)
{
}

void Validator::validate(TrainBatch&& batch)
{
    classesSeen_.clear();
    matrix_.resize(model_.classes());
    model_.setTraining(false);
    model_.setThreshold(0.2);

    auto inputSize = batch.height() * batch.width() * batch.channels();
    PxCpuVector input(inputSize, 0.0f);

    auto batchSize = batch.batchSize();
    for (auto i = 0; i < batchSize; ++i) {
        auto image = batch.slice(i);

        input.copyHost(image, inputSize);

        model_.forward(input);

        processDetects(batch.groundTruth(i));

        std::cout << '.' << std::flush;
    }

    model_.setTraining(true);
}

float Validator::mAP() const noexcept
{
    return matrix_.mAP(classesSeen_.size());
}

float Validator::microAvgF1() const noexcept
{
    return matrix_.microAvgF1();
}

float Validator::avgRecall() const noexcept
{
    return matrix_.avgRecall(classesSeen_.size());
}

void Validator::processDetects(const GroundTruthVec& gts)
{
    auto detects = model_.detections();
    detects = nms(detects, 0.45f);

    GroundTruthVec gtsCopy(gts);

    for (const auto& detect: detects) {
        classesSeen_.emplace(detect.classIndex());
        auto index = findGroundTruth(detect, gtsCopy);
        if (index < gtsCopy.size()) {
            classesSeen_.emplace(gtsCopy[index].classId);
            matrix_.update(gtsCopy[index].classId, detect.classIndex());   // true or false positive
            gtsCopy.erase(gtsCopy.begin() + index);
        } else {
            matrix_.update(-1, detect.classIndex());    // this is a "ghost prediction", a false positive
        }
    }

    for (const auto& gt: gtsCopy) {
        classesSeen_.emplace(gt.classId);
        matrix_.update(gt.classId, -1);    // this is an "undetected object", a false negative
    }
}

GroundTruthVec::size_type Validator::findGroundTruth(const Detection& detection, const GroundTruthVec& gts)
{
    auto bestIt = std::end(gts);
    auto bestIoU = -std::numeric_limits<float>::max();

    for (auto it = std::cbegin(gts); it != std::cend(gts); it++) {
        auto IoU = iou(detection, *it);
        if (IoU > bestIoU) {
            bestIoU = IoU;
            bestIt = it;
        }
    }

    return std::distance(gts.cbegin(), bestIt);
}

float Validator::iou(const Detection& detection, const GroundTruth& truth)
{
    cv::Rect2f pred(detection.box());
    cv::Rect2f gtBox(truth.box);

    auto iou = boxIoU(pred, gtBox);

    return iou;
}

} // px

