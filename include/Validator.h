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

#include "Common.h"
#include "ConfusionMatrix.h"

namespace px {

///////////////////////////////////////////////////////////////////////////////
template<Device D>
class Model;

///////////////////////////////////////////////////////////////////////////////

template<Device D>
class Validator
{
public:
    Validator(float threshold, int numClasses);

    using V = typename DeviceTraits<D>::VectorType;

    void validate(Model<D>& model, TrainBatch&& batch);
    void reset() noexcept;

    float avgRecall() const noexcept;
    float mAP() const noexcept;
    float microAvgF1() const noexcept;

private:
    void forward(Model<D>& model, const PxCpuVector& input);

    void processDetects(const Detections& detects, const GroundTruths& gts);
    GroundTruthVec::size_type bestGroundTruth(const Detection& detection, const GroundTruthVec& gtv);
    float iou(const Detection& detection, const GroundTruth& truth);

    ConfusionMatrix matrix_;
    std::unordered_set<int> classesSeen_;

    float threshold_;
};

template<Device D>
void Validator<D>::reset() noexcept
{
    matrix_.reset();
    classesSeen_.clear();
}

template<Device D>
Validator<D>::Validator(float threshold, int numClasses) : threshold_(threshold), matrix_(numClasses)
{
}

template<Device D>
float Validator<D>::microAvgF1() const noexcept
{
    return matrix_.microAvgF1();
}

template<Device D>
float Validator<D>::mAP() const noexcept
{
    return matrix_.mAP(classesSeen_);
}

template<Device D>
float Validator<D>::avgRecall() const noexcept
{
    return matrix_.avgRecall(classesSeen_);
}

template<Device D>
void Validator<D>::validate(Model<D>& model, TrainBatch&& batch)
{
    model.setTraining(false);
    model.setThreshold(threshold_);

    const PxCpuVector& input = batch.imageData();

    forward(model, input);

    processDetects(model.detections(), batch.groundTruth());

    std::cout << "." << std::flush;

    model.setTraining(true);
}

template<Device D>
void Validator<D>::forward(Model<D>& model, const PxCpuVector& input)
{
    model.forward(input);
}

template<Device D>
float Validator<D>::iou(const Detection& detection, const GroundTruth& truth)
{
    auto dbox = DarkBox(detection.box());

    auto iou = dbox.iou(truth.box);

    return iou;
}

template<Device D>
GroundTruthVec::size_type Validator<D>::bestGroundTruth(const Detection& detection, const GroundTruthVec& gts)
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

template<Device D>
void Validator<D>::processDetects(const Detections& detects, const GroundTruths& gts)
{
    auto results = nms(detects, 0.2f);

    for (std::size_t b = 0; b < gts.size(); ++b) {
        auto gtv = gts[b];   // copy the ground truth vector

        for (const auto& detect: results) {
            if (detect.batchId() != b) {
                continue;
            }

            classesSeen_.emplace(detect.classIndex());
            auto index = bestGroundTruth(detect, gtv);
            if (index < gtv.size()) {
                classesSeen_.emplace(gtv[index].classId);
                matrix_.update(gtv[index].classId, detect.classIndex());   // true or false positive
                gtv.erase(gtv.begin() + index);
            } else {
                matrix_.update(-1, detect.classIndex());    // this is a "ghost prediction", a false positive
            }
        }

        for (const auto& gt: gtv) { // these are the "undetected objects", a false negative
            classesSeen_.emplace(gt.classId);
            matrix_.update(gt.classId, -1);
        }
    }
}

}   // px

#ifdef USE_CUDA

#include "cuda/Validator.h"

#endif

