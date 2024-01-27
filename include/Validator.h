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
    Validator(float threshold);

    using V = typename DeviceTraits<D>::VectorType;

    void validate(Model<D>& model, TrainBatch&& batch);

    float avgRecall() const noexcept;
    float mAP() const noexcept;
    float microAvgF1() const noexcept;

private:
    void forward(Model<D>& model, PxCpuVector& input);

    void processDetects(const Detections& detects, const GroundTruthVec& gts);
    GroundTruthVec::size_type findGroundTruth(const Detection& detection, const GroundTruthVec& gts);
    float iou(const Detection& detection, const GroundTruth& truth);

    ConfusionMatrix matrix_;
    std::unordered_set<int> classesSeen_;
    float threshold_;
};

template<Device D>
Validator<D>::Validator(float threshold) : threshold_(threshold)
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
    return matrix_.mAP(classesSeen_.size());
}

template<Device D>
float Validator<D>::avgRecall() const noexcept
{
    return matrix_.avgRecall(classesSeen_.size());
}

template<Device D>
void Validator<D>::validate(Model<D>& model, TrainBatch&& batch)
{
    classesSeen_.clear();
    matrix_.resize(model.classes());
    model.setTraining(false);
    model.setThreshold(threshold_);

    auto inputSize = batch.height() * batch.width() * batch.channels();
    PxCpuVector input(inputSize, 0.0f);

    for (auto i = 0; i < batch.batchSize(); ++i) {
        auto image = batch.slice(i);
        input.copyHost(image, inputSize);

        forward(model, input);

        processDetects(model.detections(), batch.groundTruth(i));

        std::cout << '.' << std::flush;
    }

    model.setTraining(true);
}

template<Device D>
void Validator<D>::forward(Model<D>& model, PxCpuVector& input)
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
GroundTruthVec::size_type Validator<D>::findGroundTruth(const Detection& detection, const GroundTruthVec& gts)
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
void Validator<D>::processDetects(const Detections& detects, const GroundTruthVec& gts)
{
    auto results = nms(detects, 0.2f);

    GroundTruthVec gtsCopy(gts);

    for (const auto& detect: results) {
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

}

#ifdef USE_CUDA

#include "cuda/Validator.h"

#endif

