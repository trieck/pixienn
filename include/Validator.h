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
    Validator(float confidenceThreshold, float apConfidenceThreshold, float iouThreshold, float nmsThreshold,
              int numClasses);

    using V = typename DeviceTraits<D>::VectorType;

    void validate(Model<D>& model, const MiniBatch& batch);
    void reset() noexcept;

    float avgRecall() const noexcept;
    float mAP() const noexcept;
    float microAvgF1() const noexcept;
    struct PRPoint { float confidence; float precision; float recall; };
    std::vector<PRPoint> microPRCurve() const;
    float avgLoss() const noexcept;
    float accuracy() const noexcept;

private:
    void forward(Model<D>& model, const PxCpuVector& input);

    void processDetects(const Detections& detects, const GroundTruths& gts, std::size_t validBatchSize);
    GroundTruthVec::size_type findGroundTruth(const Detection& detection, const GroundTruthVec& gtv,
                                              int classIndex = -1);
    float iou(const Detection& detection, const GroundTruth& truth);

    ConfusionMatrix matrix_;
    std::unordered_set<int> classesSeen_;

    struct RankedPrediction
    {
        float confidence;
        bool truePositive;
    };
    std::vector<std::vector<RankedPrediction>> predictions_;
    std::vector<std::size_t> groundTruthCounts_;

    float confidenceThreshold_, apConfidenceThreshold_, iouThreshold_, nmsThreshold_;
    float totalLoss_ = 0.0f;
    int seen_ = 0;
    int correctPredictions_ = 0;
    int totalPredictions_ = 0;
};

template<Device D>
Validator<D>::Validator(float confidenceThreshold, float apConfidenceThreshold, float iouThreshold,
                        float nmsThreshold, int numClasses)
        : matrix_(numClasses), predictions_(numClasses), groundTruthCounts_(numClasses, 0),
          confidenceThreshold_(confidenceThreshold), apConfidenceThreshold_(apConfidenceThreshold),
          iouThreshold_(iouThreshold), nmsThreshold_(nmsThreshold),
          totalLoss_(0.0f), seen_(0), correctPredictions_(0),
          totalPredictions_(0)
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
    float totalAP = 0.0f;
    std::size_t classes = 0;

    for (std::size_t cls = 0; cls < groundTruthCounts_.size(); ++cls) {
        if (groundTruthCounts_[cls] == 0) {
            continue;
        }
        ++classes;
        auto ranked = predictions_[cls];
        std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.confidence > rhs.confidence;
        });

        std::vector<float> precision(ranked.size()), recall(ranked.size());
        float tp = 0.0f, fp = 0.0f;
        for (std::size_t i = 0; i < ranked.size(); ++i) {
            ranked[i].truePositive ? ++tp : ++fp;
            precision[i] = tp / (tp + fp);
            recall[i] = tp / groundTruthCounts_[cls];
        }
        for (std::size_t i = precision.size(); i > 1; --i) {
            precision[i - 2] = std::max(precision[i - 2], precision[i - 1]);
        }
        float previousRecall = 0.0f;
        for (std::size_t i = 0; i < ranked.size(); ++i) {
            totalAP += (recall[i] - previousRecall) * precision[i];
            previousRecall = recall[i];
        }
    }
    return classes == 0 ? 0.0f : totalAP / classes;
}

template<Device D>
std::vector<typename Validator<D>::PRPoint> Validator<D>::microPRCurve() const
{
    constexpr std::size_t samples = 201;
    std::vector<PRPoint> curve;
    curve.reserve(samples);

    std::size_t totalGroundTruths = 0;
    for (const auto count: groundTruthCounts_) {
        totalGroundTruths += count;
    }

    for (std::size_t sample = 0; sample < samples; ++sample) {
        const auto confidence = static_cast<float>(sample) / static_cast<float>(samples - 1);
        std::size_t truePositives = 0;
        std::size_t falsePositives = 0;
        for (const auto& ranked: predictions_) {
            for (const auto& prediction: ranked) {
                if (prediction.confidence < confidence) {
                    continue;
                }
                if (prediction.truePositive) {
                    ++truePositives;
                } else {
                    ++falsePositives;
                }
            }
        }

        const auto predicted = truePositives + falsePositives;
        curve.push_back({
                confidence,
                predicted == 0 ? 0.0f : static_cast<float>(truePositives) / static_cast<float>(predicted),
                totalGroundTruths == 0 ? 0.0f : static_cast<float>(truePositives) / static_cast<float>(totalGroundTruths)});
    }
    return curve;
}

template<Device D>
float Validator<D>::avgRecall() const noexcept
{
    return matrix_.avgRecall(classesSeen_);
}

template<Device D>
float Validator<D>::avgLoss() const noexcept
{
    if (seen_ == 0) {
        return 0;
    }

    return static_cast<float>(totalLoss_) / seen_;
}

template<Device D>
float Validator<D>::accuracy() const noexcept
{
    if (totalPredictions_ == 0) {
        return 0;
    }

    return static_cast<float>(correctPredictions_) / totalPredictions_;
}

template<Device D>
void Validator<D>::reset() noexcept
{
    matrix_.reset();
    classesSeen_.clear();
    for (auto& predictions: predictions_) {
        predictions.clear();
    }
    std::fill(groundTruthCounts_.begin(), groundTruthCounts_.end(), 0);
    totalLoss_ = 0.0f;
    seen_ = 0;
    correctPredictions_ = 0;
    totalPredictions_ = 0;
}

template<Device D>
void Validator<D>::validate(Model<D>& model, const MiniBatch& batch)
{
    const auto previousMode = model.mode();
    const auto previousThreshold = model.threshold();

    model.setMode(Mode::VALIDATING);

    // AP needs the low-confidence tail to build a useful precision/recall
    // curve. F1 and accuracy still use confidenceThreshold_ below.
    model.setThreshold(std::min(confidenceThreshold_, apConfidenceThreshold_));

    try {
        const PxCpuVector& input = batch.imageData();

        forward(model, input);

        totalLoss_ += model.cost();
        seen_++;

        processDetects(model.detections(), batch.groundTruth(), batch.validSize());

        std::cout << "." << std::flush;
    } catch (...) {
        model.setThreshold(previousThreshold);
        model.setMode(previousMode);
        throw;
    }

    model.setThreshold(previousThreshold);
    model.setMode(previousMode);
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
GroundTruthVec::size_type Validator<D>::findGroundTruth(const Detection& detection, const GroundTruthVec& gts,
                                                        int classIndex)
{
    auto bestIt = std::end(gts);
    auto bestIoU = -std::numeric_limits<float>::max();

    for (auto it = std::cbegin(gts); it != std::cend(gts); it++) {
        if (classIndex >= 0 && it->classId != classIndex) {
            continue;
        }
        auto IoU = iou(detection, *it);
        if (IoU > bestIoU) {
            bestIoU = IoU;
            bestIt = it;
        }
    }

    if (bestIoU < iouThreshold_) {   // no match
        return gts.size();
    }

    return std::distance(gts.cbegin(), bestIt);
}

template<Device D>
void Validator<D>::processDetects(const Detections& detects, const GroundTruths& gts,
                                  std::size_t validBatchSize)
{
    auto results = nms(detects, nmsThreshold_);
    // AP must use the same class-aware NMS policy as inference/Darknet's
    // exported detections. Keep the low AP confidence cutoff below so score
    // ordering still includes the tail needed to construct the PR curve.
    auto apResults = results;
    std::stable_sort(apResults.begin(), apResults.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.prob() > rhs.prob();
    });

    std::sort(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.prob() > rhs.prob();
    });

    const auto images = std::min(validBatchSize, gts.size());
    for (std::size_t b = 0; b < images; ++b) {
        auto gtv = gts[b];   // copy the ground truth vector

        for (const auto& gt: gtv) {
            classesSeen_.emplace(gt.classId);
            ++groundTruthCounts_.at(gt.classId);
        }

        std::vector<bool> apMatched(gtv.size(), false);
        for (const auto& detect: apResults) {
            if (detect.batchId() != static_cast<int>(b) || detect.prob() < apConfidenceThreshold_) {
                continue;
            }
            auto bestIndex = gtv.size();
            auto bestIoU = iouThreshold_;
            for (std::size_t i = 0; i < gtv.size(); ++i) {
                if (apMatched[i] || gtv[i].classId != detect.classIndex()) {
                    continue;
                }
                const auto overlap = iou(detect, gtv[i]);
                if (overlap >= bestIoU) {
                    bestIoU = overlap;
                    bestIndex = i;
                }
            }
            const auto truePositive = bestIndex < gtv.size();
            if (truePositive) {
                apMatched[bestIndex] = true;
            }
            predictions_.at(detect.classIndex()).push_back({ detect.prob(), truePositive });
        }

        for (const auto& detect: results) {
            if (detect.batchId() != static_cast<int>(b) || detect.prob() < confidenceThreshold_) {
                continue;
            }

            classesSeen_.emplace(detect.classIndex());
            auto index = findGroundTruth(detect, gtv, detect.classIndex());
            if (index < gtv.size()) {
                auto trueClass = gtv[index].classId;
                classesSeen_.emplace(trueClass);
                matrix_.update(trueClass, detect.classIndex());   // true or false positive
                gtv.erase(gtv.begin() + index);

                if (detect.classIndex() == trueClass) {
                    correctPredictions_++;
                }
            } else {
                matrix_.update(-1, detect.classIndex());    // this is a "ghost prediction", a false positive
            }

            totalPredictions_++;
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
