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

#include "YoloKernels.cuh"
#include "YoloTargetBuilder.h"

namespace px {

template<>
class YoloExtras<Device::CUDA>
{
protected:
    PxCpuVector hostAssignedBoxes_;
    PxCpuVectorT<int> hostAssignedClasses_;
    PxCpuVectorT<int> hostAssignedAnchors_;
    PxCudaVector assignedBoxes_;
    PxCudaVectorT<int> assignedClasses_;
    PxCudaVectorT<int> assignedAnchors_;
    PxCudaVectorT<int> masksGpu_;
    PxCudaVector anchorsGpu_;
    PxCudaVector statsGpu_;
    PxCudaVector costGpu_;
    PxCpuVector hostTruths_;
    PxCpuVectorT<int> hostTruthCounts_;
    PxCudaVector truthsGpu_;
    PxCudaVectorT<int> truthCountsGpu_;
    std::unique_ptr<YoloTargetBuilder> targetBuilder_;
    unsigned truthCapacity_ = 0;
};

template<>
inline void YoloLayer<Device::CUDA>::setup()
{
    const auto slots = this->batch() * numMasks_ * this->width() * this->height();
    hostAssignedBoxes_ = PxCpuVector(slots * 4);
    hostAssignedClasses_ = PxCpuVectorT<int>(slots, -1);
    hostAssignedAnchors_ = PxCpuVectorT<int>(slots, -1);
    assignedBoxes_ = PxCudaVector(hostAssignedBoxes_.size());
    assignedClasses_ = PxCudaVectorT<int>(slots);
    assignedAnchors_ = PxCudaVectorT<int>(slots);
    masksGpu_ = PxCudaVectorT<int>(mask_.data(), mask_.data() + mask_.size());
    PxCpuVector anchorValues(anchors_.size());
    std::transform(anchors_.begin(), anchors_.end(), anchorValues.begin(),
                   [](int value) { return static_cast<float>(value); });
    anchorsGpu_ = PxCudaVector(anchorValues.data(), anchorValues.data() + anchorValues.size());
    statsGpu_ = PxCudaVector(YOLO_STATS_SIZE);
    costGpu_ = PxCudaVector(1);
    hostTruthCounts_ = PxCpuVectorT<int>(this->batch(), 0);
    truthCountsGpu_ = PxCudaVectorT<int>(this->batch());
    targetBuilder_ = std::make_unique<YoloTargetBuilder>(
            anchors_, mask_, this->width(), this->height(),
            this->model().width(), this->model().height());
};

template<>
inline void YoloLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);

    const auto area = this->width() * this->height();
    yoloActivateGpu(input.data(), this->output_.data(), this->batch(), numMasks_,
                    this->classes(), area);
    if (this->inferring()) return;

    hostAssignedClasses_.fill(-1);
    hostAssignedAnchors_.fill(-1);
    hostAssignedBoxes_.fill(0.0f);

    auto maxTruth = 1u;
    for (auto b = 0; b < this->batch(); ++b) {
        maxTruth = std::max(maxTruth, static_cast<unsigned>(this->groundTruth(b).size()));
    }
    if (maxTruth > truthCapacity_) {
        truthCapacity_ = maxTruth;
        hostTruths_ = PxCpuVector(this->batch() * truthCapacity_ * 5, 0.0f);
        truthsGpu_ = PxCudaVector(hostTruths_.size());
    }
    hostTruths_.fill(0.0f);
    hostTruthCounts_.fill(0);

    for (auto b = 0; b < this->batch(); ++b) {
        const auto& truths = this->groundTruth(b);
        hostTruthCounts_[b] = static_cast<int>(truths.size());
        for (auto t = 0u; t < truths.size(); ++t) {
            const auto& gt = truths[t];
            const auto truthBase = (b * truthCapacity_ + t) * 5;
            hostTruths_[truthBase] = gt.box.x();
            hostTruths_[truthBase + 1] = gt.box.y();
            hostTruths_[truthBase + 2] = gt.box.w();
            hostTruths_[truthBase + 3] = gt.box.h();
            hostTruths_[truthBase + 4] = static_cast<float>(gt.classId);
        }
        const auto targets = targetBuilder_->build(truths);
        std::copy(targets.classes.begin(), targets.classes.end(),
                  hostAssignedClasses_.begin() + b * numMasks_ * area);
        std::copy(targets.anchors.begin(), targets.anchors.end(),
                  hostAssignedAnchors_.begin() + b * numMasks_ * area);
        std::copy(targets.boxes.begin(), targets.boxes.end(),
                  hostAssignedBoxes_.begin() + b * numMasks_ * area * 4);
    }

    truthsGpu_.copy(hostTruths_);
    truthCountsGpu_.copy(hostTruthCounts_);
    assignedBoxes_.copy(hostAssignedBoxes_);
    assignedClasses_.copy(hostAssignedClasses_);
    assignedAnchors_.copy(hostAssignedAnchors_);
    this->delta_.fill(0.0f);
    statsGpu_.fill(0.0f);
    costGpu_.fill(0.0f);
    yoloLossGpu(this->output_.data(), this->delta_.data(), truthsGpu_.data(), truthCountsGpu_.data(),
                truthCapacity_, assignedClasses_.data(), assignedAnchors_.data(), assignedBoxes_.data(),
                masksGpu_.data(), anchorsGpu_.data(), statsGpu_.data(), costGpu_.data(),
                this->batch(), numMasks_, numAnchors_, this->classes(), this->width(), this->height(),
                this->model().width(), this->model().height(), ignoreThresh_, truthThresh_, coordScale_,
                objectScale_, noObjectScale_, classScale_);
    const auto cost = costGpu_.asVector();
    this->cost_ = cost[0] / std::max(1, this->batch());

    if (this->training()) {
        const auto stats = statsGpu_.asVector();
        avgIoU_ += stats[0]; recall_ += stats[1]; recall75_ += stats[2];
        avgCat_ += stats[3]; avgObj_ += stats[4]; avgAnyObj_ += stats[5];
        count_ += static_cast<int>(stats[6]); classCount_ += static_cast<int>(stats[7]);
        boxCost_ = stats[8];
        objectCost_ = stats[9];
        noObjectCost_ = stats[10];
        classCost_ = stats[11];
        if (count_ > 0 && this->model().seen() % logInterval_ == 0) {
            writeStats();
            resetStats();
        }
    }
}

template<>
inline void YoloLayer<Device::CUDA>::backward(const V& input, V* grad)
{
    Layer<Device::CUDA>::backward(input, grad);

    if (grad == nullptr) {
        return;
    }

    auto alpha = 1.0f;

    const auto& ctxt = this->cublasContext();

    auto status = cublasSaxpy(ctxt, this->delta_.size(), &alpha, this->delta_.data(), 1, grad->data(), 1);

    PX_CHECK_CUBLAS(status);
}

template<>
inline void YoloLayer<Device::CUDA>::addDetects(Detections& detects, int width, int height, float threshold)
{
    auto vout = output_.asVector();
    auto* pout = vout.data();

    for (auto b = 0; b < this->batch(); ++b) {
        addDetects(detects, b, width, height, threshold, pout);
    }
}

template<>
inline void YoloLayer<Device::CUDA>::addDetects(Detections& detects, float threshold)
{
    auto vout = output_.asVector();
    auto* pout = vout.data();

    for (auto b = 0; b < this->batch(); ++b) {
        addDetects(detects, b, threshold, pout);
    }
}

}   // px
