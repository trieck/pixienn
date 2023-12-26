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

#include <cblas.h>

#include "Activation.h"
#include "Box.h"
#include "CpuUtil.h"
#include "Error.h"
#include "Math.h"
#include "Model.h"
#include "RegionLayer.h"

namespace px {

RegionLayer::RegionLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
}

void RegionLayer::setup()
{
    anchors_ = property<std::vector<float>>("anchors");
    biasMatch_ = property<bool>("bias_match", false);
    classScale_ = property<float>("class_scale", 1.0f);
    coordScale_ = property<float>("coord_scale", 1.0f);
    coords_ = property<int>("coords", 4);
    noObjectScale_ = property<float>("noobject_scale", 1.0f);
    num_ = property<int>("num", 1);
    objectScale_ = property<float>("object_scale", 1.0f);
    rescore_ = property<bool>("rescore", false);
    softmax_ = property<bool>("softmax", false);
    thresh_ = property<float>("thresh", 0.5f);

    setOutChannels(num_ * (classes() + coords_ + 1));
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(height() * width() * num_ * (classes() + coords_ + 1));

    biases_ = PxCpuTensor<1>({ (size_t) num_ * 2 }, 0.5f);
    biasUpdates_ = PxCpuTensor<1>({ (size_t) num_ * 2 }, 0.0f);

    output_ = PxCpuVector(batch() * outputs(), 0.0f);
    delta_ = PxCpuVector(batch() * outputs(), 0.0f);

    PX_CHECK(anchors_.size() == num_ * 2, "Anchors size does not match number of regions");

    for (auto i = 0; i < num_ * 2; ++i) {
        biases_[i] = anchors_[i];
    }
}

std::ostream& RegionLayer::print(std::ostream& os)
{
    Layer::print(os, "region", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void RegionLayer::resetStats()
{
    avgAnyObj_ = avgCat_ = avgObj_ = avgIoU_ = recall_ = 0;
    count_ = classCount_ = 0;
}

void RegionLayer::forward(const PxCpuVector& input)
{
    Layer::forward(input);

    output_.copy(input);

    auto size = coords_ + classes() + 1;
    auto* poutput = output_.data();

    flatten(poutput, width() * height(), size * num_, batch(), 1);

    for (auto b = 0; b < batch(); ++b) {
        for (auto i = 0; i < height() * width() * num_; ++i) {
            auto index = size * i + b * outputs();
            poutput[index + 4] = logistic_.apply(poutput[index + 4]);

            if (softmax_) {
                softmax(poutput + index + 5, classes(), 1, poutput + index + 5, 1);
            }
        }
    }

    if (inferring()) {
        return;
    }

    resetStats();

    for (auto b = 0; b < batch(); ++b) {
        for (auto j = 0; j < height(); ++j) {
            for (auto i = 0; i < width(); ++i) {
                processRegion(b, i, j);
            }
        }
        processObjects(b);
    }

    flatten(delta_.data(), width() * height(), size * num_, batch(), 0);

    cost_ = std::pow(magArray(delta_.data(), delta_.size()), 2);

    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n",
           avgIoU_ / count_,
           avgCat_ / classCount_,
           avgObj_ / count_,
           avgAnyObj_ / (width() * height() * num_ * batch()),
           recall_ / count_,
           count_);
}

void RegionLayer::processObjects(int b)
{
    auto* poutput = output_.data();
    auto* pdelta = delta_.data();

    auto size = coords_ + classes() + 1;

    for (const auto& gt: groundTruth(b)) {
        auto bestIoU = -std::numeric_limits<float>::max();
        auto bestIndex = 0;
        auto bestN = 0;

        auto i = static_cast<int>(gt.box.x * width());
        auto j = static_cast<int>(gt.box.y * height());

        auto truthShift(gt.box);
        truthShift.x = 0;
        truthShift.y = 0;

        for (auto n = 0; n < num_; ++n) {
            auto index = size * (j * width() * num_ + i * num_ + n) + b * outputs();
            auto pred = regionBox(n, index, i, j);
            if (biasMatch_) {
                pred.width = biases_[2 * n] / width();
                pred.height = biases_[2 * n + 1] / height();
            }

            pred.x = 0;
            pred.y = 0;

            auto iou = boxIoU(pred, truthShift);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestIndex = index;
                bestN = n;
            }
        }

        float iou = deltaRegionBox(gt.box, bestN, bestIndex, i, j, coordScale_);
        if (iou > 0.5) {
            recall_ += 1;
        }

        avgIoU_ += iou;
        avgObj_ += poutput[bestIndex + 4];

        pdelta[bestIndex + 4] = objectScale_ * (1 - poutput[bestIndex + 4])
                                * logistic_.gradient(poutput[bestIndex + 4]);

        if (rescore_) {
            pdelta[bestIndex + 4] = objectScale_ * (iou - poutput[bestIndex + 4])
                                 * logistic_.gradient(poutput[bestIndex + 4]);
        }

        deltaRegionClass(bestIndex + 5, gt.classId, classScale_);

        ++count_;
        ++classCount_;
    }
}

void RegionLayer::deltaRegionClass(int index, int classId, float scale)
{
    auto* output = output_.data();
    auto* delta = delta_.data();

    for (auto n = 0; n < classes(); ++n) {
        float netTruth = (n == classId) ? 1.0f : 0.0f;
        delta[index + n] = scale * (netTruth - output[index + n]);
        if (n == classId) {
            avgCat_ += output[index + n];
        }
    }
}

float RegionLayer::deltaRegionBox(const cv::Rect2f& truth, int n, int index, int i, int j, float scale)
{
    auto pred = regionBox(n, index, i, j);
    auto iou = boxIoU(pred, truth);

    auto* x = output_.data();
    auto* biases = biases_.data();
    auto* delta = delta_.data();

    auto tx = truth.x * width() - i;
    auto ty = truth.y * height() - j;
    auto tw = std::log(truth.width * width() / biases[2 * n]);
    auto th = std::log(truth.height * height() / biases[2 * n + 1]);

    delta[index + 0] = scale * (tx - logistic_.apply(x[index + 0]))
                       * logistic_.gradient(logistic_.apply(x[index + 0]));

    delta[index + 1] = scale * (ty - logistic_.apply(x[index + 1]))
                       * logistic_.gradient(logistic_.apply(x[index + 1]));

    delta[index + 2] = scale * (tw - x[index + 2]);
    delta[index + 3] = scale * (th - x[index + 3]);

    return iou;
}

void RegionLayer::processRegion(int b, int i, int j)
{
    const auto size = coords_ + classes() + 1;

    auto* poutput = output_.data();
    auto* pdelta = delta_.data();
    auto* pbias = biases_.data();

    for (auto n = 0; n < num_; ++n) {
        auto index = size * (j * width() * num_ + i * num_ + n) + b * outputs();
        avgAnyObj_ += poutput[index + 4];

        pdelta[index + 4] = noObjectScale_ * ((0 - poutput[index + 4]) * logistic_.gradient(poutput[index + 4]));

        auto pred = regionBox(n, index, i, j);
        auto iou = bestIoU(b, pred);
        if (iou > thresh_) {
            pdelta[index + 4] = 0;
        }

        if (model().seen() < 12800) {   // magic!
            cv::Rect2f truth;
            truth.x = (i + .5) / width();
            truth.y = (j + .5) / height();
            truth.width = pbias[2 * n] / width();
            truth.height = pbias[2 * n + 1] / height();

            deltaRegionBox(truth, n, index, i, j, 0.01f);
        }
    }
}

cv::Rect2f RegionLayer::regionBox(int n, int index, int i, int j)
{
    auto* x = output_.data();
    auto* biases = biases_.data();

    cv::Rect2f box;
    box.x = (i + logistic_.apply(x[index + 0])) / width();
    box.y = (j + logistic_.apply(x[index + 1])) / height();
    box.width = std::exp(x[index + 2]) * biases[2 * n] / width();
    box.height = std::exp(x[index + 3]) * biases[2 * n + 1] / height();

    return box;
}

void RegionLayer::backward(const PxCpuVector& input)
{
    auto* pDelta = delta_.data();
    auto* pNetDelta = model().delta();

    PX_CHECK(pNetDelta != nullptr, "Model delta tensor is null");
    PX_CHECK(pNetDelta->data() != nullptr, "Model delta tensor is null");
    PX_CHECK(pDelta != nullptr, "Delta tensor is null");

    const auto n = batch() * inputs();

    PX_CHECK(delta_.size() >= n, "Delta tensor is too small");
    PX_CHECK(pNetDelta->size() >= n, "Model tensor is too small");

    cblas_saxpy(n, 1, pDelta, 1, pNetDelta->data(), 1);
}

void RegionLayer::addDetects(Detections& detections, float threshold)
{
    const auto* predictions = output_.data();

    for (auto i = 0; i < width() * height(); ++i) {
        auto row = i / width();
        auto col = i % width();

        for (auto n = 0; n < num_; ++n) {
            auto index = i * num_ + n;
            auto pindex = index * (classes() + coords_ + 1) + coords_;
            auto scale = predictions[pindex];
            auto boxIndex = index * (classes() + coords_ + 1);

            auto box = regionBox(n, boxIndex, col, row);
            int left = box.x - box.width / 2;
            int right = box.x + box.width / 2;
            int top = box.y - box.height / 2;
            int bottom = box.y + box.height / 2;

            auto clsIndex = index * (classes() + coords_ + 1) + coords_ + 1;
            for (auto j = 0; j < classes(); ++j) {
                auto prob = scale * predictions[clsIndex + j];
                if (prob >= threshold) {
                    cv::Rect b{ left, top, right - left, bottom - top };
                    Detection det(b, j, prob);
                    detections.emplace_back(std::move(det));
                }
            }
        }
    }
}

void RegionLayer::addDetects(Detections& detections, int w, int h, float threshold)
{
    const auto* predictions = output_.data();

    for (auto i = 0; i < width() * height(); ++i) {
        auto row = i / width();
        auto col = i % width();

        for (auto n = 0; n < num_; ++n) {
            auto index = i * num_ + n;
            auto pindex = index * (classes() + coords_ + 1) + coords_;
            auto scale = predictions[pindex];
            auto boxIndex = index * (classes() + coords_ + 1);

            auto box = regionBox(n, boxIndex, col, row);
            int left = (box.x - box.width / 2) * w;
            int right = (box.x + box.width / 2) * w;
            int top = (box.y - box.height / 2) * h;
            int bottom = (box.y + box.height / 2) * h;

            left = std::max(0, left);
            right = std::min(w - 1, right);
            top = std::max(0, top);
            bottom = std::min(h - 1, bottom);

            auto clsIndex = index * (classes() + coords_ + 1) + coords_ + 1;
            for (auto j = 0; j < classes(); ++j) {
                auto prob = scale * predictions[clsIndex + j];
                if (prob >= threshold) {
                    cv::Rect b{ left, top, right - left, bottom - top };
                    Detection det(b, j, prob);
                    detections.emplace_back(std::move(det));
                }
            }
        }
    }
}

float RegionLayer::bestIoU(int b, const cv::Rect2f& pred)
{
    const auto& gt = groundTruth(b);

    auto bestIoU = -std::numeric_limits<float>::max();

    for (const auto& g: gt) {
        auto iou = boxIoU(pred, g.box);
        if (iou > bestIoU) {
            bestIoU = iou;
        }
    }

    return bestIoU;
}

}   // px
