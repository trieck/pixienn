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

#include <cblas.h>

#include "DetectLayer.h"
#include "Model.h"
#include "Math.h"
#include "Box.h"

namespace px {

static cv::Rect2f makeBox(const float* pbox)
{
    cv::Rect2f box;
    box.x = *pbox++;
    box.y = *pbox++;
    box.width = *pbox++;
    box.height = *pbox++;

    return box;
}

DetectLayer::DetectLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
}

void DetectLayer::setup()
{
    classScale_ = property<float>("class_scale", 1.0f);
    coordScale_ = property<float>("coord_scale", 1.0f);
    coords_ = property<int>("coords", 1);
    forced_ = property<bool>("forced", false);
    jitter_ = property<float>("jitter", 0.2f);
    noObjectScale_ = property<float>("noobject_scale", 1.0f);
    num_ = property<int>("num", 1);
    objectScale_ = property<float>("object_scale", 1.0f);
    random_ = property<bool>("random", false);
    reorg_ = property<bool>("reorg", false);
    rescore_ = property<bool>("rescore", false);
    side_ = property<int>("side", 7);
    softmax_ = property<bool>("softmax", false);
    sqrt_ = property<bool>("sqrt", false);

    setOutChannels(channels());
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(inputs());

#ifdef USE_CUDA
    if (useGpu()) {
        outputGpu_ = PxCudaVector(outputs(), 0.0f);
    } else {
        output_ = PxCpuVector(batch() * outputs(), 0.0f);
        delta_ = PxCpuVector(batch() * outputs(), 0.0f);
    }
#else
    output_ = PxCpuVector(batch() * outputs(), 0.0f);
    delta_ = PxCpuVector(batch() * outputs(), 0.0f);
#endif
}

std::ostream& DetectLayer::print(std::ostream& os)
{
    Layer::print(os, "detection", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void DetectLayer::forward(const PxCpuVector& input)
{
    Layer::forward(input);

    if (softmax_) {
        output_ = softmax(input);
    } else {
        output_.copy(input);
    }

    if (!training()) {
        return;
    }

    float avgIou = 0, avgCat = 0, avgAllcat = 0, avgObj = 0, avgAnyObj = 0;
    auto count = 0;
    auto size = batch() * inputs();
    auto nclasses = classes();
    const auto& imgbatch = imageBatch();

    auto locations = side_ * side_;
    auto* poutput = output_.data();
    auto* pdelta = delta_.data();

    for (auto b = 0; b < batch(); ++b) {
        auto index = b * inputs();
        auto gts = imgbatch.groundTruth(b);
        for (auto i = 0; i < locations; ++i) {
            auto bestIndex = -1;
            auto bestIou = -1.0f;
            const GroundTruth* bestTruth = nullptr;

            for (auto j = 0; j < num_; ++j) {
                auto pindex = index + locations * nclasses + i * num_ + j;
                pdelta[pindex] = noObjectScale_ * (0 - poutput[pindex]);
                cost_ += noObjectScale_ * std::pow(poutput[pindex], 2);
                avgAnyObj += poutput[pindex];

                auto boxIndex = index + locations * (nclasses + num_) + (i * num_ + j) * coords_;
                auto pred = makeBox(poutput + boxIndex);
                pred.x /= side_;
                pred.y /= side_;

                if (sqrt_) {
                    pred.width *= pred.width;
                    pred.height *= pred.height;
                }

                for (const auto& gt: gts) {
                    cv::Rect2f truthBox(gt.box);
                    truthBox.x /= side_;
                    truthBox.y /= side_;

                    auto iou = boxIou(pred, truthBox);
                    if (iou > bestIou) {
                        bestIou = iou;
                        bestIndex = j;
                        bestTruth = &gt;
                    }
                }
            }

            if (bestTruth == nullptr) {
                continue;
            }

            cv::Rect2f truthBox(bestTruth->box);
            truthBox.x /= side_;
            truthBox.y /= side_;

            auto row = i / side_;
            auto col = i % side_;
            auto truthRow = (int) (bestTruth->box.y * side_);
            auto truthCol = (int) (bestTruth->box.x * side_);

            // Now, we can go back and compute the class loss, etc.
            auto classIndex = index + i * nclasses;
            for (auto j = 0; j < nclasses; ++j) {
                float netTruth = bestTruth->classId == j ? 1.0f : 0.0f;
                pdelta[classIndex + j] = classScale_ * (netTruth - poutput[classIndex + j]);
                cost_ += classScale_ * pow(netTruth - poutput[classIndex + j], 2);
                if (netTruth) avgCat += poutput[classIndex + j];
                avgAllcat += poutput[classIndex + j];
            }

            auto boxIndex = index + locations * (nclasses + num_) + (i * num_ + bestIndex) * coords_;
            auto pred = makeBox(poutput + boxIndex);
            pred.x /= side_;
            pred.y /= side_;
            if (sqrt_) {
                pred.width *= pred.width;
                pred.height *= pred.height;
            }

            auto iou = boxIou(pred, truthBox);
            auto pindex = index + locations * nclasses + i * num_ + bestIndex;
            cost_ -= noObjectScale_ * std::pow(poutput[pindex], 2);
            cost_ += objectScale_ * std::pow(1 - poutput[pindex], 2);
            avgObj += poutput[pindex];
            pdelta[pindex] = objectScale_ * (1. - poutput[pindex]);

            if (rescore_) {
                pdelta[pindex] = objectScale_ * (iou - poutput[pindex]);
            }

            pdelta[boxIndex + 0] = coordScale_ * (bestTruth->box.x - poutput[boxIndex + 0]);
            pdelta[boxIndex + 1] = coordScale_ * (bestTruth->box.y - poutput[boxIndex + 1]);
            pdelta[boxIndex + 2] = coordScale_ * (bestTruth->box.width - poutput[boxIndex + 2]);
            pdelta[boxIndex + 3] = coordScale_ * (bestTruth->box.height - poutput[boxIndex + 3]);

            if (sqrt_) {
                pdelta[boxIndex + 2] = coordScale_ * (std::sqrt(bestTruth->box.width) - poutput[boxIndex + 2]);
                pdelta[boxIndex + 3] = coordScale_ * (std::sqrt(bestTruth->box.height) - poutput[boxIndex + 3]);
            }

            cost_ += std::pow(1 - iou, 2);
            avgIou += iou;
            ++count;
        }
    }

    // what was the point of all the cost stuff we just did???
    cost_ = std::pow(magArray(pdelta, batch() * outputs()), 2);

    if (count == 0) {
        printf("Detection Avg IOU: -----, Pos Cat: -----, All Cat: -----, Pos Obj: -----, Any Obj: %.2f, Count: 0\n",
               avgAnyObj / (batch() * locations * num_));
    } else {
        printf("Detection Avg IOU: %.3f, Pos Cat: %.3f, All Cat: %.3f, Pos Obj: %.3f, Any Obj: %.3f, Count: %d\n",
               avgIou / count,
               avgCat / count,
               avgAllcat / (count * nclasses),
               avgObj / count,
               avgAnyObj / (batch() * locations * num_),
               count);
    }
}

void DetectLayer::backward(const PxCpuVector& input)
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

#ifdef USE_CUDA
void DetectLayer::forwardGpu(const PxCudaVector& input)
{
    outputGpu_.copy(input);
}
#endif  // USE_CUDA

void DetectLayer::addDetects(Detections& detections, int width, int height, float threshold)
{
    addDetects(detections, width, height, threshold, output_.data());
}

#ifdef USE_CUDA

void DetectLayer::addDetectsGpu(Detections& detections, int width, int height, float threshold)
{
    auto predv = outputGpu_.asVector();
    addDetects(detections, width, height, threshold, predv.data());
}

#endif  // USE_CUDA

void
DetectLayer::addDetects(Detections& detections, int width, int height, float threshold, const float* predictions) const
{
    auto nclasses = classes();

    const auto locations = side_ * side_;
    for (auto i = 0; i < locations; ++i) {
        auto row = i / side_;
        auto col = i % side_;
        for (auto n = 0; n < num_; ++n) {
            auto pindex = locations * nclasses + i * num_ + n;
            auto scale = predictions[pindex];
            auto bindex = locations * (nclasses + num_) + (i * num_ + n) * 4;
            auto x = (predictions[bindex + 0] + col) / side_ * width;
            auto y = (predictions[bindex + 1] + row) / side_ * height;
            auto w = pow(predictions[bindex + 2], (sqrt_ ? 2 : 1)) * width;
            auto h = pow(predictions[bindex + 3], (sqrt_ ? 2 : 1)) * height;
            auto left = std::max<int>(0, (x - w / 2));
            auto right = std::min<int>(width - 1, (x + w / 2));
            auto top = std::max<int>(0, (y - h / 2));
            auto bottom = std::min<int>(height - 1, (y + h / 2));
            cv::Rect b{ left, top, right - left, bottom - top };

            Detection det(nclasses, b, scale);
            int max = 0;
            for (auto j = 0; j < nclasses; ++j) {
                auto index = i * nclasses;
                det[j] = scale * predictions[index + j];
                if (det[j] > det[max]) {
                    max = j;
                }
            }
            if (det[max] >= threshold) {
                det.setMaxClass(max);
                detections.emplace_back(std::move(det));
            }
        }
    }
}

}   // px
