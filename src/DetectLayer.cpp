/********************************************************************************
* Copyright 2020 Thomas A. Rieck, All Rights Reserved
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

#include "DetectLayer.h"
#include "Math.h"

using namespace xt;

namespace px {

DetectLayer::DetectLayer(const Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
    classScale = property<float>("class_scale", 1.0f);
    classes_ = property<int>("classes", 1);
    coordScale_ = property<float>("coord_scale", 1.0f);
    coords_ = property<int>("coords", 1);
    forced_ = property<bool>("forced", false);
    jitter_ = property<float>("jitter", 0.2f);
    maxBoxes_ = property<int>("max_boxes", 90);
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
    setOutputs(batch() * inputs());

    output_ = empty<float>({ outputs() });

#ifdef USE_CUDA
    outputGpu_ = PxDevVector<float>(outputs());
#endif
}

std::ostream& DetectLayer::print(std::ostream& os)
{
    Layer::print(os, "detection", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void DetectLayer::forward(const xt::xarray<float>& input)
{
    if (softmax_) {
        output_ = softmax(input);
    } else {
        output_ = input;
    }
}

#ifdef USE_CUDA
void DetectLayer::forwardGpu(const PxDevVector<float>& input)
{
    // FIXME:  why doesn't darknet perform a softmax?
    outputGpu_.fromDevice(input);
}
#endif  // USE_CUDA

void DetectLayer::addDetects(Detections& detections, int width, int height, float threshold)
{
    const auto* predictions = output_.data();

    const auto area = side_ * side_;

    for (auto i = 0; i < area; ++i) {
        auto row = i / side_;
        auto col = i % side_;

        for (auto n = 0; n < num_; ++n) {
            auto pindex = area * classes_ + i * num_ + n;
            auto scale = predictions[pindex];

            auto bindex = area * (classes_ + num_) + (i * num_ + n) * 4;

            auto x = (predictions[bindex + 0] + col) / side_ * width;
            auto y = (predictions[bindex + 1] + row) / side_ * height;
            auto w = pow(predictions[bindex + 2], (sqrt_ ? 2 : 1)) * width;
            auto h = pow(predictions[bindex + 3], (sqrt_ ? 2 : 1)) * height;

            auto left = std::max<int>(0, (x - w / 2));
            auto right = std::min<int>(width - 1, (x + w / 2));
            auto top = std::max<int>(0, (y - h / 2));
            auto bottom = std::min<int>(height - 1, (y + h / 2));

            cv::Rect b;
            b.x = left;
            b.y = top;
            b.width = right - left;
            b.height = bottom - top;

            Detection det(classes_, b, scale);
            int max = 0;

            for (auto j = 0; j < classes_; ++j) {
                auto index = i * classes_;
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

#ifdef USE_CUDA
void DetectLayer::addDetectsGpu(Detections& detections, int width, int height, float threshold)
{
    auto predv = outputGpu_.asHost();
    const auto* predictions = predv.data();

    // FIXME:  this is ridiculous
    
    const auto area = side_ * side_;

    for (auto i = 0; i < area; ++i) {
        auto row = i / side_;
        auto col = i % side_;

        for (auto n = 0; n < num_; ++n) {
            auto pindex = area * classes_ + i * num_ + n;
            auto scale = predictions[pindex];

            auto bindex = area * (classes_ + num_) + (i * num_ + n) * 4;

            auto x = (predictions[bindex + 0] + col) / side_ * width;
            auto y = (predictions[bindex + 1] + row) / side_ * height;
            auto w = pow(predictions[bindex + 2], (sqrt_ ? 2 : 1)) * width;
            auto h = pow(predictions[bindex + 3], (sqrt_ ? 2 : 1)) * height;

            auto left = std::max<int>(0, (x - w / 2));
            auto right = std::min<int>(width - 1, (x + w / 2));
            auto top = std::max<int>(0, (y - h / 2));
            auto bottom = std::min<int>(height - 1, (y + h / 2));

            cv::Rect b;
            b.x = left;
            b.y = top;
            b.width = right - left;
            b.height = bottom - top;

            Detection det(classes_, b, scale);
            int max = 0;

            for (auto j = 0; j < classes_; ++j) {
                auto index = i * classes_;
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

#endif // USE_CUDA

}   // px
