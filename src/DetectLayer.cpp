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
}

std::ostream& DetectLayer::print(std::ostream& os)
{
    os << std::setfill('.');

    os << std::setw(100) << std::left << "detection" << std::endl;

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

void DetectLayer::addDetects(std::vector<Detection>& detections, int width, int height, float threshold)
{
    const auto* predictions = output_.data();

    for (auto i = 0; i < side_ * side_; ++i) {
        auto row = i / side_;
        auto col = i % side_;

        for (auto n = 0; n < num_; ++n) {
            auto pindex = side_ * side_ * classes_ + i * num_ + n;
            auto scale = predictions[pindex];

            auto bindex = side_ * side_ * (classes_ + num_) + (i * num_ + n) * 4;

            cv::Rect2f b;
            b.x = (predictions[bindex + 0] + col) / side_ * width;
            b.y = (predictions[bindex + 1] + row) / side_ * height;
            b.width = pow(predictions[bindex + 2], (sqrt_ ? 2 : 1)) * width;
            b.height = pow(predictions[bindex + 3], (sqrt_ ? 2 : 1)) * height;

            Detection det(classes_, b, scale);
            for (auto j = 0; j < classes_; ++j) {
                auto index = i * classes_;
                auto prob = scale * predictions[index + j];
                det[j] = prob >= threshold ? prob : 0;
            }

            detections.emplace_back(std::move(det));
        }
    }
}

}   // px
