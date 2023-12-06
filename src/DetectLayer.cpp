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

namespace px {

DetectLayer::DetectLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
}

void DetectLayer::setup()
{
    classScale = property<float>("class_scale", 1.0f);
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

    assert(side_ * side_ * ((1 + coords_) * num_ + classes()) == inputs());

    setOutChannels(channels());
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(inputs());

#ifdef USE_CUDA
    if (useGpu()) {
        outputGpu_ = PxCudaVector(outputs(), 0.0f);
    } else {
        output_ = PxCpuVector(outputs(), 0.0f);
        delta_ = PxCpuVector(outputs(), 0.0f);
    }
#else
    output_ = PxCpuVector(outputs(), 0.0f);
    delta_ = PxCpuVector(outputs(), 0.0f);
#endif
}

std::ostream& DetectLayer::print(std::ostream& os)
{
    Layer::print(os, "detection", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

uint32_t DetectLayer::truths() const noexcept
{
    auto nclasses = classes();
    auto result = side_ * side_ * (1 + coords_ + nclasses);

    return result;
}

void DetectLayer::forward(const PxCpuVector& input)
{
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
    const auto& gndTruth = truth();

    cost() = 0;
    delta_.fill(0);

    auto locations = side_ * side_;
    auto* poutput = output_.data();
    auto* pdelta = delta_.data();

    for (auto b = 0; b < batch(); ++b) {
        auto index = b * inputs();

        for (auto i = 0; i < locations; ++i) {
            auto truthIndex = (b * locations + i);
            //auto isObject = gndTruth.hasObject(truthIndex);

            for (auto j = 0; j < num_; ++j) {
                auto pindex = index + locations * nclasses + i * num_ + j;
                pdelta[pindex] = noObjectScale_ * (0 - poutput[pindex]);

                cost() += noObjectScale_ * std::pow(poutput[pindex], 2);
                avgAnyObj += poutput[pindex];
            }

            auto bestIndex = -1;
            auto bestIou = 0.0f;
            auto bestRmse = 20.0f;

            /*if (!isObject) {
                continue;
            }*/

            auto classIndex = index + i * nclasses;

            for (auto j = 0; j < nclasses; ++j) {
                // FIXME: pdelta[classIndex + j] = classScale;
            }
        }
    }
}

void DetectLayer::backward(const PxCpuVector& input)
{
    PX_CHECK(delta_.data() != nullptr, "Layer delta tensor is null");
    PX_CHECK(model().delta() != nullptr, "Model delta tensor is null");

    cblas_saxpy(batch() * inputs(), 1, delta_.data(), 1, model().delta(), 1);
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
