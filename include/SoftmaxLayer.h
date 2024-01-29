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

#include "Detection.h"
#include "Layer.h"
#include "Math.h"

namespace px {

template<Device D>
class SoftmaxLayer : public Layer<D>, public Detector
{
public:
    using V = typename Layer<D>::V;

    SoftmaxLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;

    virtual bool hasCost() const noexcept override;

    std::ostream& print(std::ostream& os) override;

    void addDetects(Detections& detections, float threshold) override;
    void addDetects(Detections& detections, int width, int height, float threshold) override;

private:
    void addDetects(Detections& detections, int width, int height, float threshold, const float* predictions) const;

    void computeLoss();
    V loss_;
    float temperature_;
    int groups_;
    bool detector_;
};

template<Device D>
SoftmaxLayer<D>::SoftmaxLayer(Model<D>& model, const Node& layerDef) : Layer<D>(model, layerDef)
{
    detector_ = this->template property<bool>("detector", true);
    groups_ = std::max(1, this->template property<int>("groups", 1));
    temperature_ = this->template property<float>("temperature", 1.0f);

    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->inputs());

    auto outputSize = this->batch() * this->outputs();

    this->output_ = V(outputSize, 0.0f);
    this->delta_ = V(outputSize, 0.0f);
    loss_ = V(outputSize, 0.0f);
};

template<Device D>
void SoftmaxLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    softmax(input.data(),
            this->inputs() / groups_,
            this->batch(),
            this->inputs(),
            groups_,
            this->inputs() / groups_,
            1,
            this->temperature_,
            this->output_.data());

    if (this->training()) {
        computeLoss();
        this->cost_ = sumArray(this->loss_.data(), this->batch() * this->outputs());
    }
}

template<Device D>
void SoftmaxLayer<D>::computeLoss()
{
    for (auto b = 0; b < this->batch(); ++b) {
        auto* pout = this->output_.data() + b * this->outputs();
        auto* ploss = this->loss_.data() + b * this->outputs();
        auto* pdelta = this->delta_.data() + b * this->outputs();

        for (const auto& gt: this->groundTruth(b)) {
            for (auto i = 0; i < this->outputs(); ++i) {
                auto t = gt.classId == i ? 1.0f : 0.0f;
                auto p = pout[i];
                ploss[i] = -t * std::log(p + 1e-9f);
                pdelta[i] = t - p;
            }
        }
    }
}

template<Device D>
void SoftmaxLayer<D>::backward(const V& input)
{
    Layer<D>::backward(input);

    auto* pDelta = this->delta_.data();
    auto* pNetDelta = this->netDelta()->data();
    const auto n = this->batch() * this->inputs();

    cblas_saxpy(n, 1, pDelta, 1, pNetDelta, 1);
}

template<Device D>
std::ostream& SoftmaxLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "softmax", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

template<Device D>
bool SoftmaxLayer<D>::hasCost() const noexcept
{
    return true;
}

template<Device D>
void SoftmaxLayer<D>::addDetects(Detections& detections, int width, int height, float threshold)
{
    if (!detector_) {
        return;
    }

    addDetects(detections, width, height, threshold, this->output_.data());
}

template<Device D>
void SoftmaxLayer<D>::addDetects(Detections& detections, int width, int height, float threshold,
                                 const float* predictions) const
{
    auto maxClass = std::max_element(predictions, predictions + this->classes());
    auto maxClassProb = *maxClass;
    auto maxClassId = std::distance(predictions, maxClass);

    if (maxClassProb >= threshold) {
        cv::Rect box = { 0, 0, width, height };

        auto detection = Detection(std::move(box), maxClassId, maxClassProb);
        detections.emplace_back(std::move(detection));
    }
}

template<Device D>
void SoftmaxLayer<D>::addDetects(Detections& detections, float threshold)
{
    if (!detector_) {
        return;
    }

    addDetects(detections, this->model().width(), this->model().height(), threshold, this->output_.data());
}

}   // px