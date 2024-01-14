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

#include "Box.h"
#include "GroundTruth.h"
#include "Layer.h"
#include "Math.h"

namespace px {

struct GroundTruthContext
{
    const GroundTruth* bestGT;
    DarkBox pred;
    float bestIoU;
    int batch;
    int gridIndex;
};

struct GroundTruthResult
{
    const GroundTruth* gt;
    float bestIoU;
};

template<Device D = Device::CPU>
class DetectLayer : public Layer<D>, public Detector
{
public:
    using V = typename Layer<D>::V;

    DetectLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;
    void update() override;

    bool hasCost() const noexcept override;
    std::ostream& print(std::ostream& os) override;

    void addDetects(Detections& detections, float threshold) override;
    void addDetects(Detections& detections, int width, int height, float threshold) override;

private:
    void addDetects(Detections& detections, int width, int height, float threshold, const float* predictions) const;
    void addDetects(Detections& detections, float threshold, const float* predictions) const;

    void resetStats();
    void processDetects(int b, int i);
    DarkBox predBox(const float* poutput) const;
    GroundTruthResult groundTruth(const GroundTruthContext& ctxt);

    int coords_, num_, side_, count_ = 0;
    bool rescore_, softmax_, sqrt_, forced_, random_, reorg_;
    float coordScale_, objectScale_, noObjectScale_, classScale_, jitter_;
    float avgIoU_, avgObj_, avgAnyObj_, avgCat_, avgAllCat_;
};

template<Device D>
DetectLayer<D>::DetectLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    classScale_ = this->template property<float>("class_scale", 1.0f);
    coordScale_ = this->template property<float>("coord_scale", 1.0f);
    coords_ = this->template property<int>("coords", 1);
    forced_ = this->template property<bool>("forced", false);
    jitter_ = this->template property<float>("jitter", 0.2f);
    noObjectScale_ = this->template property<float>("noobject_scale", 1.0f);
    num_ = this->template property<int>("num", 1);
    objectScale_ = this->template property<float>("object_scale", 1.0f);
    random_ = this->template property<bool>("random", false);
    reorg_ = this->template property<bool>("reorg", false);
    rescore_ = this->template property<bool>("rescore", false);
    side_ = this->template property<int>("side", 7);
    softmax_ = this->template property<bool>("softmax", false);
    sqrt_ = this->template property<bool>("sqrt", false);

    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->inputs());

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);
}

template<Device D>
void DetectLayer<D>::resetStats()
{
    this->avgAnyObj_ = this->avgObj_ = this->avgCat_ = this->avgAllCat_ = this->avgIoU_ = 0;
    this->count_ = 0;
}

template<Device D>
bool DetectLayer<D>::hasCost() const noexcept
{
    return true;
}

template<Device D>
std::ostream& DetectLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "detection", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

template<Device D>
void DetectLayer<D>::addDetects(Detections& detections, int width, int height, float threshold)
{
    addDetects(detections, width, height, threshold, this->output_.data());
}

template<>
inline void DetectLayer<Device::CUDA>::addDetects(Detections& detections, int width, int height, float threshold)
{
    PxCpuVector output(this->output_.size());
    output.copyDevice(output_.data(), output_.size());
    addDetects(detections, width, height, threshold, output.data());
}

template<Device D>
void DetectLayer<D>::addDetects(Detections& detections, float threshold)
{
    addDetects(detections, threshold, this->output_.data());
}

template<Device D>
void DetectLayer<D>::addDetects(Detections& detections, int width, int height, float threshold,
                                const float* predictions) const
{
    auto nclasses = this->classes();

    const auto locations = this->side_ * this->side_;
    for (auto i = 0; i < locations; ++i) {
        auto row = i / this->side_;
        auto col = i % this->side_;
        for (auto n = 0; n < this->num_; ++n) {
            auto pindex = locations * nclasses + i * this->num_ + n;
            auto scale = predictions[pindex];
            auto bindex = locations * (nclasses + this->num_) + (i * this->num_ + n) * this->coords_;
            auto x = (predictions[bindex + 0] + col) / this->side_ * width;
            auto y = (predictions[bindex + 1] + row) / this->side_ * height;
            auto w = std::pow<float>(predictions[bindex + 2], this->sqrt_ ? 2.0f : 1.0f) * width;
            auto h = std::pow<float>(predictions[bindex + 3], this->sqrt_ ? 2.0f : 1.0f) * height;

            auto maxClass = 0;
            auto maxProb = std::numeric_limits<float>::lowest();
            for (auto j = 0; j < nclasses; ++j) {
                auto index = i * nclasses + j;
                auto prob = scale * predictions[index];
                if (prob > maxProb) {
                    maxClass = j;
                    maxProb = prob;
                }
            }

            if (maxProb >= threshold) {
                DarkBox b{ x, y, w, h };
                Detection det{ b.rect(), maxClass, maxProb };
                detections.emplace_back(std::move(det));
            }
        }
    }
}

template<Device D>
void DetectLayer<D>::addDetects(Detections& detections, float threshold, const float* predictions) const
{
    // TODO: implement
}

template<Device D>
void DetectLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    if (this->softmax_) {
        this->output_.copy(softmax(input));
    } else {
        this->output_.copy(input);
    }

    if (this->inferring()) {
        return;
    }

    resetStats();

    auto locations = this->side_ * this->side_;
    for (auto b = 0; b < this->batch(); ++b) {
        for (auto i = 0; i < locations; ++i) {
            processDetects(b, i);
        }
    }
}

template<>
inline void DetectLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);

    this->output_.copy(input);
}

template<Device D>
DarkBox DetectLayer<D>::predBox(const float* poutput) const
{
    DarkBox box{};

    box.x() = *poutput++ / this->side_;
    box.y() = *poutput++ / this->side_;
    box.w() = *poutput++;
    box.h() = *poutput++;

    if (this->sqrt_) {
        box.w() *= box.w();
        box.h() *= box.h();
    }

    return box;
}

template<Device D>
void DetectLayer<D>::processDetects(int b, int i)
{
    // The output tensor has dimensions [S, S, (B * 5 + C)].
    //  - S represents the size of each grid cell.
    //  - B is the number of bounding boxes predicted by each grid cell.
    //  - C is the number of classes.

    // The tensor layout is as follows:
    //   - Class probabilities for each grid cell (C x S x S).
    //   - Objectness score for each bounding box (B x S x S).
    //   - Coordinates of each bounding box ([x, y, w, h] x (B x S x S)).
    //
    // The tensor is organized in memory as a contiguous block with the following structure:
    //  ---------------------------------------------------------------------------------------------
    //  |   Class Probs (C x S x S)  |     Objectness (B x S x S)     | Coordinates (B x S x S x 4) |
    //  ---------------------------------------------------------------------------------------------
    //  |<----- C*S*S elements ----->|<------- B*S*S elements ------->|<----- B*S*S*4 elements ----->

    const auto* poutput = this->output_.data();
    const auto index = b * i;
    const auto nclasses = this->classes();
    const auto locations = this->side_ * this->side_;
    auto* pdelta = this->delta_.data();
    auto bestJ = 0;

    const GroundTruth* gt = nullptr;
    GroundTruthContext ctxt{};
    ctxt.batch = b;
    ctxt.gridIndex = i;
    ctxt.bestIoU = std::numeric_limits<float>::lowest();

    for (auto j = 0; j < this->num_; ++j) {
        auto pobject = index + locations * nclasses + i * this->num_ + j;
        pdelta[pobject] = -this->noObjectScale_ * (0 - poutput[pobject]);
        avgAnyObj_ += poutput[pobject];

        auto boxIndex = index + locations * (nclasses + this->num_) + (i * this->num_ + j) * this->coords_;
        ctxt.pred = predBox(poutput + boxIndex);
        auto result = groundTruth(ctxt);


    }
}

template<Device D>
GroundTruthResult DetectLayer<D>::groundTruth(const GroundTruthContext& ctxt)
{
    GroundTruthResult result;
    result.gt = ctxt.bestGT;
    result.bestIoU = ctxt.bestIoU;

    //  TODO: implement

    return result;
}

template<Device D>
void DetectLayer<D>::backward(const V& input)
{
}

template<Device D>
void DetectLayer<D>::update()
{

}

using CpuDetect = DetectLayer<>;
using CudaDetect = DetectLayer<Device::CUDA>;


} // px
