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
#include "event.pb.h"

using namespace tensorflow;

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

template<Device D>
class DetectExtras
{
};

template<Device D = Device::CPU>
class DetectLayer : public Layer<D>, public Detector, public DetectExtras<D>
{
public:
    using V = typename Layer<D>::V;

    DetectLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input, V* grad) override;

    bool hasCost() const noexcept override;
    std::ostream& print(std::ostream& os) override;

    void addDetects(Detections& detections, float threshold) override;
    void addDetects(Detections& detections, int width, int height, float threshold) override;

private:
    void addDetects(Detections& detections, int batch, int width, int height, float threshold,
                    const float* predictions) const;
    void addDetects(Detections& detections, int batch, float threshold, const float* predictions) const;

    void forwardCpu(const PxCpuVector& input);
    void resetStats();
    void processDetects(int b, int i);
    DarkBox predBox(const float* poutput) const;
    GroundTruthResult groundTruth(const GroundTruthContext& ctxt);
    void setup();
    void writeStats();
    void writeAvgIoU();
    void writePosCat();
    void writeAllCat();
    void writePosObj();
    void writeAnyObj();

    PxCpuVector* poutput_, * pdelta_;

    int coords_, num_, side_, count_ = 0, logInterval_;
    bool rescore_, softmax_, sqrt_;
    float coordScale_, objectScale_, noObjectScale_, classScale_;
    float avgIoU_, avgObj_, avgAnyObj_, avgCat_, avgAllCat_;
};

template<Device D>
DetectLayer<D>::DetectLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    classScale_ = this->template property<float>("class_scale", 1.0f);
    coordScale_ = this->template property<float>("coord_scale", 1.0f);
    coords_ = this->template property<int>("coords", 1);
    noObjectScale_ = this->template property<float>("noobject_scale", 1.0f);
    num_ = this->template property<int>("num", 1);
    objectScale_ = this->template property<float>("object_scale", 1.0f);
    rescore_ = this->template property<bool>("rescore", false);
    side_ = this->template property<int>("side", 7);
    softmax_ = this->template property<bool>("softmax", false);
    sqrt_ = this->template property<bool>("sqrt", false);
    logInterval_ = this->template property<int>("log_interval", 1000);

    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->inputs());

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);

    setup();
}

template<Device D>
void DetectLayer<D>::setup()
{
    poutput_ = &this->output_;
    pdelta_ = &this->delta_;
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
    for (auto b = 0; b < this->batch(); ++b) {
        auto* pred = this->output_.data() + b * this->outputs();
        addDetects(detections, b, width, height, threshold, pred);
    }
}

template<Device D>
void DetectLayer<D>::addDetects(Detections& detections, float threshold)
{
    for (auto b = 0; b < this->batch(); ++b) {
        auto* pred = this->output_.data() + b * this->outputs();
        addDetects(detections, b, threshold, pred);
    }
}

template<Device D>
void DetectLayer<D>::addDetects(Detections& detections, int batch, int width, int height, float threshold,
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
                DarkBox box{ x, y, w, h };
                Detection det{ box.rect(), batch, maxClass, maxProb };
                detections.emplace_back(std::move(det));
            }
        }
    }
}

template<Device D>
void DetectLayer<D>::addDetects(Detections& detections, int batch, float threshold, const float* predictions) const
{
    addDetects(detections, batch, 1, 1, threshold, predictions);
}

template<Device D>
void DetectLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    forwardCpu(input);
}

template<Device D>
void DetectLayer<D>::forwardCpu(const PxCpuVector& input)
{
    PX_CHECK(poutput_ != nullptr, "Output vector is null.");
    PX_CHECK(pdelta_ != nullptr, "Delta vector is null.");

    if (this->softmax_) {
        this->poutput_->copy(softmax(input));
    } else {
        this->poutput_->copy(input);
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

    this->cost_ = std::pow(magArray(this->pdelta_->data(), this->pdelta_->size()), 2);

    if (this->training() && count_ > 0 && this->model().seen() % logInterval_ == 0) {
        writeStats();
    }
}

template<Device D>
void DetectLayer<D>::writeStats()
{
    writeAvgIoU();
    writePosCat();
    writeAllCat();
    writePosObj();
    writeAnyObj();
}

template<Device D>
void DetectLayer<D>::writeAvgIoU()
{
    auto avgIoU = count_ > 0 ? avgIoU_ / count_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().seen());

    auto tag = boost::format{ "detect-%d-avg-iou" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(avgIoU);

    this->recordWriter().write(event);
}

template<Device D>
void DetectLayer<D>::writePosCat()
{
    auto posCat = count_ > 0 ? avgCat_ / count_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().seen());

    auto tag = boost::format{ "detect-%d-pos-cat" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(posCat);

    this->recordWriter().write(event);
}

template<Device D>
void DetectLayer<D>::writeAllCat()
{
    auto allCat = count_ > 0 ? avgAllCat_ / (count_ * this->classes()) : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().seen());

    auto tag = boost::format{ "detect-%d-all-cat" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(allCat);

    this->recordWriter().write(event);
}

template<Device D>
void DetectLayer<D>::writePosObj()
{
    auto posObj = count_ > 0 ? avgObj_ / count_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().seen());

    auto tag = boost::format{ "detect-%d-pos-obj" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(posObj);

    this->recordWriter().write(event);
}

template<Device D>
void DetectLayer<D>::writeAnyObj()
{
    auto locations = this->side_ * this->side_;

    auto anyObj = count_ > 0 ? avgAnyObj_ / (this->batch() * locations * num_) : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().seen());

    auto tag = boost::format{ "detect-%d-any-obj" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(anyObj);

    this->recordWriter().write(event);
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

    const auto* poutput = this->poutput_->data();
    const auto index = b * i;
    const auto nclasses = this->classes();
    const auto locations = this->side_ * this->side_;
    auto* pdelta = this->pdelta_->data();
    auto bestJ = 0;

    const GroundTruth* gt = nullptr;
    GroundTruthContext ctxt{};
    ctxt.batch = b;
    ctxt.gridIndex = i;
    ctxt.bestIoU = std::numeric_limits<float>::lowest();

    for (auto j = 0; j < this->num_; ++j) {
        auto pobject = index + locations * nclasses + i * this->num_ + j;
        pdelta[pobject] = this->noObjectScale_ * (0 - poutput[pobject]);
        avgAnyObj_ += poutput[pobject];

        auto boxIndex = index + locations * (nclasses + this->num_) + (i * this->num_ + j) * this->coords_;
        ctxt.pred = predBox(poutput + boxIndex);
        auto result = groundTruth(ctxt);
        if (result.bestIoU > ctxt.bestIoU) {
            ctxt.bestIoU = result.bestIoU;
            gt = result.gt;
            bestJ = j;
        }
    }

    if (gt == nullptr) {
        return; // no ground truth for this grid cell
    }

    // Compute the class loss
    auto classIndex = index + i * nclasses;
    for (auto j = 0; j < nclasses; ++j) {
        float netTruth = gt->classId == j ? 1.0f : 0.0f;
        pdelta[classIndex + j] = classScale_ * (netTruth - poutput[classIndex + j]);
        if (netTruth) {
            avgCat_ += poutput[classIndex + j];
        }
        avgAllCat_ += poutput[classIndex + j];
    }

    DarkBox truthBox(gt->box);
    truthBox.x() /= side_;
    truthBox.y() /= side_;

    auto row = i / side_;
    auto col = i % side_;
    auto truthRow = (int) (gt->box.y() * side_);
    auto truthCol = (int) (gt->box.x() * side_);

    PX_CHECK(row == truthRow, "The ground truth box is not in the grid cell row.");
    PX_CHECK(col == truthCol, "The ground truth box is not in the grid cell column.");

    auto pobject = index + locations * nclasses + i * num_ + bestJ;
    auto boxIndex = index + locations * (nclasses + num_) + (i * num_ + bestJ) * coords_;

    auto pred = predBox(poutput + boxIndex);
    auto iou = pred.iou(truthBox);

    avgObj_ += poutput[pobject];
    pdelta[pobject] = objectScale_ * (1. - poutput[pobject]);

    if (rescore_) {
        pdelta[pobject] = objectScale_ * (iou - poutput[pobject]);
    }

    pdelta[boxIndex + 0] = coordScale_ * (gt->box.x() - poutput[boxIndex + 0]);
    pdelta[boxIndex + 1] = coordScale_ * (gt->box.y() - poutput[boxIndex + 1]);
    pdelta[boxIndex + 2] = coordScale_ * (gt->box.w() - poutput[boxIndex + 2]);
    pdelta[boxIndex + 3] = coordScale_ * (gt->box.h() - poutput[boxIndex + 3]);

    if (sqrt_) {
        pdelta[boxIndex + 2] = coordScale_ * (std::sqrt(gt->box.w()) - poutput[boxIndex + 2]);
        pdelta[boxIndex + 3] = coordScale_ * (std::sqrt(gt->box.h()) - poutput[boxIndex + 3]);
    }

    avgIoU_ += iou;
    count_++;
}

template<Device D>
GroundTruthResult DetectLayer<D>::groundTruth(const GroundTruthContext& ctxt)
{
    GroundTruthResult result;
    result.gt = ctxt.bestGT;
    result.bestIoU = ctxt.bestIoU;

    auto row = ctxt.gridIndex / side_;
    auto col = ctxt.gridIndex % side_;

    const auto& gts = Layer<D>::groundTruth(ctxt.batch);
    for (const auto& gt: gts) {
        auto truthRow = static_cast<int>(gt.box.y() * side_);
        auto truthCol = static_cast<int>(gt.box.x() * side_);
        if (!(truthRow == row && truthCol == col)) {
            continue;   // should we do this?
        }

        DarkBox truthBox(gt.box);
        truthBox.x() /= side_;
        truthBox.y() /= side_;

        auto iou = ctxt.pred.iou(truthBox);
        if (iou > result.bestIoU) {
            result.bestIoU = iou;
            result.gt = &gt;
        }
    }

    return result;
}

template<Device D>
void DetectLayer<D>::backward(const V& input, V* grad)
{
    Layer<D>::backward(input, grad);

    if (grad == nullptr) {
        return;
    }

    auto* pDelta = this->delta_.data();

    const auto n = this->batch() * this->inputs();

    cblas_saxpy(n, 1, pDelta, 1, grad->data(), 1);
}

using CpuDetect = DetectLayer<>;
using CudaDetect = DetectLayer<Device::CUDA>;

} // px

#ifdef USE_CUDA

#include "cuda/DetectLayer.h"

#endif

