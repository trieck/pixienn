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

#include "Layer.h"
#include "Utility.h"
#include "event.pb.h"

namespace px {

template<Device D>
class RegionExtras
{
};

template<Device D = Device::CPU>
class RegionLayer : public Layer<D>, public Detector, public RegionExtras<D>
{
public:
    using V = typename Layer<D>::V;

    RegionLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;

    std::ostream& print(std::ostream& os) override;

    void addDetects(Detections& detections, float threshold) override;
    void addDetects(Detections& detections, int width, int height, float threshold) override;

    bool hasCost() const noexcept override
    { return true; }

private:
    void forwardCpu(const PxCpuVector& input);

    void addDetects(Detections& detections, int batch, int width, int height, float threshold,
                    const float* predictions) const;
    void addDetects(Detections& detections, int batch, float threshold, const float* predictions) const;

    DarkBox regionBox(const float* p, int n, int index, int i, int j) const;
    float deltaRegionBox(const DarkBox& truth, int n, int index, int i, int j, float scale);
    void deltaRegionClass(int index, int classId, float scale);
    float bestIoU(int b, const DarkBox& pred);

    int entryIndex(int batch, int location, int entry) const noexcept;
    void resetStats();
    void processRegion(int b, int i, int j);
    void processObjects(int b);
    void setup();
    void writeStats();
    void writeAvgIoU();
    void writeClass();
    void writeObj();
    void writeNoObj();
    void writeRecall();

    LogisticActivation<Device::CPU> logistic_;

    PxCpuVector biases_;
    std::vector<float> anchors_;
    bool biasMatch_, softmax_, rescore_;
    int coords_, num_;
    float objectScale_, noObjectScale_, classScale_, coordScale_, thresh_;

    float avgAnyObj_ = 0.0f;
    float avgCat_ = 0.0f;
    float avgIoU_ = 0.0f;
    float avgObj_ = 0.0f;
    float recall_ = 0.0f;
    int count_ = 0, classCount_ = 0;
    int logInterval_;

    PxCpuVector* poutput_, * pdelta_;
};

template<Device D>
void RegionLayer<D>::writeStats()
{
    writeAvgIoU();
    writeClass();
    writeObj();
    writeNoObj();
    writeRecall();
}

template<Device D>
void RegionLayer<D>::writeAvgIoU()
{
    auto avgIoU = count_ > 0 ? avgIoU_ / count_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().seen());

    auto tag = boost::format{ "region-%d-avg-iou" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(avgIoU);

    this->recordWriter().write(event);
}

template<Device D>
void RegionLayer<D>::writeClass()
{
    auto clazz = classCount_ > 0 ? avgCat_ / classCount_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().seen());

    auto tag = boost::format{ "region-%d-class" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(clazz);

    this->recordWriter().write(event);
}

template<Device D>
void RegionLayer<D>::writeObj()
{
    auto obj = count_ > 0 ? avgObj_ / count_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().seen());

    auto tag = boost::format{ "region-%d-object" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(obj);

    this->recordWriter().write(event);
}

template<Device D>
void RegionLayer<D>::writeNoObj()
{
    auto noObj = count_ > 0 ? avgAnyObj_ / (this->width() * this->height() * num_ * this->batch()) : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().seen());

    auto tag = boost::format{ "region-%d-noobject" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(noObj);

    this->recordWriter().write(event);
}

template<Device D>
void RegionLayer<D>::writeRecall()
{
    auto recall = count_ > 0 ? recall_ / count_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().seen());

    auto tag = boost::format{ "region-%d-recall" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(recall);

    this->recordWriter().write(event);
}

template<Device D>
RegionLayer<D>::RegionLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    anchors_ = this->template property<std::vector<float>>("anchors");
    biasMatch_ = this->template property<bool>("bias_match", false);
    classScale_ = this->template property<float>("class_scale", 1.0f);
    coordScale_ = this->template property<float>("coord_scale", 1.0f);
    coords_ = this->template property<int>("coords", 4);
    noObjectScale_ = this->template property<float>("noobject_scale", 1.0f);
    num_ = this->template property<int>("num", 1);
    objectScale_ = this->template property<float>("object_scale", 1.0f);
    rescore_ = this->template property<bool>("rescore", false);
    softmax_ = this->template property<bool>("softmax", false);
    thresh_ = this->template property<float>("thresh", 0.5f);
    logInterval_ = this->template property<int>("log_interval", 1000);

    PX_CHECK(anchors_.size() == 2 * num_, "Anchors size does not match number of regions.");

    this->setOutChannels(num_ * (coords_ + 1 + this->classes()));
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->height() * this->width() * num_ * (coords_ + 1 + this->classes()));

    biases_ = PxCpuVector(num_ * 2, 0.0f);
    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);

    for (auto i = 0; i < num_ * 2; ++i) {
        biases_[i] = anchors_[i];
    }

    setup();
}

template<Device D>
void RegionLayer<D>::setup()
{
    poutput_ = &this->output_;
    pdelta_ = &this->delta_;
}

template<Device D>
std::ostream& RegionLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "region", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

template<Device D>
int RegionLayer<D>::entryIndex(int batch, int location, int entry) const noexcept
{
    auto n = location / (this->width() * this->height());
    auto loc = location % (this->width() * this->height());

    return batch * this->outputs() + n * this->width() * this->height() * (coords_ + 1 + this->classes()) +
           entry * this->width() * this->height() + loc;
}

template<Device D>
void RegionLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    forwardCpu(input);
}

template<Device D>
void RegionLayer<D>::forwardCpu(const PxCpuVector& input)
{
    PX_CHECK(poutput_ != nullptr, "Output vector is null.");
    PX_CHECK(pdelta_ != nullptr, "Delta vector is null.");

    poutput_->copy(input);

    auto size = coords_ + 1 + this->classes();
    auto* poutput = poutput_->data();

    flatten(poutput, this->width() * this->height(), size * num_, this->batch(), 1);

    for (auto b = 0; b < this->batch(); ++b) {
        for (auto i = 0; i < this->height() * this->width() * num_; ++i) {
            auto index = i * size + b * this->outputs();
            poutput[index + 4] = logistic_.apply(poutput[index + 4]);

            if (softmax_) {
                softmax(poutput + index + coords_ + 1, this->classes(), 1, poutput + index + coords_ + 1, 1);
            }
        }
    }

    if (this->inferring()) {
        return;
    }

    resetStats();

    for (auto b = 0; b < this->batch(); ++b) {
        for (auto j = 0; j < this->height(); ++j) {
            for (auto i = 0; i < this->width(); ++i) {
                processRegion(b, i, j);
            }
        }
        processObjects(b);
    }

    flatten(pdelta_->data(), this->width() * this->height(), size * num_, this->batch(), 0);

    this->cost_ = std::pow(magArray(pdelta_->data(), pdelta_->size()), 2);

    if (count_ > 0 && this->model().seen() % logInterval_ == 0) {
        writeStats();
    }
}

template<Device D>
void RegionLayer<D>::backward(const V& input)
{
    Layer<D>::backward(input);

    auto* pDelta = this->delta_.data();
    auto* pNetDelta = this->model().delta();

    PX_CHECK(pNetDelta != nullptr, "Model delta tensor is null");
    PX_CHECK(pNetDelta->data() != nullptr, "Model delta tensor is null");
    PX_CHECK(pDelta != nullptr, "Delta tensor is null");

    const auto n = this->batch() * this->inputs();

    PX_CHECK(this->delta_.size() >= n, "Delta tensor is too small");
    PX_CHECK(pNetDelta->size() >= n, "Model tensor is too small");

    cblas_saxpy(n, 1, pDelta, 1, pNetDelta->data(), 1);
}

template<Device D>
void RegionLayer<D>::addDetects(Detections& detects, float threshold)
{
    for (auto b = 0; b < this->batch(); ++b) {
        auto* pred = this->output_.data() + b * this->outputs();
        addDetects(detects, b, threshold, pred);
    }
}

template<Device D>
void RegionLayer<D>::addDetects(Detections& detects, int width, int height, float threshold)
{
    for (auto b = 0; b < this->batch(); ++b) {
        auto* pred = this->output_.data() + b * this->outputs();
        addDetects(detects, b, width, height, threshold, pred);
    }
}

template<Device D>
void RegionLayer<D>::addDetects(Detections& detections, int batch, int width, int height, float threshold,
                                const float* predictions) const
{
    for (auto i = 0; i < this->height() * this->width(); ++i) {
        auto row = i / this->width();
        auto col = i % this->width();
        for (auto n = 0; n < num_; ++n) {
            auto index = i * num_ + n;
            auto pindex = index * (coords_ + 1 + this->classes()) + coords_;
            auto scale = predictions[pindex];
            auto bindex = index * (coords_ + 1 + this->classes());
            auto box = regionBox(predictions, n, bindex, col, row);

            auto clsIndex = index * (coords_ + 1 + this->classes()) + coords_ + 1;

            for (auto j = 0; j < this->classes(); ++j) {
                auto prob = scale * predictions[clsIndex + j];
                if (prob >= threshold) {
                    auto rect = lightBox(box, { width, height });
                    Detection det(rect, batch, j, prob);
                    detections.emplace_back(std::move(det));
                }
            }
        }
    }
}

template<Device D>
DarkBox RegionLayer<D>::regionBox(const float* p, int n, int index, int i, int j) const
{
    auto* biases = biases_.data();

    auto x = (i + logistic_.apply(p[index + 0])) / this->width();
    auto y = (j + logistic_.apply(p[index + 1])) / this->height();

    auto w = std::exp(p[index + 2]) * biases[2 * n] / this->width();
    auto h = std::exp(p[index + 3]) * biases[2 * n + 1] / this->height();

    return { x, y, w, h };
}

template<Device D>
float RegionLayer<D>::deltaRegionBox(const DarkBox& truth, int n, int index, int i, int j, float scale)
{
    auto* x = poutput_->data();

    auto pred = regionBox(x, n, index, i, j);
    auto iou = pred.iou(truth);

    auto* biases = biases_.data();
    auto* delta = pdelta_->data();

    auto tx = truth.x() * this->width() - i;
    auto ty = truth.y() * this->height() - j;
    auto tw = std::log(truth.w() * this->width() / biases[2 * n]);
    auto th = std::log(truth.h() * this->height() / biases[2 * n + 1]);

    delta[index + 0] = scale * (tx - logistic_.apply(x[index + 0]))
                       * logistic_.gradient(logistic_.apply(x[index + 0]));

    delta[index + 1] = scale * (ty - logistic_.apply(x[index + 1]))
                       * logistic_.gradient(logistic_.apply(x[index + 1]));

    delta[index + 2] = scale * (tw - x[index + 2]);
    delta[index + 3] = scale * (th - x[index + 3]);

    return iou;
}

template<Device D>
void RegionLayer<D>::deltaRegionClass(int index, int classId, float scale)
{
    auto* output = poutput_->data();
    auto* delta = pdelta_->data();

    for (auto n = 0; n < this->classes(); ++n) {
        float netTruth = (n == classId) ? 1.0f : 0.0f;
        delta[index + n] = scale * (netTruth - output[index + n]);
        if (n == classId) {
            avgCat_ += output[index + n];
        }
    }
}

template<Device D>
float RegionLayer<D>::bestIoU(int b, const DarkBox& pred)
{
    const auto& gt = this->groundTruth(b);

    auto bestIoU = std::numeric_limits<float>::lowest();

    for (const auto& g: gt) {
        auto iou = pred.iou(g.box);
        if (iou > bestIoU) {
            bestIoU = iou;
        }
    }

    return bestIoU;
}

template<Device D>
void RegionLayer<D>::addDetects(Detections& detections, int batch, float threshold, const float* predictions) const
{
    addDetects(detections, batch, 1, 1, threshold, predictions);
}

template<Device D>
void RegionLayer<D>::resetStats()
{
    avgAnyObj_ = avgCat_ = avgObj_ = avgIoU_ = recall_ = 0;
    count_ = classCount_ = 0;
}

template<Device D>
void RegionLayer<D>::processRegion(int b, int i, int j)
{
    const auto size = coords_ + this->classes() + 1;

    auto* poutput = poutput_->data();
    auto* pdelta = pdelta_->data();
    auto* pbias = biases_.data();

    const auto scale = 0.01f;

    for (auto n = 0; n < num_; ++n) {
        auto index = size * (j * this->width() * num_ + i * num_ + n) + b * this->outputs();
        avgAnyObj_ += poutput[index + 4];

        pdelta[index + 4] = this->noObjectScale_ * ((0 - poutput[index + 4]) * logistic_.gradient(poutput[index + 4]));

        auto pred = regionBox(poutput, n, index, i, j);
        auto iou = bestIoU(b, pred);
        if (iou > thresh_) {
            pdelta[index + 4] = 0;
        }

        if (this->seen() < 12800) {   // magic!
            auto x = (i + .5f) / this->width();
            auto y = (j + .5f) / this->height();
            auto w = pbias[2 * n] / this->width();
            auto h = pbias[2 * n + 1] / this->height();

            deltaRegionBox({ x, y, w, h }, n, index, i, j, scale);
        }
    }
}

template<Device D>
void RegionLayer<D>::processObjects(int b)
{
    auto* poutput = poutput_->data();
    auto* pdelta = pdelta_->data();

    auto size = coords_ + this->classes() + 1;

    for (const auto& gt: this->groundTruth(b)) {
        auto bestIoU = std::numeric_limits<float>::lowest();
        auto bestIndex = 0;
        auto bestN = 0;

        auto i = static_cast<int>(gt.box.x() * this->width());
        auto j = static_cast<int>(gt.box.y() * this->height());

        auto truthShift(gt.box);
        truthShift.x() = 0;
        truthShift.y() = 0;

        for (auto n = 0; n < num_; ++n) {
            auto index = size * (j * this->width() * num_ + i * num_ + n) + b * this->outputs();
            auto pred = regionBox(poutput, n, index, i, j);
            if (biasMatch_) {
                pred.w() = biases_[2 * n] / this->width();
                pred.h() = biases_[2 * n + 1] / this->height();
            }

            pred.x() = pred.y() = 0;
            auto iou = pred.iou(truthShift);
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

////////////////////////////////////////////////////////////////////////////////

using CpuRegion = RegionLayer<>;
using CudaRegion = RegionLayer<Device::CUDA>;

} // px

#ifdef USE_CUDA

#include "cuda/RegionLayer.h"

#endif
