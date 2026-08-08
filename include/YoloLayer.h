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

#include "event.pb.h"
#include "Layer.h"

using namespace tensorflow;

namespace px {

struct YoloGTCtxt
{
    const GroundTruthVec* gt;
    DarkBox pred;
};

struct YoloGTResult
{
    const GroundTruth* gt;
    float bestIoU;
};

static YoloGTResult bestGT(const YoloGTCtxt& ctxt);

template<Device D>
class YoloExtras
{
};

template<Device D = Device::CPU>
class YoloLayer : public Layer<D>, public Detector, public YoloExtras<D>
{
public:
    using V = typename Layer<D>::V;

    YoloLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input, V* grad) override;

    bool hasCost() const noexcept override
    {
        return true;
    }

    std::ostream& print(std::ostream& os) override;

    void addDetects(Detections& detections, float threshold) override;
    void addDetects(Detections& detections, int width, int height, float threshold) override;

private:
    void forwardCpu(const PxCpuVector& input);

    void addDetects(Detections& detections, int batch, int width, int height, float threshold,
                    const float* predictions) const;
    void addDetects(Detections& detections, int batch, float threshold, const float* predictions) const;

    int entryIndex(int batch, int location, int entry) const noexcept;
    DarkBox yoloBox(const float* p, int mask, int index, int i, int j) const;
    cv::Rect scaledYoloBox(const float* p, int mask, int index, int i, int j, int w, int h) const;
    void setup();
    void resetStats();
    void processRegion(int b, int i, int j);
    void processObjects(int b);
    int maskIndex(int n) const;
    float deltaYoloBox(const GroundTruth& truth, int mask, int index, int i, int j);
    void deltaYoloClass(int index, int classId);
    void writeStats();
    void writeAvgIoU();
    void writeAvgClass();
    void writeObjectness();
    void writeRecall50();
    void writeRecall75();
    void writeCost();
    void writeComponentCost(const char* tag, float value);
    void updateLossComponents();
    std::size_t positiveTargetCount() const noexcept;
    float noObjectScaleFor(std::size_t positiveTargets, std::size_t slots) const noexcept;
    float positiveObjectnessScale() const noexcept;
    float classNegativeScale() const noexcept;

    std::vector<int> mask_, anchors_;
    int numAnchors_, numMasks_;
    float ignoreThresh_, truthThresh_;
    float coordScale_, objectScale_, noObjectScale_, objectNormalizer_, classScale_;
    bool normalizeNoObject_ = false;
    bool normalizeClass_ = false;
    float effectiveNoObjectScale_ = 1.0f;

    LogisticActivation<Device::CPU> logistic_;
    PxCpuVector* poutput_, * pdelta_;

    float avgIoU_ = 0.0f;
    float recall_ = 0.0f;
    float recall75_ = 0.0f;
    float avgCat_ = 0.0f;
    float avgObj_ = 0.0f;
    float avgAnyObj_ = 0.0f;
    float boxCost_ = 0.0f;
    float objectCost_ = 0.0f;
    float noObjectCost_ = 0.0f;
    float classCost_ = 0.0f;
    int count_ = 0;
    int classCount_ = 0;
    int logInterval_ = 0;
};

template<Device D>
YoloLayer<D>::YoloLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    anchors_ = this->template property<std::vector<int>>("anchors");
    mask_ = this->template property<std::vector<int>>("mask");
    numMasks_ = mask_.size();
    numAnchors_ = this->template property<int>("num", 1);
    ignoreThresh_ = this->template property<float>("ignore_thresh", 0.5f);
    truthThresh_ = this->template property<float>("truth_thresh", 1.0f);
    coordScale_ = this->template property<float>("coord_scale", 1.0f);
    objectScale_ = this->template property<float>("object_scale", 1.0f);
    noObjectScale_ = this->template property<float>("noobject_scale", 1.0f);
    objectNormalizer_ = this->template property<float>("obj_normalizer", 1.0f);
    normalizeNoObject_ = this->template property<bool>("normalize_noobject", false);
    effectiveNoObjectScale_ = noObjectScale_;
    normalizeClass_ = this->template property<bool>("normalize_class", false);
    classScale_ = this->template property<float>("class_scale", 1.0f);
    logInterval_ = this->template property<int>("log_interval", 1000);

    auto nclasses = this->classes();

    PX_CHECK(anchors_.size() == 2 * numAnchors_, "anchors must be twice num size");
    PX_CHECK(this->channels() == numMasks_ * (nclasses + 4 + 1),
             "YoloLayer: output channel count must match mask-based anchor count");

    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->outHeight() * this->outWidth() * numMasks_ * (nclasses + 4 + 1));

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);

    setup();
}

template<Device D>
float YoloLayer<D>::positiveObjectnessScale() const noexcept
{
    // Keep the legacy object_scale knob useful while retaining Darknet's
    // obj_normalizer as an optional overall multiplier.
    return objectScale_ * objectNormalizer_;
}

template<Device D>
float YoloLayer<D>::classNegativeScale() const noexcept
{
    if (!normalizeClass_ || this->classes() <= 1) {
        return 1.0f;
    }

    // Each assigned object has one positive class and classes - 1 negative
    // classes. Equalizing their total weight prevents the negative class
    // terms from driving all sigmoid outputs toward zero during scratch
    // training.
    return 1.0f / static_cast<float>(this->classes() - 1);
}

template<Device D>
std::size_t YoloLayer<D>::positiveTargetCount() const noexcept
{
    auto count = std::size_t{0};
    for (auto b = 0; b < this->batch(); ++b) {
        for (const auto& gt: this->groundTruth(b)) {
            auto bestIoU = std::numeric_limits<float>::lowest();
            auto bestAnchor = 0;

            DarkBox truthShift(gt.box);
            truthShift.x() = 0;
            truthShift.y() = 0;

            for (auto anchor = 0; anchor < numAnchors_; ++anchor) {
                DarkBox candidate;
                candidate.w() = static_cast<float>(anchors_[2 * anchor]) / this->model().width();
                candidate.h() = static_cast<float>(anchors_[2 * anchor + 1]) / this->model().height();

                const auto iou = candidate.iou(truthShift);
                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestAnchor = anchor;
                }
            }

            if (maskIndex(bestAnchor) >= 0) {
                ++count;
            }
        }
    }
    return count;
}

template<Device D>
float YoloLayer<D>::noObjectScaleFor(std::size_t positiveTargets, std::size_t slots) const noexcept
{
    if (!normalizeNoObject_) {
        return noObjectScale_;
    }

    const auto positives = std::max<std::size_t>(1, positiveTargets);
    const auto negatives = std::max<std::size_t>(1, slots > positives ? slots - positives : 1);
    return noObjectScale_ * static_cast<float>(positives) / static_cast<float>(negatives);
}

template<Device D>
void YoloLayer<D>::setup()
{
    poutput_ = &this->output_;
    pdelta_ = &this->delta_;
}

template<Device D>
void YoloLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    forwardCpu(input);
}

template<Device D>
void YoloLayer<D>::forwardCpu(const PxCpuVector& input)
{
    PX_CHECK(poutput_ != nullptr, "Output vector is null.");
    PX_CHECK(pdelta_ != nullptr, "Delta vector is null.");

    this->poutput_->copy(input);
    // Class and box supervision is sparse. Never allow assignments from a
    // previous batch to survive at locations untouched by the current batch.
    this->pdelta_->fill(0.0f);

    auto area = std::max(1, this->height() * this->width());
    auto nclasses = this->classes();

    auto* poutput = this->poutput_->data();
    for (auto b = 0; b < this->batch(); ++b) {
        for (auto n = 0; n < numMasks_; ++n) {
            auto index = entryIndex(b, n * area, 0);
            auto* start = poutput + index;
            auto* end = start + 2 * area;

            logistic_.apply(start, end);
            index = entryIndex(b, n * area, 4);
            start = poutput + index;
            end = start + (1 + nclasses) * area;

            logistic_.apply(start, end);
        }
    }

    if (this->inferring()) {
        return;
    }

    const auto training = this->training();
    const auto savedAvgIoU = avgIoU_;
    const auto savedRecall = recall_;
    const auto savedRecall75 = recall75_;
    const auto savedAvgCat = avgCat_;
    const auto savedAvgObj = avgObj_;
    const auto savedAvgAnyObj = avgAnyObj_;
    const auto savedCount = count_;
    const auto savedClassCount = classCount_;

    if (!training) {
        resetStats();
    }

    const auto slots = static_cast<std::size_t>(this->batch()) * numMasks_ * area;
    effectiveNoObjectScale_ = noObjectScaleFor(positiveTargetCount(), slots);

    for (auto b = 0; b < this->batch(); ++b) {
        for (auto j = 0; j < this->height(); ++j) {
            for (auto i = 0; i < this->width(); ++i) {
                processRegion(b, i, j);
            }
        }
        processObjects(b);
    }

    this->cost_ = std::pow(magArray(pdelta_->data(), pdelta_->size()), 2) / std::max(1, this->batch());
    updateLossComponents();

    if (training && count_ > 0 && this->model().updateCount() % logInterval_ == 0) {
        writeStats();
        resetStats();
    } else if (!training) {
        // Validation needs the loss deltas above, but its teacher-forced head
        // statistics must not erase or contaminate the current training window.
        avgIoU_ = savedAvgIoU;
        recall_ = savedRecall;
        recall75_ = savedRecall75;
        avgCat_ = savedAvgCat;
        avgObj_ = savedAvgObj;
        avgAnyObj_ = savedAvgAnyObj;
        count_ = savedCount;
        classCount_ = savedClassCount;
    }
}

template<Device D>
void YoloLayer<D>::writeStats()
{
    writeAvgIoU();
    writeAvgClass();
    writeObjectness();
    writeRecall50();
    writeRecall75();
    writeCost();
    writeComponentCost((boost::format{ "yolo-%d-box-loss" } % this->index()).str().c_str(), boxCost_);
    writeComponentCost((boost::format{ "yolo-%d-object-loss" } % this->index()).str().c_str(), objectCost_);
    writeComponentCost((boost::format{ "yolo-%d-no-object-loss" } % this->index()).str().c_str(), noObjectCost_);
    writeComponentCost((boost::format{ "yolo-%d-class-loss" } % this->index()).str().c_str(), classCost_);
}

template<Device D>
void YoloLayer<D>::writeAvgIoU()
{
    auto avgIoU = count_ > 0 ? avgIoU_ / count_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().updateCount());

    auto tag = boost::format{ "yolo-%d-avg-iou" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(avgIoU);

    this->recordWriter().write(event);
}

template<Device D>
void YoloLayer<D>::writeAvgClass()
{
    auto avgClass = count_ > 0 ? avgCat_ / count_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().updateCount());

    auto tag = boost::format{ "yolo-%d-avg-class" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(avgClass);

    this->recordWriter().write(event);
}

template<Device D>
void YoloLayer<D>::writeObjectness()
{
    auto objectness = count_ > 0 ? avgObj_ / count_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().updateCount());

    auto tag = boost::format{ "yolo-%d-objectness" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(objectness);

    this->recordWriter().write(event);
}

template<Device D>
void YoloLayer<D>::writeRecall50()
{
    auto recall50 = recall_ > 0 ? recall_ / classCount_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().updateCount());

    auto tag = boost::format{ "yolo-%d-recall50" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(recall50);

    this->recordWriter().write(event);
}

template<Device D>
void YoloLayer<D>::writeRecall75()
{
    auto recall75 = recall75_ > 0 ? recall75_ / classCount_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().updateCount());

    auto tag = boost::format{ "yolo-%d-recall75" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(recall75);

    this->recordWriter().write(event);
}

template<Device D>
void YoloLayer<D>::writeCost()
{
    auto cost = this->cost_ > 0 ? this->cost_ : 0.0f;

    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().updateCount());

    auto tag = boost::format{ "yolo-%d-cost" } % this->index();

    auto* summary = event.mutable_summary();
    auto* value = summary->add_value();
    value->set_tag(tag.str());
    value->set_simple_value(cost);

    this->recordWriter().write(event);
}

template<Device D>
void YoloLayer<D>::writeComponentCost(const char* tag, float value)
{
    Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(this->model().updateCount());
    auto* summary = event.mutable_summary();
    auto* metric = summary->add_value();
    metric->set_tag(tag);
    metric->set_simple_value(value);
    this->recordWriter().write(event);
}

template<Device D>
void YoloLayer<D>::deltaYoloClass(int index, int classId)
{
    auto* poutput = poutput_->data();
    auto* pdelta = pdelta_->data();

    auto stride = this->width() * this->height();

    for (auto i = 0; i < this->classes(); ++i) {
        auto netTruth = (i == classId) ? 1.0f : 0.0f;
        const auto scale = netTruth == 1.0f ? classScale_ : classScale_ * classNegativeScale();
        pdelta[index + i * stride] = scale * (netTruth - poutput[index + i * stride]);

        if (netTruth) {
            avgCat_ += std::min(1.0f, poutput[index + i * stride]);
        }
    }
}

template<Device D>
float YoloLayer<D>::deltaYoloBox(const GroundTruth& truth, int mask, int index, int i, int j)
{
    auto* delta = pdelta_->data();
    auto* x = poutput_->data();

    auto w = this->model().width();
    auto h = this->model().height();

    auto pred = yoloBox(x, mask, index, i, j);
    auto iou = pred.iou(truth.box);

    constexpr float eps = 1e-9f;

    auto tx = truth.box.x() * this->width() - i;
    auto ty = truth.box.y() * this->height() - j;
    auto tw = std::log(std::max(eps, truth.box.w() * w / anchors_[2 * mask]));
    auto th = std::log(std::max(eps, truth.box.h() * h / anchors_[2 * mask + 1]));

    auto scale = coordScale_ * (2 - truth.box.w() * truth.box.h());
    auto stride = this->width() * this->height();

    delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);

    return iou;
}

template<Device D>
int YoloLayer<D>::maskIndex(int n) const
{
    auto it = std::find(mask_.begin(), mask_.end(), n);

    if (it != mask_.end()) {
        return std::distance(mask_.begin(), it);
    }

    return -1;
}

template<Device D>
void YoloLayer<D>::processObjects(int b)
{
    auto* poutput = poutput_->data();
    auto* pdelta = pdelta_->data();
    for (const auto& gt: this->groundTruth(b)) {
        auto bestIoU = std::numeric_limits<float>::lowest();
        auto bestN = 0;

        auto i = static_cast<int>(gt.box.x() * this->width());
        auto j = static_cast<int>(gt.box.y() * this->height());
        i = std::max(0, std::min(i, this->width() - 1));
        j = std::max(0, std::min(j, this->height() - 1));

        auto truthShift(gt.box);
        truthShift.x() = 0;
        truthShift.y() = 0;

        for (auto n = 0; n < numAnchors_; ++n) {
            DarkBox pred;
            pred.w() = static_cast<float>(anchors_[2 * n]) / this->model().width();
            pred.h() = static_cast<float>(anchors_[2 * n + 1]) / this->model().height();

            auto iou = pred.iou(truthShift);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestN = n;
            }
        }

        auto maskN = maskIndex(bestN);
        if (maskN >= 0) {
            auto location = maskN * this->width() * this->height() + j * this->width() + i;

            // Darknet writes the best-anchor target directly; a later truth
            // at the same cell/anchor overwrites the earlier target.
            auto boxIndex = entryIndex(b, location, 0);
            auto iou = deltaYoloBox(gt, bestN, boxIndex, i, j);

            auto objIndex = entryIndex(b, location, 4);
            avgObj_ += poutput[objIndex];
            pdelta[objIndex] = positiveObjectnessScale() * (1 - poutput[objIndex]);

            auto clsIndex = entryIndex(b, location, 4 + 1);
            deltaYoloClass(clsIndex, gt.classId);

            ++count_;
            ++classCount_;

            if (iou > .5) {
                ++recall_;
            }
            if (iou > .75) {
                ++recall75_;
            }

            avgIoU_ += iou;
        }
    }
}

template<Device D>
void YoloLayer<D>::processRegion(int b, int i, int j)
{
    auto* poutput = poutput_->data();
    auto* pdelta = pdelta_->data();

    YoloGTCtxt ctxt;
    ctxt.gt = &this->groundTruth(b);

    for (auto n = 0; n < numMasks_; ++n) {
        auto entry = n * this->width() * this->height() + j * this->width() + i;

        auto boxIndex = entryIndex(b, entry, 0);
        ctxt.pred = yoloBox(poutput, mask_[n], boxIndex, i, j);

        auto result = bestGT(ctxt);
        const auto* gt = result.gt;

        auto objIndex = entryIndex(b, entry, 4);
        avgAnyObj_ += poutput[objIndex];

        if (gt == nullptr || result.bestIoU < ignoreThresh_) {
            pdelta[objIndex] = objectNormalizer_ * effectiveNoObjectScale_ * (0 - poutput[objIndex]);
        }
        // A truth-threshold match must override the no-object penalty.  The
        // thresholds are commonly configured with truth_thresh < ignore_thresh;
        // using else-if here incorrectly labels that entire IoU interval as
        // background.
        if (gt != nullptr && result.bestIoU > truthThresh_) {
            pdelta[objIndex] = positiveObjectnessScale() * (1 - poutput[objIndex]);

            auto clsIndex = entryIndex(b, entry, 4 + 1);
            deltaYoloClass(clsIndex, gt->classId);
            deltaYoloBox(*gt, mask_[n], boxIndex, i, j);
        }
    }
}

template<Device D>
void YoloLayer<D>::resetStats()
{
    avgIoU_ = 0.0f;
    recall_ = 0.0f;
    recall75_ = 0.0f;
    avgCat_ = 0.0f;
    avgObj_ = 0.0f;
    avgAnyObj_ = 0.0f;
    count_ = 0;
    classCount_ = 0;
}

template<Device D>
void YoloLayer<D>::updateLossComponents()
{
    boxCost_ = 0.0f;
    objectCost_ = 0.0f;
    noObjectCost_ = 0.0f;
    classCost_ = 0.0f;

    const auto area = this->width() * this->height();
    const auto stride = std::max(1, area);
    const auto* delta = pdelta_->data();
    for (auto b = 0; b < this->batch(); ++b) {
        for (auto n = 0; n < numMasks_; ++n) {
            for (auto cell = 0; cell < area; ++cell) {
                const auto location = n * area + cell;
                const auto boxIndex = entryIndex(b, location, 0);
                for (auto component = 0; component < 4; ++component) {
                    const auto value = delta[boxIndex + component * stride];
                    boxCost_ += value * value;
                }

                const auto objectIndex = entryIndex(b, location, 4);
                const auto objectValue = delta[objectIndex];
                if (objectValue >= 0.0f) {
                    objectCost_ += objectValue * objectValue;
                } else {
                    noObjectCost_ += objectValue * objectValue;
                }

                const auto classIndex = entryIndex(b, location, 5);
                for (auto c = 0; c < this->classes(); ++c) {
                    const auto value = delta[classIndex + c * stride];
                    classCost_ += value * value;
                }
            }
        }
    }
}

template<Device D>
cv::Rect YoloLayer<D>::scaledYoloBox(const float* p, int mask, int index, int i, int j, int w, int h) const
{
    const auto stride = this->width() * this->height();
    const auto netW = this->model().width();
    const auto netH = this->model().height();

    int newW, newH;
    if (((float) netW / w) < ((float) netH / h)) {
        newW = netW;
        newH = (h * netW) / w;
    } else {
        newH = netH;
        newW = (w * netH) / h;
    }

    auto x = (i + p[index + 0 * stride]) / this->width();
    x = (x - (netW - newW) / 2.0f / netW) / ((float) newW / netW);

    auto y = (j + p[index + 1 * stride]) / this->height();
    y = (y - (netH - newH) / 2.0f / netH) / ((float) newH / netH);

    auto width = std::exp(p[index + 2 * stride]) * anchors_[2 * mask] / netW;
    width *= (float) netW / newW;

    auto height = std::exp(p[index + 3 * stride]) * anchors_[2 * mask + 1] / netH;
    height *= (float) netH / newH;

    auto left = std::max<int>(0, (x - width / 2) * w);
    auto right = std::min<int>(w - 1, (x + width / 2) * w);
    auto top = std::max<int>(0, (y - height / 2) * h);
    auto bottom = std::min<int>(h - 1, (y + height / 2) * h);

    return { left, top, right - left, bottom - top };
}

template<Device D>
DarkBox YoloLayer<D>::yoloBox(const float* p, int mask, int index, int i, int j) const
{
    auto stride = this->width() * this->height();

    auto w = this->model().width();
    auto h = this->model().height();

    auto x = (i + p[index + 0 * stride]) / this->width();
    auto y = (j + p[index + 1 * stride]) / this->height();
    auto width = std::exp(p[index + 2 * stride]) * anchors_[2 * mask] / w;
    auto height = std::exp(p[index + 3 * stride]) * anchors_[2 * mask + 1] / h;

    return { x, y, width, height };
}

template<Device D>
void YoloLayer<D>::addDetects(Detections& detections, int batch, float threshold, const float* predictions) const
{
    addDetects(detections, 0, 0, batch, threshold, predictions);
}

template<Device D>
void YoloLayer<D>::addDetects(Detections& detections, int batch, int width, int height, float threshold,
                              const float* predictions)
const
{
    const auto scaled = width > 0 && height > 0;

    auto area = std::max(1, this->width() * this->height());
    auto nclasses = this->classes();

    for (auto i = 0; i < area; ++i) {
        auto row = i / this->width();
        auto col = i % this->width();

        for (auto n = 0; n < numMasks_; ++n) {
            auto objIndex = entryIndex(batch, n * area + i, 4);
            auto objectness = predictions[objIndex];
            if (objectness < threshold) {
                continue;
            }

            auto boxIndex = entryIndex(batch, n * area + i, 0);

            cv::Rect2f box;
            if (scaled) {
                box = scaledYoloBox(predictions, mask_[n], boxIndex, col, row, width, height);
            } else {
                box = yoloBox(predictions, mask_[n], boxIndex, col, row).rect();
            }

            for (auto j = 0; j < nclasses; ++j) {
                int clsIndex = entryIndex(batch, n * area + i, 5 + j);
                auto prob = objectness * predictions[clsIndex];
                if (prob >= threshold) {
                    detections.emplace_back(box, batch, j, prob);
                }
            }
        }
    }
}

template<Device D>
void YoloLayer<D>::addDetects(Detections& detections, int width, int height, float threshold)
{
    for (auto b = 0; b < this->batch(); ++b) {
        auto predictions = this->output_.data() + b * this->outputs();
        addDetects(detections, b, width, height, threshold, predictions);
    }
}

template<Device D>
void YoloLayer<D>::addDetects(Detections& detections, float threshold)
{
    for (auto b = 0; b < this->batch(); ++b) {
        auto predictions = this->output_.data() + b * this->outputs();
        addDetects(detections, b, threshold, predictions);
    }
}

template<Device D>
std::ostream& YoloLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "yolo", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });
    return os;
}

template<Device D>
int YoloLayer<D>::entryIndex(int batch, int location, int entry) const noexcept
{
    auto area = std::max(1, this->width() * this->height());

    auto n = location / area;
    auto loc = location % area;

    return batch * this->outputs() + n * area * (4 + this->classes() + 1) + entry * area + loc;
}

template<Device D>
void YoloLayer<D>::backward(const V& input, V* grad)
{
    Layer<D>::backward(input, grad);

    if (grad != nullptr) {
        const auto n = this->batch() * this->inputs();
        cblas_saxpy(n, 1, this->delta_.data(), 1, grad->data(), 1);
    }
}

YoloGTResult bestGT(const YoloGTCtxt& ctxt)
{
    YoloGTResult result;
    result.gt = nullptr;
    result.bestIoU = std::numeric_limits<float>::lowest();

    for (const auto& gt: *ctxt.gt) {
        auto iou = ctxt.pred.iou(gt.box);
        if (iou > result.bestIoU) {
            result.bestIoU = iou;
            result.gt = &gt;
        }
    }

    return result;
}

using CpuYolo = YoloLayer<>;
using CudaYolo = YoloLayer<Device::CUDA>;

} // px

#ifdef USE_CUDA

#include "cuda/YoloLayer.h"

#endif // USE_CUDA
