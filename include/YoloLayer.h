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
    void backward(const V& input) override;

    bool hasCost() const noexcept override
    {
        return true;
    }

    std::ostream& print(std::ostream& os) override;

    void addDetects(Detections& detections, float threshold) override;
    void addDetects(Detections& detections, int width, int height, float threshold) override;

private:
    void forwardCpu(const PxCpuVector& input);

    void addDetects(Detections& detections, int width, int height, float threshold, const float* predictions) const;
    void addDetects(Detections& detections, float threshold, const float* predictions) const;

    int entryIndex(int batch, int location, int entry) const noexcept;
    DarkBox yoloBox(const float* p, int mask, int index, int i, int j) const;
    cv::Rect scaledYoloBox(const float* p, int mask, int index, int i, int j, int w, int h) const;
    void setup();
    void resetStats();
    void processRegion(int b, int i, int j);
    void processObjects(int b);
    int maskIndex(int n);
    float deltaYoloBox(const GroundTruth& truth, int mask, int index, int i, int j);
    void deltaYoloClass(int index, int classId);

    PxCpuVector biases_;
    std::vector<int> mask_, anchors_;
    int num_, n_;
    float ignoreThresh_, truthThresh_;

    LogisticActivation<Device::CPU> logistic_;
    PxCpuVector* poutput_, * pdelta_;

    float avgIoU = 0.0f;
    float recall_ = 0.0f;
    float recall75_ = 0.0f;
    float avgCat_ = 0.0f;
    float avgObj_ = 0.0f;
    float avgAnyObj_ = 0.0f;
    int count_ = 0;
    int classCount_ = 0;
};

template<Device D>
YoloLayer<D>::YoloLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    anchors_ = this->template property<std::vector<int>>("anchors");
    mask_ = this->template property<std::vector<int>>("mask");
    n_ = mask_.size();
    num_ = this->template property<int>("num", 1);
    ignoreThresh_ = this->template property<float>("ignore_thresh", 0.5f);
    truthThresh_ = this->template property<float>("truth_thresh", 1.0f);

    PX_CHECK(anchors_.size() == 2 * num_, "anchors must be twice num size");

    auto nclasses = this->classes();
    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->outHeight() * this->outWidth() * n_ * (nclasses + 4 + 1));

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);

    biases_ = PxCpuVector(num_ * 2);
    for (auto i = 0; i < num_ * 2; ++i) {
        biases_[i] = static_cast<float>(anchors_[i]);
    }

    setup();
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

    auto area = std::max(1, this->height() * this->width());
    auto nclasses = this->classes();

    auto* poutput = this->poutput_->data();
    for (auto b = 0; b < this->batch(); ++b) {
        for (auto n = 0; n < n_; ++n) {
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

    resetStats();

    for (auto b = 0; b < this->batch(); ++b) {
        for (auto j = 0; j < this->height(); ++j) {
            for (auto i = 0; i < this->width(); ++i) {
                processRegion(b, i, j);
            }
        }
        processObjects(b);
    }

    this->cost_ = std::pow(magArray(pdelta_->data(), pdelta_->size()), 2);

    if (count_ > 0) {
        printf("Yolo %d: Avg. IoU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n",
               this->index(),
               avgIoU / count_,
               avgCat_ / classCount_,
               avgObj_ / count_,
               avgAnyObj_ / (this->batch() * this->width() * this->height() * this->n_),
               recall_ / count_,
               recall75_ / count_,
               count_);
    }
}

template<Device D>
void YoloLayer<D>::deltaYoloClass(int index, int classId)
{
    auto* poutput = poutput_->data();
    auto* pdelta = pdelta_->data();

    auto stride = this->width() * this->height();

    if (pdelta[index]) {
        pdelta[index + classId * stride] = 1 - poutput[index + classId * stride];
        avgCat_ += poutput[index + classId * stride];
        return;
    }

    for (auto i = 0; i < this->classes(); ++i) {
        auto netTruth = (i == classId) ? 1.0f : 0.0f;
        pdelta[index + i * stride] = netTruth - poutput[index + i * stride];

        if (netTruth) {
            avgCat_ += poutput[index + i * stride];
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

    auto tx = truth.box.x() * this->width() - i;
    auto ty = truth.box.y() * this->height() - j;
    auto tw = std::log(truth.box.w() * w / biases_[2 * mask]);
    auto th = std::log(truth.box.h() * h / biases_[2 * mask + 1]);

    auto scale = 2 - truth.box.w() * truth.box.h();
    auto stride = this->width() * this->height();

    delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);

    return iou;
}

template<Device D>
int YoloLayer<D>::maskIndex(int n)
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
        auto bestIoU = -std::numeric_limits<float>::max();
        auto bestN = 0;

        auto i = static_cast<int>(gt.box.x() * this->width());
        auto j = static_cast<int>(gt.box.y() * this->height());
        i = std::max(0, std::min(i, this->width() - 1));
        j = std::max(0, std::min(j, this->height() - 1));

        auto truthShift(gt.box);
        truthShift.x() = 0;
        truthShift.y() = 0;

        for (auto n = 0; n < num_; ++n) {
            DarkBox pred;
            pred.w() = biases_[2 * n] / this->model().width();
            pred.h() = biases_[2 * n + 1] / this->model().height();

            auto iou = pred.iou(truthShift);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestN = n;
            }
        }

        auto maskN = maskIndex(bestN);
        if (maskN >= 0) {
            auto location = maskN * this->width() * this->height() + j * this->width() + i;
            auto boxIndex = entryIndex(b, location, 0);
            auto iou = deltaYoloBox(gt, bestN, boxIndex, i, j);

            auto objIndex = entryIndex(b, location, 4);
            avgObj_ += poutput[objIndex];
            pdelta[objIndex] = 1 - poutput[objIndex];

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

            avgIoU += iou;
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

    for (auto n = 0; n < n_; ++n) {
        auto entry = n * this->width() * this->height() + j * this->width() + i;

        auto boxIndex = entryIndex(b, entry, 0);
        ctxt.pred = yoloBox(poutput, mask_[n], boxIndex, i, j);

        auto result = bestGT(ctxt);
        const auto* gt = result.gt;

        auto objIndex = entryIndex(b, entry, 4);
        avgAnyObj_ += poutput[objIndex];

        pdelta[objIndex] = 0 - poutput[objIndex];
        if (result.bestIoU > ignoreThresh_) {
            pdelta[objIndex] = 0;
        }

        if (gt == nullptr) {
            continue;   // no ground truth
        }

        if (result.bestIoU > truthThresh_) {
            pdelta[objIndex] = 1 - poutput[objIndex];

            auto clsIndex = entryIndex(b, entry, 4 + 1);
            deltaYoloClass(clsIndex, gt->classId);
            deltaYoloBox(*gt, mask_[n], boxIndex, i, j);
        }
    }
}

template<Device D>
void YoloLayer<D>::resetStats()
{
    avgIoU = 0.0f;
    recall_ = 0.0f;
    recall75_ = 0.0f;
    avgCat_ = 0.0f;
    avgObj_ = 0.0f;
    avgAnyObj_ = 0.0f;
    count_ = 0;
    classCount_ = 0;
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

    auto width = std::exp(p[index + 2 * stride]) * biases_[2 * mask] / netW;
    width *= (float) netW / newW;

    auto height = std::exp(p[index + 3 * stride]) * biases_[2 * mask + 1] / netH;
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
    auto width = std::exp(p[index + 2 * stride]) * biases_[2 * mask] / w;
    auto height = std::exp(p[index + 3 * stride]) * biases_[2 * mask + 1] / h;

    return { x, y, width, height };
}

template<Device D>
void YoloLayer<D>::addDetects(Detections& detections, float threshold, const float* predictions) const
{
    // TODO: Implement
}

template<Device D>
void YoloLayer<D>::addDetects(Detections& detections, int width, int height, float threshold, const float* predictions)
const
{
    auto area = std::max(1, this->width() * this->height());
    auto nclasses = this->classes();

    for (auto i = 0; i < area; ++i) {
        auto row = i / this->width();
        auto col = i % this->width();

        for (auto n = 0; n < n_; ++n) {
            auto objIndex = entryIndex(0, n * area + i, 4);
            auto objectness = predictions[objIndex];
            if (objectness < threshold) {
                continue;
            }

            auto boxIndex = entryIndex(0, n * area + i, 0);
            auto box = scaledYoloBox(predictions, mask_[n], boxIndex, col, row, width, height);

            int maxClass = 0;
            float maxProb = -std::numeric_limits<float>::max();

            for (auto j = 0; j < nclasses; ++j) {
                int clsIndex = entryIndex(0, n * area + i, 5 + j);
                auto prob = objectness * predictions[clsIndex];
                if (prob > maxProb) {
                    maxClass = j;
                    maxProb = prob;
                }
            }

            if (maxProb >= threshold) {
                Detection det(box, maxClass, maxProb);
                detections.emplace_back(std::move(det));
            }
        }
    }
}

template<Device D>
void YoloLayer<D>::addDetects(Detections& detections, int width, int height, float threshold)
{
    addDetects(detections, width, height, threshold, this->output_.data());
}

template<Device D>
void YoloLayer<D>::addDetects(Detections& detections, float threshold)
{
    addDetects(detections, threshold, this->output_.data());
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
void YoloLayer<D>::backward(const V& input)
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

