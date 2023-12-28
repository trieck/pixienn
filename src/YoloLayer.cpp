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

#include "Box.h"
#include "Model.h"
#include "YoloLayer.h"

namespace px {

struct GroundTruthContext
{
    const GroundTruthVec* gt;
    cv::Rect2f pred;
};

struct GroundTruthResult
{
    const GroundTruth* gt;
    float bestIoU;
};

static GroundTruthResult bestGT(const GroundTruthContext& ctxt);

YoloLayer::YoloLayer(Model& model, const YAML::Node& layerDef) : Layer(model, layerDef)
{
}

void YoloLayer::setup()
{
    anchors_ = property<std::vector<int>>("anchors");
    mask_ = property<std::vector<int>>("mask");
    n_ = mask_.size();
    num_ = property<int>("num", 1);
    ignoreThresh_ = property<float>("ignore_thresh", 0.5f);
    truthThresh_ = property<float>("truth_thresh", 1.0f);

    PX_CHECK(anchors_.size() == num_ * 2, "Anchors size must be twice num size.");

    auto nclasses = classes();
    setOutChannels(channels());
    setOutHeight(height());
    setOutWidth(width());
    setOutputs(outHeight() * outWidth() * n_ * (nclasses + 4 + 1));

#ifdef USE_CUDA
    if (useGpu()) {
        outputGpu_ = PxCudaVector(batch() * outChannels() * outHeight() * outWidth());
    } else {
        output_ = PxCpuVector(batch() * outChannels() * outHeight() * outWidth());
    }
#else
    output_ = PxCpuVector(batch() * outputs(), 0.0f);
    delta_ = PxCpuVector(batch() * outputs(), 0.0f);
    biases_ = PxCpuTensor<1>({ (size_t) num_ * 2 }, 0.5f);

#endif
    for (auto i = 0; i < num_ * 2; ++i) {
        biases_[i] = anchors_[i];
    }
}

std::ostream& YoloLayer::print(std::ostream& os)
{
    Layer::print(os, "yolo", { height(), width(), channels() }, { outHeight(), outWidth(), outChannels() });

    return os;
}

void YoloLayer::forward(const PxCpuVector& input)
{
    Layer::forward(input);

    output_.copy(input);

    auto area = std::max(1, width() * height());
    auto nclasses = classes();

    auto* poutput = output_.data();
    for (auto b = 0; b < batch(); ++b) {
        for (auto n = 0; n < n_; ++n) {
            auto index = entryIndex(b, n * area, 0);
            auto* start = poutput + index;
            auto* end = start + 2 * area + 1;
            logistic_.apply(start, end);
            index = entryIndex(b, n * area, 4);
            start = poutput + index;
            end = start + (1 + nclasses) * area + 1;
            logistic_.apply(start, end);
        }
    }

    if (inferring()) {
        return;
    }

    resetStats();

    for (auto b = 0; b < batch(); ++b) {
        for (auto j = 0; j < height(); ++j) {
            for (auto i = 0; i < width(); ++i) {
                processRegion(b, i, j);
            }
        }
        processObjects(b);
    }

    cost_ = std::pow(magArray(delta_.data(), delta_.size()), 2);

    if (count_ == 0) {
        printf("Region %d Avg. IoU: -----, Class: -----, Obj: -----, No Obj: %f, .5R: -----, .75R: -----,  count: 0\n",
               index(),
               avgAnyObj_ / (batch() * width() * height() * n_));
    } else {
        printf("Region %d Avg. IoU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n",
               index(),
               avgIoU / count_,
               avgCat_ / classCount_,
               avgObj_ / count_,
               avgAnyObj_ / (batch() * width() * height() * n_),
               recall_ / count_,
               recall75_ / count_,
               count_);
    }
}

void YoloLayer::processRegion(int b, int i, int j)
{
    auto* poutput = output_.data();
    auto* pdelta = delta_.data();

    GroundTruthContext ctxt;
    ctxt.gt = &groundTruth(b);

    for (auto n = 0; n < n_; ++n) {
        auto entry = n * width() * height() + j * width() + i;

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

void YoloLayer::processObjects(int b)
{
    auto* poutput = output_.data();
    auto* pdelta = delta_.data();

    for (const auto& gt: groundTruth(b)) {
        auto bestIoU = -std::numeric_limits<float>::max();
        auto bestIndex = 0;
        auto bestN = 0;

        auto i = static_cast<int>(gt.box.x * width());
        auto j = static_cast<int>(gt.box.y * height());

        auto truthShift(gt.box);
        truthShift.x = 0;
        truthShift.y = 0;

        for (auto n = 0; n < num_; ++n) {
            cv::Rect2f pred;
            pred.width = biases_[2 * n] / model().width();
            pred.height = biases_[2 * n + 1] / model().height();
            auto iou = boxIoU(pred, truthShift);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestN = n;
            }
        }

        auto maskN = maskIndex(bestN);
        if (maskN >= 0) {
            auto location = maskN * width() * height() + j * width() + i;
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

int YoloLayer::maskIndex(int n)
{
    auto it = std::find(mask_.begin(), mask_.end(), n);

    if (it != mask_.end()) {
        return std::distance(mask_.begin(), it);
    }

    return -1;
}

float YoloLayer::deltaYoloBox(const GroundTruth& truth, int mask, int index, int i, int j)
{
    auto* delta = delta_.data();
    auto* x = output_.data();

    auto w = model().width();
    auto h = model().height();

    auto pred = yoloBox(x, mask, index, i, j);
    auto iou = boxIoU(pred, truth.box);

    auto tx = truth.box.x * width() - i;
    auto ty = truth.box.y * height() - j;
    auto tw = std::log(truth.box.width * w / biases_[2 * mask]);
    auto th = std::log(truth.box.height * h / biases_[2 * mask + 1]);

    auto scale = 2 - truth.box.width * truth.box.height;
    auto stride = width() * height();

    delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);

    return iou;
}

void YoloLayer::deltaYoloClass(int index, int classId)
{
    auto* pdelta = delta_.data();
    auto* poutput = output_.data();

    auto stride = width() * height();

    if (pdelta[index]) {
        pdelta[index + classId * stride] = 1 - poutput[index + classId * stride];
        avgCat_ += poutput[index + classId * stride];
        return;
    }

    for (auto i = 0; i < classes(); ++i) {
        auto netTruth = (i == classId) ? 1 : 0;
        pdelta[index + i * stride] = netTruth - poutput[index + i * stride];

        if (netTruth) {
            avgCat_ += poutput[index + i * stride];
        }
    }
}

GroundTruthResult bestGT(const GroundTruthContext& ctxt)
{
    GroundTruthResult result;
    result.gt = nullptr;
    result.bestIoU = -std::numeric_limits<float>::max();

    for (const auto& gt: *ctxt.gt) {
        auto iou = boxIoU(ctxt.pred, gt.box);
        if (iou > result.bestIoU) {
            result.bestIoU = iou;
            result.gt = &gt;
        }
    }

    return result;
}

void YoloLayer::backward(const PxCpuVector& input)
{
    Layer::backward(input);

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
void YoloLayer::forwardGpu(const PxCudaVector& input)
{
    outputGpu_.copy(input);

    auto area = std::max(1, width() * height());
    auto nclasses = classes();

    auto* poutput = outputGpu_.data();
    for (auto b = 0; b < batch(); ++b) {
        for (auto n = 0; n < n_; ++n) {
            auto index = entryIndex(b, n * area, 0);
            auto* start = poutput + index;
            activation_->applyGpu(start, 2 * area);
            index = entryIndex(b, n * area, 4);
            start = poutput + index;
            activation_->applyGpu(start, (1 + nclasses) * area + 1);
        }
    }
}
#endif // USE_CUDA

int YoloLayer::entryIndex(int batch, int location, int entry) const noexcept
{
    auto area = std::max(1, width() * height());

    auto n = location / area;
    int loc = location % area;

    return batch * outputs() + n * area * (4 + classes() + 1) + entry * area + loc;
}

cv::Rect2f YoloLayer::yoloBox(const float* p, int mask, int index, int i, int j) const
{
    auto stride = width() * height();

    auto w = model().width();
    auto h = model().height();

    auto x = (i + p[index + 0 * stride]) / width();
    auto y = (j + p[index + 1 * stride]) / height();
    auto width = std::exp(p[index + 2 * stride]) * biases_[2 * mask] / w;
    auto height = std::exp(p[index + 3 * stride]) * biases_[2 * mask + 1] / h;

    return { x, y, width, height };
}

cv::Rect YoloLayer::scaledYoloBox(const float* p, int mask, int index, int i, int j, int w, int h) const
{
    const auto stride = width() * height();
    const auto netW = model().width();
    const auto netH = model().height();

    int newW, newH;
    if (((float) netW / w) < ((float) netH / h)) {
        newW = netW;
        newH = (h * netW) / w;
    } else {
        newH = netH;
        newW = (w * netH) / h;
    }

    auto x = (i + p[index + 0 * stride]) / width();
    x = (x - (netW - newW) / 2.0f / netW) / ((float) newW / netW);

    auto y = (j + p[index + 1 * stride]) / height();
    y = (y - (netH - newH) / 2.0f / netH) / ((float) newH / netH);

    auto width = std::exp(p[index + 2 * stride]) * biases_[2 * mask] / netW;
    width *= (float) netW / newW;

    auto height = std::exp(p[index + 3 * stride]) * biases_[2 * mask + 1] / netH;
    height *= (float) netH / newH;

    auto left = std::max<int>(0, (x - width / 2) * w);
    auto right = std::min<int>(w - 1, (x + width / 2) * w);
    auto top = std::max<int>(0, (y - height / 2) * h);
    auto bottom = std::min<int>(h - 1, (y + height / 2) * h);

    cv::Rect b;
    b.x = left;
    b.y = top;
    b.width = right - left;
    b.height = bottom - top;

    return b;
}

void YoloLayer::addDetects(Detections& detections, int width, int height, float threshold)
{
    addDetects(detections, width, height, threshold, output_.data());
}

void YoloLayer::addDetects(Detections& detections, float threshold)
{
    addDetects(detections, threshold, output_.data());
}

void YoloLayer::addDetects(Detections& detections, int width, int height, float threshold,
                           const float* predictions) const
{
    auto area = std::max(1, this->width() * this->height());
    auto nclasses = classes();

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

void YoloLayer::addDetects(Detections& detections, float threshold, const float* predictions) const
{
    // TODO:
}

void YoloLayer::resetStats()
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

#ifdef USE_CUDA

void YoloLayer::addDetectsGpu(Detections& detections, int width, int height, float threshold)
{
    auto predv = outputGpu_.asVector();
    addDetects(detections, width, height, threshold, predv.data());
}

#endif // USE_CUDA

} // px
