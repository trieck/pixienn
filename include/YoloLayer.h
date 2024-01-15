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

template<Device D = Device::CPU>
class YoloLayer : public Layer<D>, public Detector
{
public:
    using V = typename Layer<D>::V;

    YoloLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;
    void update() override;

    std::ostream& print(std::ostream& os) override;

    void addDetects(Detections& detections, float threshold) override;
    void addDetects(Detections& detections, int width, int height, float threshold) override;

private:
    void addDetects(Detections& detections, int width, int height, float threshold, const float* predictions) const;
    void addDetects(Detections& detections, float threshold, const float* predictions) const;

    int entryIndex(int batch, int location, int entry) const noexcept;
    DarkBox yoloBox(const float* p, int mask, int index, int i, int j) const;
    cv::Rect scaledYoloBox(const float* p, int mask, int index, int i, int j, int w, int h) const;

    PxCpuVector biases_;
    std::vector<int> mask_, anchors_;
    int num_, n_;
    float ignoreThresh_, truthThresh_;

    LogisticActivation <D> logistic_;
};

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
void YoloLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    this->output_.copy(input);

    auto area = std::max(1, this->height() * this->width());
    auto nclasses = this->classes();

    auto* poutput = this->output_.data();
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

    // TODO: Implement training
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
    std::cout << "YoloLayer::backward" << std::endl;
}

template<Device D>
void YoloLayer<D>::update()
{
}

using CpuYolo = YoloLayer<>;
using CudaYolo = YoloLayer<Device::CUDA>;

} // px

#ifdef USE_CUDA

#include "cuda/YoloLayer.h"

#endif // USE_CUDA

