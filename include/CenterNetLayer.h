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

#include "Activation.h"
#include "CenterNetTargetBuilder.h"
#include "Detection.h"
#include "Layer.h"
#include "event.pb.h"

using namespace tensorflow;

namespace px {

float focalLoss(const PxCpuVector& pred, const PxCpuVector& target, int numClasses, int H, int W, float alpha,
                float beta, int numObjects);

float sizeLoss(const float* pred, const GroundTruthVec& gt, int H, int W, float stride);

template<Device D>
class CenterNetExtras
{
};

template<Device D = Device::CPU>
class CenterNetLayer : public Layer<D>, public Detector, public CenterNetExtras<D>
{
public:
    using V = typename Layer<D>::V;

    CenterNetLayer(Model<D>& model, const YAML::Node& layerDef);

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
    void setup();

    void forwardCpu(const PxCpuVector& input);
    void addDetects(Detections& detections, int batch, int width, int height, float threshold,
                    const float* predictions) const;
    void addDetects(Detections& detections, int batch, float threshold, const float* predictions) const;

    void computeHeatmapGradient(const PxCpuVector& targetHeatmap);

    CenterNetTargets targets_;
    LogisticActivation <Device::CPU> logistic_;
    PxCpuVector* poutput_, * pdelta_;

    int stride_;
    int logInterval_;
    float alpha_, beta_;
    bool init_ = false;
};

template<Device D>
CenterNetLayer<D>::CenterNetLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    stride_ = this->template property<int>("stride", 4);
    logInterval_ = this->template property<int>("log_interval", 1000);
    alpha_ = this->template property<float>("alpha", 2.0f);
    beta_ = this->template property<float>("beta", 4.0f);

    auto nclasses = this->classes();

    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());

    // nclasses for heatmap, 2 for size (w/h), 2 for offset (dx/dy)
    this->setOutputs(this->outHeight() * this->outWidth() * (nclasses + 2 + 2));

    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);

    // Setup the layer
    setup();
}

template<Device D>
void CenterNetLayer<D>::setup()
{
    poutput_ = &this->output_;
    pdelta_ = &this->delta_;
}

template<Device D>
void CenterNetLayer<D>::forward(const CenterNetLayer::V& input)
{
    Layer<D>::forward(input);

    forwardCpu(input);
}

template<Device D>
void CenterNetLayer<D>::forwardCpu(const PxCpuVector& input)
{
    this->poutput_->copy(input);

    if (this->inferring()) {
        return;
    }

    if (!init_) {
        CenterNetTargetBuilder builder(this->classes(), stride_, this->model().width(), this->model().height());
        targets_ = builder.buildTargets(this->groundTruth(0));  // assuming batch=1 for now
        init_ = true;
    }

    auto fmapArea = this->height() * this->width();
    auto* outPtr = this->poutput_->data();
    auto* start = outPtr + 0 * this->outputs();
    auto* end = start + this->classes() * fmapArea;

    logistic_.apply(start, end);

    const auto& heatmap = *this->poutput_;
    const auto* size = outPtr + this->classes() * fmapArea;
    const auto* offset = size + 2 * fmapArea;

    const auto* predSize = size;
    const auto* predOffset = offset;

    auto* deltaHeatmap = this->pdelta_->data();
    auto* deltaSize = deltaHeatmap + this->classes() * fmapArea;
    auto* deltaOffset = deltaSize + 2 * fmapArea;

    computeHeatmapGradient(targets_.heatmap);

    const auto* targetSize = targets_.size.data();
    const auto* targetOffset = targets_.offset.data();
    const auto* mask = targets_.mask.data();

    auto count = 0;
    for (int i = 0; i < fmapArea; ++i) {
        if (mask[i] > 0.0f) {
            ++count;
        }
    }

    auto scale = (count > 0) ? 1.0f / count : 0.0f;

    for (auto c = 0; c < 2; ++c) {
        auto base = c * fmapArea;

        for (auto i = 0; i < fmapArea; ++i) {
            if (mask[i] > 0.0f) {
                deltaSize[base + i] = 0.1 * scale * (predSize[base + i] - targetSize[base + i]);
                deltaOffset[base + i] = scale * (predOffset[base + i] - targetOffset[base + i]);
            } else {
                deltaSize[base + i] = 0.0f;
                deltaOffset[base + i] = 0.0f;
            }
        }
    }

    auto numObjects = this->groundTruth(0).size();  // assuming batch=1 for now

    auto hmLoss = focalLoss(heatmap, targets_.heatmap, this->classes(), this->height(), this->width(), alpha_, beta_,
                            numObjects);

    auto sizeLoss = sizeLoss(size, this->groundTruth(0), this->height(), this->width(), stride_);

    auto offsetLoss = l1MaskedLoss(offset, targets_.offset.data(), targets_.mask.data(), 2, this->height(),
                                   this->width(), numObjects);

    std::cout << "CenterNetLayer: hmLoss=" << hmLoss
              << ", sizeLoss=" << sizeLoss
              << ", offsetLoss=" << offsetLoss
              << ", numObjects=" << numObjects
              << std::endl;

    this->cost_ = 1.0 * hmLoss + 0.1 * sizeLoss + offsetLoss;
}

template<Device D>
void CenterNetLayer<D>::backward(const V& input, V* grad)
{
    Layer<D>::backward(input, grad);

    if (!grad) {
        return;  // no upstream gradient to accumulate into
    }

    // Compute: grad += delta_
    const auto n = this->batch() * this->inputs();
    cblas_saxpy(n, 1, this->delta_.data(), 1, grad->data(), 1);
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, float threshold)
{
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, int width, int height, float threshold)
{
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, int batch, int width, int height, float threshold,
                                   const float* predictions) const
{
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, int batch, float threshold, const float* predictions) const
{
}

template<Device D>
std::ostream& CenterNetLayer<D>::print(std::ostream& os)
{
    return os;
}

template<Device D>
void CenterNetLayer<D>::computeHeatmapGradient(const PxCpuVector& targetHeatmap)
{
    const auto spatialSize = this->height() * this->width();
    const auto numClasses = this->classes();

    const auto* pred = this->poutput_->data();
    const auto* target = targetHeatmap.data();

    auto* delta = this->pdelta_->data();

    for (auto c = 0; c < numClasses; ++c) {
        auto base = c * spatialSize;

        for (auto i = 0; i < spatialSize; ++i) {
            auto p = std::clamp(pred[base + i], 1e-4f, 1.0f - 1e-4f);
            auto y = target[base + i];

            auto grad = 0.0f;
            if (y > 0.99f) {
                // Positive sample — pull p → 1
                grad = alpha_ * pow(1 - p, alpha_ - 1) * (p - 1);
            } else if (y < 0.01f) {
                // Negative sample — push p → 0
                grad = -pow(p, alpha_ + 1);  // or use (p - y) form
            } else {
                grad = 0;
            }

            delta[base + i] = grad;
        }
    }
}

inline float focalLoss(const PxCpuVector& pred, const PxCpuVector& target, int numClasses, int H, int W,
                       float alpha, float beta, int numObjects)
{
    auto loss = 0.0f;
    auto spatialSize = H * W;

    for (auto c = 0; c < numClasses; ++c) {
        auto base = c * spatialSize;
        for (auto i = 0; i < spatialSize; ++i) {
            auto yHat = std::clamp(pred[base + i], 1e-4f, 1.0f - 1e-4f);
            auto y = target[base + i];

            if (y > 0.99f) {
                loss += std::pow(1.0f - yHat, alpha) * std::log(yHat);
            } else if (y < 0.01f) {
                loss += std::pow(1.0f - y, beta) * std::pow(yHat, alpha) * std::log(1.0f - yHat);
            }
        }
    }

    // Normalize by the number of objects
    if (numObjects > 0) {
        loss = -loss / numObjects;
    } else {
        loss = 0.0f;
    }

    return loss;
}

inline float sizeLoss(const float* pred, const GroundTruthVec& gt, int H, int W, float stride)
{
    auto loss = 0.0f;
    auto numObjects = gt.size();

    auto scaleX = W * stride;
    auto scaleY = H * stride;

    for (const auto& g: gt) {
        auto x1 = g.box.x() * scaleX;
        auto y1 = g.box.y() * scaleY;
        auto x2 = (g.box.x() + g.box.w()) * scaleX;
        auto y2 = (g.box.y() + g.box.h()) * scaleY;

        auto cx = (x1 + x2) / 2.0f;
        auto cy = (y1 + y2) / 2.0f;

        auto gx = static_cast<int>(cx / stride);
        auto gy = static_cast<int>(cy / stride);

        if (gx < 0 || gx >= W || gy < 0 || gy >= H) {
            continue;  // skip invalid center
        }

        auto idx = gy * W + gx;

        auto trueW = x2 - x1;
        auto trueH = y2 - y1;

        auto predW = pred[0 * H * W + idx]; // channel 0 width
        auto predH = pred[1 * H * W + idx]; // channel 1 height

        loss += std::abs(predW - trueW) + std::abs(predH - trueH);
    }

    return numObjects > 0 ? loss / numObjects : 0.0f;
}

using CpuCenterNet = CenterNetLayer<>;
using CudaCenterNet = CenterNetLayer<Device::CUDA>;

} // px

#ifdef USE_CUDA

#include "cuda/CenterNetLayer.h"

#endif  // USE_CUDA
