#pragma once

#include "Activation.h"
#include "CenterNetTargetBuilder.h"
#include "Detection.h"
#include "Layer.h"

namespace px {

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
    bool hasCost() const noexcept override { return true; }
    std::ostream& print(std::ostream& os) override;

    void addDetects(Detections& detections, float threshold) override;
    void addDetects(Detections& detections, int width, int height, float threshold) override;

private:
    void setup();
    void forwardCpu(const PxCpuVector& input);
    void addDetects(Detections& detections, int batch, int width, int height, float threshold,
                    const float* predictions) const;
    void addDetects(Detections& detections, int batch, float threshold, const float* predictions) const;
    bool localMaximum(const float* predictions, int classId, int x, int y) const;

    LogisticActivation<Device::CPU> logistic_;
    PxCpuVector* poutput_ = nullptr;
    PxCpuVector* pdelta_ = nullptr;
    float focalAlpha_;
    float focalBeta_;
    float heatmapBias_;
    float sizeWeight_;
    int maxDetections_;
};

template<Device D>
CenterNetLayer<D>::CenterNetLayer(Model<D>& model, const YAML::Node& layerDef)
        : Layer<D>(model, layerDef),
          focalAlpha_(this->template property<float>("focal_alpha", 2.0f)),
          focalBeta_(this->template property<float>("focal_beta", 4.0f)),
          heatmapBias_(this->template property<float>("heatmap_bias", -2.19f)),
          sizeWeight_(this->template property<float>("size_weight", 0.1f)),
          maxDetections_(this->template property<int>("max_detections", 100))
{
    PX_CHECK(this->channels() == this->classes() + 4,
             "CenterNet input channels must equal classes + 4 (heatmap, size, offset).");
    PX_CHECK(maxDetections_ > 0, "CenterNet max_detections must be positive.");

    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->height() * this->width() * this->channels());
    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);
    setup();
}

template<Device D>
void CenterNetLayer<D>::setup()
{
    poutput_ = &this->output_;
    pdelta_ = &this->delta_;
}

template<Device D>
void CenterNetLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);
    forwardCpu(input);
}

template<Device D>
void CenterNetLayer<D>::forwardCpu(const PxCpuVector& input)
{
    poutput_->copy(input);

    const auto area = this->width() * this->height();
    const auto heatmapSize = this->classes() * area;
    auto* output = poutput_->data();
    auto* delta = pdelta_->data();

    for (auto b = 0; b < this->batch(); ++b) {
        auto* batchOutput = output + b * this->outputs();
        for (auto i = 0; i < heatmapSize; ++i) {
            batchOutput[i] += heatmapBias_;
        }
        logistic_.apply(batchOutput, batchOutput + heatmapSize);
    }

    if (this->inferring()) {
        return;
    }

    CenterNetTargetBuilder builder(this->classes(), this->width(), this->height());
    std::vector<CenterNetTargets> targets;
    targets.reserve(this->batch());
    std::size_t totalObjects = 0;
    for (auto b = 0; b < this->batch(); ++b) {
        targets.emplace_back(builder.build(this->groundTruth(b)));
        totalObjects += targets.back().objects;
    }
    const auto normalizer = 1.0f / std::max<std::size_t>(1, totalObjects);

    auto heatmapLoss = 0.0f;
    auto sizeLoss = 0.0f;
    auto offsetLoss = 0.0f;

    for (auto b = 0; b < this->batch(); ++b) {
        auto* batchOutput = output + b * this->outputs();
        auto* batchDelta = delta + b * this->outputs();
        const auto& target = targets[b];

        for (auto i = 0; i < heatmapSize; ++i) {
            const auto prediction = std::clamp(batchOutput[i], 1e-4f, 1.0f - 1e-4f);
            const auto truth = target.heatmap[i];
            if (truth >= 1.0f - 1e-6f) {
                const auto oneMinus = 1.0f - prediction;
                heatmapLoss -= std::pow(oneMinus, focalAlpha_) * std::log(prediction);
                batchDelta[i] = normalizer * (
                        std::pow(oneMinus, focalAlpha_ + 1.0f)
                        - focalAlpha_ * prediction * std::pow(oneMinus, focalAlpha_) * std::log(prediction));
            } else {
                const auto weight = std::pow(1.0f - truth, focalBeta_);
                heatmapLoss -= weight * std::pow(prediction, focalAlpha_) * std::log(1.0f - prediction);
                batchDelta[i] = normalizer * weight * (
                        focalAlpha_ * std::pow(prediction, focalAlpha_) * (1.0f - prediction)
                        * std::log(1.0f - prediction) - std::pow(prediction, focalAlpha_ + 1.0f));
            }
        }

        const auto sizeOffset = heatmapSize;
        const auto centerOffset = heatmapSize + 2 * area;
        for (auto i = 0; i < area; ++i) {
            if (target.mask[i] == 0.0f) {
                continue;
            }
            for (auto channel = 0; channel < 2; ++channel) {
                const auto mapIndex = channel * area + i;
                const auto sizeError = target.size[mapIndex] - batchOutput[sizeOffset + mapIndex];
                const auto centerError = target.offset[mapIndex] - batchOutput[centerOffset + mapIndex];
                sizeLoss += std::abs(sizeError);
                offsetLoss += std::abs(centerError);
                batchDelta[sizeOffset + mapIndex] = sizeWeight_ * normalizer *
                                                    ((sizeError > 0.0f) - (sizeError < 0.0f));
                batchDelta[centerOffset + mapIndex] = normalizer *
                                                      ((centerError > 0.0f) - (centerError < 0.0f));
            }
        }
    }

    this->cost_ = normalizer * (heatmapLoss + sizeWeight_ * sizeLoss + offsetLoss);
}

template<Device D>
void CenterNetLayer<D>::backward(const V& input, V* grad)
{
    Layer<D>::backward(input, grad);
    if (grad != nullptr) {
        cblas_saxpy(this->batch() * this->outputs(), 1.0f, this->delta_.data(), 1, grad->data(), 1);
    }
}

template<Device D>
bool CenterNetLayer<D>::localMaximum(const float* predictions, int classId, int x, int y) const
{
    const auto area = this->width() * this->height();
    const auto value = predictions[classId * area + y * this->width() + x];
    for (auto yy = std::max(0, y - 1); yy <= std::min(this->height() - 1, y + 1); ++yy) {
        for (auto xx = std::max(0, x - 1); xx <= std::min(this->width() - 1, x + 1); ++xx) {
            if ((xx != x || yy != y) && predictions[classId * area + yy * this->width() + xx] > value) {
                return false;
            }
        }
    }
    return true;
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, int batch, int width, int height, float threshold,
                                   const float* predictions) const
{
    const auto area = this->width() * this->height();
    const auto sizeOffset = this->classes() * area;
    const auto centerOffset = sizeOffset + 2 * area;
    const auto imageWidth = width > 0 ? float(width) : 1.0f;
    const auto imageHeight = height > 0 ? float(height) : 1.0f;
    Detections candidates;

    for (auto classId = 0; classId < this->classes(); ++classId) {
        for (auto y = 0; y < this->height(); ++y) {
            for (auto x = 0; x < this->width(); ++x) {
                const auto index = y * this->width() + x;
                const auto score = predictions[classId * area + index];
                if (!std::isfinite(score) || score < threshold ||
                    !localMaximum(predictions, classId, x, y)) {
                    continue;
                }

                const auto boxWidth = predictions[sizeOffset + index];
                const auto boxHeight = predictions[sizeOffset + area + index];
                const auto offsetX = predictions[centerOffset + index];
                const auto offsetY = predictions[centerOffset + area + index];
                if (!std::isfinite(boxWidth) || !std::isfinite(boxHeight) ||
                    !std::isfinite(offsetX) || !std::isfinite(offsetY) ||
                    boxWidth <= 0.0f || boxHeight <= 0.0f) {
                    continue;
                }

                const auto centerX = (x + offsetX) / this->width();
                const auto centerY = (y + offsetY) / this->height();
                cv::Rect2f box{
                        (centerX - boxWidth / 2.0f) * imageWidth,
                        (centerY - boxHeight / 2.0f) * imageHeight,
                        boxWidth * imageWidth,
                        boxHeight * imageHeight
                };
                candidates.emplace_back(box, batch, classId, score);
            }
        }
    }

    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.prob() > rhs.prob();
    });
    if (candidates.size() > static_cast<std::size_t>(maxDetections_)) {
        candidates.erase(candidates.begin() + maxDetections_, candidates.end());
    }
    detections.insert(detections.end(), candidates.begin(), candidates.end());
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, int batch, float threshold,
                                   const float* predictions) const
{
    addDetects(detections, batch, 0, 0, threshold, predictions);
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, int width, int height, float threshold)
{
    for (auto b = 0; b < this->batch(); ++b) {
        addDetects(detections, b, width, height, threshold,
                   this->output_.data() + b * this->outputs());
    }
}

template<Device D>
void CenterNetLayer<D>::addDetects(Detections& detections, float threshold)
{
    for (auto b = 0; b < this->batch(); ++b) {
        addDetects(detections, b, threshold, this->output_.data() + b * this->outputs());
    }
}

template<Device D>
std::ostream& CenterNetLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "centernet", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });
    return os;
}

using CpuCenterNet = CenterNetLayer<>;
using CudaCenterNet = CenterNetLayer<Device::CUDA>;

} // namespace px

#ifdef USE_CUDA
#include "cuda/CenterNetLayer.h"
#endif
