#pragma once

#include "CenterNetKernels.cuh"

namespace px {

template<>
class CenterNetExtras<Device::CUDA>
{
protected:
    PxCpuVector hostHeatmap_;
    PxCpuVector hostSize_;
    PxCpuVector hostOffset_;
    PxCpuVector hostMask_;
    PxCudaVector targetHeatmap_;
    PxCudaVector targetSize_;
    PxCudaVector targetOffset_;
    PxCudaVector targetMask_;
    PxCudaVector costs_;
};

template<>
inline void CenterNetLayer<Device::CUDA>::setup()
{
    const auto area = this->width() * this->height();
    hostHeatmap_ = PxCpuVector(this->batch() * this->classes() * area);
    hostSize_ = PxCpuVector(this->batch() * 2 * area);
    hostOffset_ = PxCpuVector(this->batch() * 2 * area);
    hostMask_ = PxCpuVector(this->batch() * area);
    targetHeatmap_ = PxCudaVector(hostHeatmap_.size());
    targetSize_ = PxCudaVector(hostSize_.size());
    targetOffset_ = PxCudaVector(hostOffset_.size());
    targetMask_ = PxCudaVector(hostMask_.size());
    costs_ = PxCudaVector(3);
}

template<>
inline void CenterNetLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);
    const auto area = this->width() * this->height();
    centerNetActivateGpu(input.data(), this->output_.data(), this->batch(),
                         this->classes(), area, heatmapBias_);
    if (this->inferring()) return;

    CenterNetTargetBuilder builder(this->classes(), this->width(), this->height());
    std::size_t totalObjects = 0;
    for (auto b = 0; b < this->batch(); ++b) {
        const auto targets = builder.build(this->groundTruth(b));
        totalObjects += targets.objects;
        std::copy(targets.heatmap.begin(), targets.heatmap.end(),
                  hostHeatmap_.begin() + b * this->classes() * area);
        std::copy(targets.size.begin(), targets.size.end(),
                  hostSize_.begin() + b * 2 * area);
        std::copy(targets.offset.begin(), targets.offset.end(),
                  hostOffset_.begin() + b * 2 * area);
        std::copy(targets.mask.begin(), targets.mask.end(),
                  hostMask_.begin() + b * area);
    }

    targetHeatmap_.copy(hostHeatmap_);
    targetSize_.copy(hostSize_);
    targetOffset_.copy(hostOffset_);
    targetMask_.copy(hostMask_);
    this->delta_.fill(0.0f);
    costs_.fill(0.0f);
    const auto normalizer = 1.0f / std::max<std::size_t>(1, totalObjects);
    centerNetLossGpu(this->output_.data(), this->delta_.data(),
                     targetHeatmap_.data(), targetSize_.data(), targetOffset_.data(),
                     targetMask_.data(), costs_.data(), this->batch(), this->classes(),
                     area, normalizer, focalAlpha_, focalBeta_, sizeWeight_);
    const auto costs = costs_.asVector();
    this->cost_ = normalizer * (costs[0] + sizeWeight_ * costs[1] + costs[2]);
}

template<>
inline void CenterNetLayer<Device::CUDA>::backward(const V& input, V* grad)
{
    Layer<Device::CUDA>::backward(input, grad);
    if (grad == nullptr) {
        return;
    }
    const auto alpha = 1.0f;
    const auto status = cublasSaxpy(this->cublasContext(), this->delta_.size(), &alpha,
                                   this->delta_.data(), 1, grad->data(), 1);
    PX_CHECK_CUBLAS(status);
}

template<>
inline void CenterNetLayer<Device::CUDA>::addDetects(Detections& detections, int width, int height,
                                                     float threshold)
{
    const auto output = this->output_.asVector();
    for (auto b = 0; b < this->batch(); ++b) {
        addDetects(detections, b, width, height, threshold, output.data() + b * this->outputs());
    }
}

template<>
inline void CenterNetLayer<Device::CUDA>::addDetects(Detections& detections, float threshold)
{
    const auto output = this->output_.asVector();
    for (auto b = 0; b < this->batch(); ++b) {
        addDetects(detections, b, threshold, output.data() + b * this->outputs());
    }
}

} // namespace px
