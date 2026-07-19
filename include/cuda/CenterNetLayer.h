#pragma once

namespace px {

template<>
class CenterNetExtras<Device::CUDA>
{
protected:
    PxCpuVector cpuOutput_;
    PxCpuVector cpuDelta_;
};

template<>
inline void CenterNetLayer<Device::CUDA>::setup()
{
    cpuOutput_ = PxCpuVector(this->output_.size());
    cpuDelta_ = PxCpuVector(this->delta_.size());
    poutput_ = &cpuOutput_;
    pdelta_ = &cpuDelta_;
}

template<>
inline void CenterNetLayer<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);
    PxCpuVector cpuInput(input.size());
    cpuInput.copyDevice(input.data(), input.size());
    forwardCpu(cpuInput);
    this->output_.copyHost(cpuOutput_.data(), cpuOutput_.size());
    this->delta_.copyHost(cpuDelta_.data(), cpuDelta_.size());
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
