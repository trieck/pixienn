#pragma once

#include <cblas.h>
#include <cmath>
#include <istream>
#include <vector>

#include "Cublas.h"
#include "CudaUtils.cuh"
#include "TransformerKernel.cuh"

namespace px {

template<>
inline LayerNorm<Device::CUDA>::LayerNorm(Model<Device::CUDA>& model, YAML::Node layerDef)
        : Layer<Device::CUDA>(model, layerDef)
{
    epsilon_ = this->property<float>("epsilon", 1e-5f);
    PX_CHECK(std::isfinite(epsilon_) && epsilon_ > 0.0f, "layernorm epsilon must be finite and positive");
    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->outHeight() * this->outWidth() * this->outChannels());
    biases_ = V(this->channels(), 0.0f);
    biasUpdates_ = V(this->channels(), 0.0f);
    scales_ = V(this->channels(), 1.0f);
    scaleUpdates_ = V(this->channels(), 0.0f);
    mean_ = V(this->batch() * this->height() * this->width(), 0.0f);
    variance_ = V(mean_.size(), 0.0f);
    normalized_ = V(this->batch() * this->outputs(), 0.0f);
    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);
}

template<>
inline void LayerNorm<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);
    layerNormForwardGpu(this->batch(), this->channels(), this->height() * this->width(), epsilon_, input.data(),
                        scales_.data(), biases_.data(), mean_.data(), variance_.data(), normalized_.data(),
                        this->output_.data());
}

template<>
inline void LayerNorm<Device::CUDA>::backward(const V& input, V* grad)
{
    Layer<Device::CUDA>::backward(input, grad);
    biasUpdates_.fill(0.0f);
    scaleUpdates_.fill(0.0f);
    layerNormBackwardGpu(this->batch(), this->channels(), this->height() * this->width(), epsilon_, this->delta_.data(),
                         scales_.data(), normalized_.data(), variance_.data(), this->delta_.data(),
                         scaleUpdates_.data(), biasUpdates_.data());
    if (grad != nullptr) {
        auto one = 1.0f;
        PX_CHECK_CUBLAS(cublasSaxpy(this->cublasContext(), this->delta_.size(), &one, this->delta_.data(), 1,
                                    grad->data(), 1));
    }
}

template<>
inline void LayerNorm<Device::CUDA>::update()
{
    Layer<Device::CUDA>::update();
    const auto rate = this->model().learningRate() / this->model().updateBatch();
    const auto momentum = this->model().momentum();
    auto& context = this->cublasContext();
    PX_CHECK_CUBLAS(cublasSaxpy(context, scales_.size(), &rate, scaleUpdates_.data(), 1, scales_.data(), 1));
    PX_CHECK_CUBLAS(cublasSscal(context, scaleUpdates_.size(), &momentum, scaleUpdates_.data(), 1));
    PX_CHECK_CUBLAS(cublasSaxpy(context, biases_.size(), &rate, biasUpdates_.data(), 1, biases_.data(), 1));
    PX_CHECK_CUBLAS(cublasSscal(context, biasUpdates_.size(), &momentum, biasUpdates_.data(), 1));
}

template<>
inline void LayerNorm<Device::CUDA>::scaleGradients()
{
    Layer<Device::CUDA>::scaleGradients();
    this->scaleTensor(biasUpdates_);
    this->scaleTensor(scaleUpdates_);
}

template<>
inline void LayerNorm<Device::CUDA>::clipGradients()
{
    Layer<Device::CUDA>::clipGradients();
    constrainGpu(biasUpdates_.size(), this->gradientClipValue_, biasUpdates_.data());
    constrainGpu(scaleUpdates_.size(), this->gradientClipValue_, scaleUpdates_.data());
}

template<>
inline std::streamoff LayerNorm<Device::CUDA>::loadWeights(std::istream& is)
{
    const auto start = is.tellg();
    std::vector<float> biases(biases_.size()), scales(scales_.size());
    is.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(scales.data()), scales.size() * sizeof(float));
    PX_CHECK(is.good(), "Could not read CUDA layer normalization parameters");
    biases_.copyHost(biases.data(), biases.size());
    scales_.copyHost(scales.data(), scales.size());
    return is.tellg() - start;
}

template<>
inline std::streamoff LayerNorm<Device::CUDA>::saveWeights(std::ostream& os)
{
    const auto start = os.tellp();
    const auto biases = biases_.asVector();
    const auto scales = scales_.asVector();
    os.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(scales.data()), scales.size() * sizeof(float));
    PX_CHECK(os.good(), "Could not write CUDA layer normalization parameters");
    return os.tellp() - start;
}

template<>
inline void LayerNorm<Device::CUDA>::copyScales(const V& scales)
{
    PX_CHECK(scales.size() == scales_.size(), "layer normalization scales have the wrong size");
    scales_.copy(scales);
}

template<>
inline void LayerNorm<Device::CUDA>::copyBiases(const V& biases)
{
    PX_CHECK(biases.size() == biases_.size(), "layer normalization biases have the wrong size");
    biases_.copy(biases);
}

template<>
inline std::ostream& LayerNorm<Device::CUDA>::print(std::ostream& os)
{
    Layer<Device::CUDA>::print(os, "layernorm", { this->height(), this->width(), this->channels() },
                               { this->outHeight(), this->outWidth(), this->outChannels() });
    return os;
}

} // namespace px
