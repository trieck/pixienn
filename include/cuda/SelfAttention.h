#pragma once

#include <cmath>
#include <istream>
#include <vector>

#include "Cublas.h"
#include "TransformerKernel.cuh"

namespace px {

template<>
inline SelfAttention<Device::CUDA>::SelfAttention(Model<Device::CUDA>& model, YAML::Node layerDef)
        : Layer<Device::CUDA>(model, layerDef)
{
    PX_CHECK(this->channels() > 0 && this->height() > 0 && this->width() > 0,
             "self-attention requires positive dimensions");
    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->outHeight() * this->outWidth() * this->outChannels());
    channels_ = this->channels();
    heads_ = this->property<int>("heads", 1);
    PX_CHECK(heads_ > 0 && channels_ % heads_ == 0,
             "self-attention heads must be positive and divide channels");
    headChannels_ = channels_ / heads_;
    tokens_ = this->height() * this->width();
    scale_ = 1.0f / std::sqrt(static_cast<float>(headChannels_));
    attention_ = V(this->batch() * heads_ * tokens_ * tokens_, 0.0f);
    query_ = V(this->batch() * tokens_ * channels_, 0.0f);
    key_ = V(query_.size(), 0.0f); value_ = V(query_.size(), 0.0f); context_ = V(query_.size(), 0.0f);
    contextGradient_ = V(query_.size(), 0.0f); queryGradient_ = V(query_.size(), 0.0f);
    keyGradient_ = V(query_.size(), 0.0f); valueGradient_ = V(query_.size(), 0.0f);
    inputGradient_ = V(this->batch() * this->outputs(), 0.0f);
    const auto projectionSize = channels_ * channels_;
    queryWeights_ = V(projectionSize, 0.0f); keyWeights_ = V(projectionSize, 0.0f);
    valueWeights_ = V(projectionSize, 0.0f); outputWeights_ = V(projectionSize, 0.0f);
    queryBiases_ = V(channels_, 0.0f); keyBiases_ = V(channels_, 0.0f);
    valueBiases_ = V(channels_, 0.0f); outputBiases_ = V(channels_, 0.0f);
    queryUpdates_ = V(projectionSize, 0.0f); keyUpdates_ = V(projectionSize, 0.0f);
    valueUpdates_ = V(projectionSize, 0.0f); outputUpdates_ = V(projectionSize, 0.0f);
    queryBiasUpdates_ = V(channels_, 0.0f); keyBiasUpdates_ = V(channels_, 0.0f);
    valueBiasUpdates_ = V(channels_, 0.0f); outputBiasUpdates_ = V(channels_, 0.0f);
    std::vector<float> identity(projectionSize, 0.0f);
    for (auto c = 0; c < channels_; ++c) identity[c * channels_ + c] = 1.0f;
    queryWeights_.copyHost(identity.data(), identity.size()); keyWeights_.copyHost(identity.data(), identity.size());
    valueWeights_.copyHost(identity.data(), identity.size()); outputWeights_.copyHost(identity.data(), identity.size());
    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);
}

template<>
inline void SelfAttention<Device::CUDA>::forward(const V& input)
{
    Layer<Device::CUDA>::forward(input);
    selfAttentionProjectGpu(this->batch(), tokens_, channels_, input.data(), queryWeights_.data(), queryBiases_.data(),
                            keyWeights_.data(), keyBiases_.data(), valueWeights_.data(), valueBiases_.data(),
                            query_.data(), key_.data(), value_.data());
    selfAttentionForwardGpu(this->batch(), tokens_, channels_, heads_, scale_, query_.data(), key_.data(), value_.data(),
                            attention_.data(), context_.data());
    selfAttentionOutputGpu(this->batch(), tokens_, channels_, context_.data(), outputWeights_.data(),
                           outputBiases_.data(), this->output_.data());
}

template<>
inline void SelfAttention<Device::CUDA>::backward(const V& input, V* grad)
{
    Layer<Device::CUDA>::backward(input, grad);
    contextGradient_.fill(0.0f);
    queryGradient_.fill(0.0f);
    keyGradient_.fill(0.0f);
    valueGradient_.fill(0.0f);
    inputGradient_.fill(0.0f);
    queryUpdates_.fill(0.0f);
    keyUpdates_.fill(0.0f);
    valueUpdates_.fill(0.0f);
    outputUpdates_.fill(0.0f);
    queryBiasUpdates_.fill(0.0f);
    keyBiasUpdates_.fill(0.0f);
    valueBiasUpdates_.fill(0.0f);
    outputBiasUpdates_.fill(0.0f);
    selfAttentionOutputBackwardGpu(this->batch(), tokens_, channels_, this->delta_.data(), context_.data(),
                                   outputWeights_.data(), contextGradient_.data(), outputUpdates_.data(),
                                   outputBiasUpdates_.data());
    selfAttentionAttentionBackwardGpu(this->batch(), tokens_, channels_, heads_, scale_, attention_.data(),
                                      query_.data(), key_.data(), value_.data(), contextGradient_.data(),
                                      queryGradient_.data(), keyGradient_.data(), valueGradient_.data());
    selfAttentionProjectBackwardGpu(this->batch(), tokens_, channels_, input.data(), queryGradient_.data(),
                                    keyGradient_.data(), valueGradient_.data(), queryWeights_.data(),
                                    keyWeights_.data(), valueWeights_.data(), inputGradient_.data(),
                                    queryUpdates_.data(), queryBiasUpdates_.data(), keyUpdates_.data(),
                                    keyBiasUpdates_.data(), valueUpdates_.data(), valueBiasUpdates_.data());
    this->delta_.copy(inputGradient_);
    if (grad != nullptr) {
        auto one = 1.0f;
        PX_CHECK_CUBLAS(cublasSaxpy(this->cublasContext(), this->delta_.size(), &one, this->delta_.data(), 1,
                                    grad->data(), 1));
    }
}

template<>
inline void SelfAttention<Device::CUDA>::update()
{
    Layer<Device::CUDA>::update();
    auto rate = this->model().learningRate() / this->model().updateBatch();
    auto momentum = this->model().momentum();
    auto& c = this->cublasContext();
    const auto update = [&c, rate, momentum](V& weights, V& values) {
        PX_CHECK_CUBLAS(cublasSaxpy(c, weights.size(), &rate, values.data(), 1, weights.data(), 1));
        PX_CHECK_CUBLAS(cublasSscal(c, values.size(), &momentum, values.data(), 1));
    };
    update(queryWeights_, queryUpdates_); update(keyWeights_, keyUpdates_);
    update(valueWeights_, valueUpdates_); update(outputWeights_, outputUpdates_);
    update(queryBiases_, queryBiasUpdates_); update(keyBiases_, keyBiasUpdates_);
    update(valueBiases_, valueBiasUpdates_); update(outputBiases_, outputBiasUpdates_);
}

template<>
inline std::streamoff SelfAttention<Device::CUDA>::loadWeights(std::istream& is)
{
    const auto start = is.tellg();
    const auto read = [&is](V& values) { std::vector<float> host(values.size()); is.read(reinterpret_cast<char*>(host.data()), host.size()*sizeof(float)); values.copyHost(host.data(), host.size()); };
    read(queryWeights_); read(queryBiases_); read(keyWeights_); read(keyBiases_);
    read(valueWeights_); read(valueBiases_); read(outputWeights_); read(outputBiases_);
    PX_CHECK(is.good(), "Could not read CUDA self-attention parameters");
    return is.tellg() - start;
}

template<>
inline std::streamoff SelfAttention<Device::CUDA>::saveWeights(std::ostream& os)
{
    const auto start = os.tellp();
    const auto write = [&os](const V& values) { const auto host = values.asVector(); os.write(reinterpret_cast<const char*>(host.data()), host.size()*sizeof(float)); };
    write(queryWeights_); write(queryBiases_); write(keyWeights_); write(keyBiases_);
    write(valueWeights_); write(valueBiases_); write(outputWeights_); write(outputBiases_);
    PX_CHECK(os.good(), "Could not write CUDA self-attention parameters");
    return os.tellp() - start;
}

template<>
inline const SelfAttention<Device::CUDA>::V& SelfAttention<Device::CUDA>::queryWeights() const noexcept { return queryWeights_; }
template<>
inline const SelfAttention<Device::CUDA>::V& SelfAttention<Device::CUDA>::keyWeights() const noexcept { return keyWeights_; }
template<>
inline const SelfAttention<Device::CUDA>::V& SelfAttention<Device::CUDA>::valueWeights() const noexcept { return valueWeights_; }
template<>
inline const SelfAttention<Device::CUDA>::V& SelfAttention<Device::CUDA>::outputWeights() const noexcept { return outputWeights_; }
template<>
inline void SelfAttention<Device::CUDA>::copyQueryWeights(const V& v) { PX_CHECK(v.size()==queryWeights_.size(), "invalid query weights"); queryWeights_.copy(v); }
template<>
inline void SelfAttention<Device::CUDA>::copyKeyWeights(const V& v) { PX_CHECK(v.size()==keyWeights_.size(), "invalid key weights"); keyWeights_.copy(v); }
template<>
inline void SelfAttention<Device::CUDA>::copyValueWeights(const V& v) { PX_CHECK(v.size()==valueWeights_.size(), "invalid value weights"); valueWeights_.copy(v); }
template<>
inline void SelfAttention<Device::CUDA>::copyOutputWeights(const V& v) { PX_CHECK(v.size()==outputWeights_.size(), "invalid output weights"); outputWeights_.copy(v); }

template<>
inline std::ostream& SelfAttention<Device::CUDA>::print(std::ostream& os)
{
    Layer<Device::CUDA>::print(os, "self-attention", { this->height(), this->width(), this->channels() },
                               { this->outHeight(), this->outWidth(), this->outChannels() });
    return os;
}

} // namespace px
