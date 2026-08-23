/********************************************************************************
* Copyright 2026 Thomas A. Rieck, All Rights Reserved
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

#include <algorithm>
#include <cmath>
#include <cblas.h>
#include <limits>

#include "Layer.h"

namespace px {

/**
 * Single-head scaled dot-product self-attention over spatial tokens.
 *
 * A PixieNN tensor has the layout [batch, channels, height, width]. The
 * channels at each spatial location form one token embedding, and attention
 * connects every spatial token to every other token in the same image.
 * This CPU reference layer includes trainable query, key, value, and output
 * projections. The optional `heads` property splits the channel embedding
 * into independent attention heads before the output projection recombines
 * them. The default is one head.
 */
template<Device D = Device::CPU>
class SelfAttention : public Layer<D>
{
public:
    using V = Layer<D>::V;

    SelfAttention(Model<D>& model, YAML::Node layerDef);

    void forward(const V& input) override;
    void backward(const V& input, V* grad) override;
    void update() override;

    std::streamoff loadWeights(std::istream& is) override;
    std::streamoff saveWeights(std::ostream& os) override;

    const V& queryWeights() const noexcept;
    const V& keyWeights() const noexcept;
    const V& valueWeights() const noexcept;
    const V& outputWeights() const noexcept;

    void copyQueryWeights(const V& weights);
    void copyKeyWeights(const V& weights);
    void copyValueWeights(const V& weights);
    void copyOutputWeights(const V& weights);

    std::ostream& print(std::ostream& os) override;

private:
    float scale_ = 1.0f;
    int channels_ = 0;
    int heads_ = 1;
    int headChannels_ = 0;
    int tokens_ = 0;
    V attention_;
    V query_, key_, value_, context_;
    V contextGradient_, queryGradient_, keyGradient_, valueGradient_;
    V inputGradient_;
    V queryWeights_, keyWeights_, valueWeights_, outputWeights_;
    V queryBiases_, keyBiases_, valueBiases_, outputBiases_;
    V queryUpdates_, keyUpdates_, valueUpdates_, outputUpdates_;
    V queryBiasUpdates_, keyBiasUpdates_, valueBiasUpdates_, outputBiasUpdates_;
};

using CpuSelfAttention = SelfAttention<>;

template<>
inline SelfAttention<>::SelfAttention(Model<>& model, YAML::Node layerDef) : Layer<>(model, layerDef)
{
    PX_CHECK(this->channels() > 0, "self-attention requires positive channels");
    PX_CHECK(this->height() > 0 && this->width() > 0,
             "self-attention requires positive height and width");

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
    key_ = V(query_.size(), 0.0f);
    value_ = V(query_.size(), 0.0f);
    context_ = V(query_.size(), 0.0f);
    contextGradient_ = V(query_.size(), 0.0f);
    queryGradient_ = V(query_.size(), 0.0f);
    keyGradient_ = V(query_.size(), 0.0f);
    valueGradient_ = V(query_.size(), 0.0f);
    inputGradient_ = V(this->batch() * this->outputs(), 0.0f);
    const auto projectionSize = channels_ * channels_;
    queryWeights_ = V(projectionSize, 0.0f);
    keyWeights_ = V(projectionSize, 0.0f);
    valueWeights_ = V(projectionSize, 0.0f);
    outputWeights_ = V(projectionSize, 0.0f);
    queryBiases_ = V(channels_, 0.0f);
    keyBiases_ = V(channels_, 0.0f);
    valueBiases_ = V(channels_, 0.0f);
    outputBiases_ = V(channels_, 0.0f);
    queryUpdates_ = V(projectionSize, 0.0f);
    keyUpdates_ = V(projectionSize, 0.0f);
    valueUpdates_ = V(projectionSize, 0.0f);
    outputUpdates_ = V(projectionSize, 0.0f);
    queryBiasUpdates_ = V(channels_, 0.0f);
    keyBiasUpdates_ = V(channels_, 0.0f);
    valueBiasUpdates_ = V(channels_, 0.0f);
    outputBiasUpdates_ = V(channels_, 0.0f);
    for (auto c = 0; c < channels_; ++c) {
        queryWeights_[c * channels_ + c] = 1.0f;
        keyWeights_[c * channels_ + c] = 1.0f;
        valueWeights_[c * channels_ + c] = 1.0f;
        outputWeights_[c * channels_ + c] = 1.0f;
    }
    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);
}

template<>
inline void SelfAttention<>::forward(const V& input)
{
    Layer<>::forward(input);

    const auto batch = this->batch();
    const auto channels = channels_;
    for (auto b = 0; b < batch; ++b) {
        const auto batchOffset = b * heads_ * tokens_ * tokens_;
        const auto inputOffset = b * this->outputs();
        const auto sequenceOffset = b * tokens_ * channels;
        for (auto i = 0; i < tokens_; ++i) {
            for (auto o = 0; o < channels; ++o) {
                auto query = queryBiases_[o];
                auto key = keyBiases_[o];
                auto value = valueBiases_[o];
                for (auto c = 0; c < channels; ++c) {
                    const auto inputValue = input[inputOffset + i + tokens_ * c];
                    query += queryWeights_[o * channels + c] * inputValue;
                    key += keyWeights_[o * channels + c] * inputValue;
                    value += valueWeights_[o * channels + c] * inputValue;
                }
                query_[sequenceOffset + i * channels + o] = query;
                key_[sequenceOffset + i * channels + o] = key;
                value_[sequenceOffset + i * channels + o] = value;
            }
        }
        for (auto head = 0; head < heads_; ++head) {
            const auto headOffset = batchOffset + head * tokens_ * tokens_;
            const auto channelOffset = head * headChannels_;
            for (auto i = 0; i < tokens_; ++i) {
                auto maximum = -std::numeric_limits<float>::infinity();
                for (auto j = 0; j < tokens_; ++j) {
                    auto score = 0.0f;
                    for (auto c = 0; c < headChannels_; ++c) {
                        score += query_[sequenceOffset + i * channels + channelOffset + c]
                                * key_[sequenceOffset + j * channels + channelOffset + c];
                    }
                    score *= scale_;
                    maximum = std::max(maximum, score);
                    attention_[headOffset + i * tokens_ + j] = score;
                }

                auto total = 0.0f;
                for (auto j = 0; j < tokens_; ++j) {
                    auto& weight = attention_[headOffset + i * tokens_ + j];
                    weight = std::exp(weight - maximum);
                    total += weight;
                }
                for (auto j = 0; j < tokens_; ++j) {
                    attention_[headOffset + i * tokens_ + j] /= total;
                }

                for (auto c = 0; c < headChannels_; ++c) {
                    auto result = 0.0f;
                    for (auto j = 0; j < tokens_; ++j) {
                        result += attention_[headOffset + i * tokens_ + j]
                                * value_[sequenceOffset + j * channels + channelOffset + c];
                    }
                    context_[sequenceOffset + i * channels + channelOffset + c] = result;
                }
            }
        }
        for (auto i = 0; i < tokens_; ++i) {
            for (auto c = 0; c < channels; ++c) {
                auto output = outputBiases_[c];
                for (auto o = 0; o < channels; ++o) {
                    output += outputWeights_[c * channels + o]
                            * context_[sequenceOffset + i * channels + o];
                }
                this->output_[inputOffset + i + tokens_ * c] = output;
            }
        }
    }
}

template<>
inline void SelfAttention<>::backward(const V& input, V* grad)
{
    Layer<>::backward(input, grad);
    inputGradient_.fill(0.0f);
    contextGradient_.fill(0.0f);
    queryGradient_.fill(0.0f);
    keyGradient_.fill(0.0f);
    valueGradient_.fill(0.0f);
    queryUpdates_.fill(0.0f);
    keyUpdates_.fill(0.0f);
    valueUpdates_.fill(0.0f);
    outputUpdates_.fill(0.0f);
    queryBiasUpdates_.fill(0.0f);
    keyBiasUpdates_.fill(0.0f);
    valueBiasUpdates_.fill(0.0f);
    outputBiasUpdates_.fill(0.0f);

    const auto batch = this->batch();
    const auto channels = channels_;
    for (auto b = 0; b < batch; ++b) {
        const auto batchOffset = b * heads_ * tokens_ * tokens_;
        const auto inputOffset = b * this->outputs();
        const auto sequenceOffset = b * tokens_ * channels;
        for (auto i = 0; i < tokens_; ++i) {
            for (auto o = 0; o < channels; ++o) {
                const auto outputIndex = inputOffset + i + tokens_ * o;
                const auto outputGradient = this->delta_[outputIndex];
                outputBiasUpdates_[o] += outputGradient;
                for (auto c = 0; c < channels; ++c) {
                    outputUpdates_[o * channels + c] += outputGradient
                            * context_[sequenceOffset + i * channels + c];
                    contextGradient_[sequenceOffset + i * channels + c] += outputWeights_[o * channels + c]
                            * outputGradient;
                }
            }

            for (auto head = 0; head < heads_; ++head) {
                const auto headOffset = batchOffset + head * tokens_ * tokens_;
                const auto channelOffset = head * headChannels_;
                auto weightedGradient = 0.0f;
                for (auto j = 0; j < tokens_; ++j) {
                    auto valueGradient = 0.0f;
                    for (auto c = 0; c < headChannels_; ++c) {
                        valueGradient += contextGradient_[sequenceOffset + i * channels + channelOffset + c]
                                * value_[sequenceOffset + j * channels + channelOffset + c];
                    }
                    weightedGradient += attention_[headOffset + i * tokens_ + j] * valueGradient;
                }

                for (auto j = 0; j < tokens_; ++j) {
                    auto valueGradient = 0.0f;
                    for (auto c = 0; c < headChannels_; ++c) {
                        valueGradient += contextGradient_[sequenceOffset + i * channels + channelOffset + c]
                                * value_[sequenceOffset + j * channels + channelOffset + c];
                    }
                    const auto scoreGradient = attention_[headOffset + i * tokens_ + j]
                            * (valueGradient - weightedGradient);

                    for (auto c = 0; c < headChannels_; ++c) {
                        const auto queryIndex = sequenceOffset + i * channels + channelOffset + c;
                        const auto keyIndex = sequenceOffset + j * channels + channelOffset + c;
                        valueGradient_[keyIndex] += attention_[headOffset + i * tokens_ + j]
                                * contextGradient_[queryIndex];
                        queryGradient_[queryIndex] += scoreGradient * key_[keyIndex] * scale_;
                        keyGradient_[keyIndex] += scoreGradient * query_[queryIndex] * scale_;
                    }
                }
            }
        }

        for (auto i = 0; i < tokens_; ++i) {
            for (auto o = 0; o < channels; ++o) {
                const auto queryGradient = queryGradient_[sequenceOffset + i * channels + o];
                const auto keyGradient = keyGradient_[sequenceOffset + i * channels + o];
                const auto valueGradient = valueGradient_[sequenceOffset + i * channels + o];
                queryBiasUpdates_[o] += queryGradient;
                keyBiasUpdates_[o] += keyGradient;
                valueBiasUpdates_[o] += valueGradient;
                for (auto c = 0; c < channels; ++c) {
                    const auto inputIndex = inputOffset + i + tokens_ * c;
                    const auto inputValue = input[inputIndex];
                    queryUpdates_[o * channels + c] += queryGradient * inputValue;
                    keyUpdates_[o * channels + c] += keyGradient * inputValue;
                    valueUpdates_[o * channels + c] += valueGradient * inputValue;
                    inputGradient_[inputIndex] += queryWeights_[o * channels + c] * queryGradient
                            + keyWeights_[o * channels + c] * keyGradient
                            + valueWeights_[o * channels + c] * valueGradient;
                }
            }
        }
    }

    this->delta_.copy(inputGradient_);
    if (grad != nullptr) {
        cblas_saxpy(this->batch() * this->outputs(), 1.0f,
                   this->delta_.data(), 1, grad->data(), 1);
    }
}

template<>
inline void SelfAttention<>::update()
{
    Layer<>::update();
    const auto rate = this->model().learningRate() / this->model().updateBatch();
    const auto momentum = this->model().momentum();
    const auto update = [rate, momentum](V& weights, V& updates) {
        cblas_saxpy(weights.size(), rate, updates.data(), 1, weights.data(), 1);
        cblas_sscal(updates.size(), momentum, updates.data(), 1);
    };
    update(queryWeights_, queryUpdates_);
    update(keyWeights_, keyUpdates_);
    update(valueWeights_, valueUpdates_);
    update(outputWeights_, outputUpdates_);
    update(queryBiases_, queryBiasUpdates_);
    update(keyBiases_, keyBiasUpdates_);
    update(valueBiases_, valueBiasUpdates_);
    update(outputBiases_, outputBiasUpdates_);
}

template<>
inline std::streamoff SelfAttention<>::loadWeights(std::istream& is)
{
    const auto start = is.tellg();
    const auto read = [&is](V& values) {
        is.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
    };
    read(queryWeights_);
    read(queryBiases_);
    read(keyWeights_);
    read(keyBiases_);
    read(valueWeights_);
    read(valueBiases_);
    read(outputWeights_);
    read(outputBiases_);
    PX_CHECK(is.good(), "Could not read self-attention parameters");
    return is.tellg() - start;
}

template<>
inline std::streamoff SelfAttention<>::saveWeights(std::ostream& os)
{
    const auto start = os.tellp();
    const auto write = [&os](const V& values) {
        os.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    };
    write(queryWeights_);
    write(queryBiases_);
    write(keyWeights_);
    write(keyBiases_);
    write(valueWeights_);
    write(valueBiases_);
    write(outputWeights_);
    write(outputBiases_);
    PX_CHECK(os.good(), "Could not write self-attention parameters");
    return os.tellp() - start;
}

template<>
inline const SelfAttention<>::V& SelfAttention<>::queryWeights() const noexcept
{
    return queryWeights_;
}

template<>
inline const SelfAttention<>::V& SelfAttention<>::keyWeights() const noexcept
{
    return keyWeights_;
}

template<>
inline const SelfAttention<>::V& SelfAttention<>::valueWeights() const noexcept
{
    return valueWeights_;
}

template<>
inline const SelfAttention<>::V& SelfAttention<>::outputWeights() const noexcept
{
    return outputWeights_;
}

template<>
inline void SelfAttention<>::copyQueryWeights(const V& weights)
{
    PX_CHECK(weights.size() == queryWeights_.size(), "self-attention query weights have the wrong size");
    queryWeights_.copy(weights);
}

template<>
inline void SelfAttention<>::copyKeyWeights(const V& weights)
{
    PX_CHECK(weights.size() == keyWeights_.size(), "self-attention key weights have the wrong size");
    keyWeights_.copy(weights);
}

template<>
inline void SelfAttention<>::copyValueWeights(const V& weights)
{
    PX_CHECK(weights.size() == valueWeights_.size(), "self-attention value weights have the wrong size");
    valueWeights_.copy(weights);
}

template<>
inline void SelfAttention<>::copyOutputWeights(const V& weights)
{
    PX_CHECK(weights.size() == outputWeights_.size(), "self-attention output weights have the wrong size");
    outputWeights_.copy(weights);
}

template<>
inline std::ostream& SelfAttention<>::print(std::ostream& os)
{
    Layer<>::print(os, "self-attention",
                   { this->height(), this->width(), this->channels() },
                   { this->outHeight(), this->outWidth(), this->outChannels() });
    return os;
}

} // namespace px

#ifdef USE_CUDA
#include "cuda/SelfAttention.h"
#endif
