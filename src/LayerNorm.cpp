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

#include <cmath>
#include <cblas.h>

#include "Model.h"
#include "LayerNorm.h"

namespace px {

using CpuLayer = LayerNorm<>;
using V = CpuLayer::V;

template<>
LayerNorm<>::LayerNorm(Model<>& model, YAML::Node layerDef) : Layer<>(model, layerDef)
{
    epsilon_ = this->property<float>("epsilon", 1e-5f);
    PX_CHECK(std::isfinite(epsilon_) && epsilon_ > 0.0f,
             "layernorm epsilon must be finite and positive");

    this->setOutChannels(this->channels());
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->outHeight() * this->outWidth() * this->outChannels());

    biases_ = V(this->channels(), 0.0f);
    biasUpdates_ = V(this->channels(), 0.0f);
    scales_ = V(this->channels(), 1.0f);
    scaleUpdates_ = V(this->channels(), 0.0f);

    const auto spatial = this->height() * this->width();
    mean_ = V(this->batch() * spatial, 0.0f);
    variance_ = V(this->batch() * spatial, 0.0f);
    normalized_ = V(this->batch() * this->outputs(), 0.0f);
    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);
}

template<>
void CpuLayer::forward(const V& input)
{
    Layer<>::forward(input);

    const auto batch = this->batch();
    const auto channels = this->outChannels();
    const auto spatialSize = this->outHeight() * this->outWidth();
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < spatialSize; ++s) {
            float average = 0.0f;
            for (int c = 0; c < channels; ++c) {
                const auto index = s + spatialSize * (c + channels * b);
                average += input[index];
            }
            average /= static_cast<float>(channels);

            float variance = 0.0f;
            for (int c = 0; c < channels; ++c) {
                const auto index = s + spatialSize * (c + channels * b);
                const auto centered = input[index] - average;
                variance += centered * centered;
            }
            variance /= static_cast<float>(channels);

            const auto token = b * spatialSize + s;
            mean_[token] = average;
            variance_[token] = variance;
            const auto inverseStandardDeviation = 1.0f / std::sqrt(variance + epsilon_);
            for (int c = 0; c < channels; ++c) {
                const auto index = s + spatialSize * (c + channels * b);
                const auto normalized = (input[index] - average) * inverseStandardDeviation;
                normalized_[index] = normalized;
                this->output_[index] = scales_[c] * normalized + biases_[c];
            }
        }
    }
}

template<>
void CpuLayer::backward(const V& input, V* grad)
{
    Layer<>::backward(input, grad);

    const auto batch = this->batch();
    const auto channels = this->outChannels();
    const auto spatialSize = this->outHeight() * this->outWidth();
    scaleUpdates_.fill(0.0f);
    biasUpdates_.fill(0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < spatialSize; ++s) {
            const auto token = b * spatialSize + s;
            const auto inverseStandardDeviation = 1.0f / std::sqrt(variance_[token] + epsilon_);
            float sumScaledDelta = 0.0f;
            float sumScaledDeltaNormalized = 0.0f;
            for (int c = 0; c < channels; ++c) {
                const auto index = s + spatialSize * (c + channels * b);
                const auto scaledDelta = this->delta_[index] * scales_[c];
                sumScaledDelta += scaledDelta;
                sumScaledDeltaNormalized += scaledDelta * normalized_[index];
                scaleUpdates_[c] += this->delta_[index] * normalized_[index];
                biasUpdates_[c] += this->delta_[index];
            }
            for (int c = 0; c < channels; ++c) {
                const auto index = s + spatialSize * (c + channels * b);
                const auto scaledDelta = this->delta_[index] * scales_[c];
                this->delta_[index] = inverseStandardDeviation / static_cast<float>(channels)
                        * (static_cast<float>(channels) * scaledDelta
                           - sumScaledDelta
                           - normalized_[index] * sumScaledDeltaNormalized);
            }
        }
    }

    if (grad != nullptr) {
        cblas_saxpy(this->batch() * this->outputs(), 1.0f, this->delta_.data(), 1, grad->data(), 1);
    }
}

template<>
void CpuLayer::update()
{
    const auto learningRate = this->model().learningRate();
    const auto momentum = this->model().momentum();
    const auto batch = this->model().updateBatch();

    Layer<>::update();
    cblas_saxpy(this->outChannels(), learningRate / batch, biasUpdates_.data(), 1, biases_.data(), 1);
    cblas_sscal(this->outChannels(), momentum, biasUpdates_.data(), 1);
    cblas_saxpy(this->outChannels(), learningRate / batch, scaleUpdates_.data(), 1, scales_.data(), 1);
    cblas_sscal(this->outChannels(), momentum, scaleUpdates_.data(), 1);
}

template<>
std::streamoff CpuLayer::loadWeights(std::istream& is)
{
    const auto start = is.tellg();
    is.read(reinterpret_cast<char*>(biases_.data()), biases_.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(scales_.data()), scales_.size() * sizeof(float));
    PX_CHECK(is.good(), "Could not read layer normalization parameters");
    return is.tellg() - start;
}

template<>
std::streamoff CpuLayer::saveWeights(std::ostream& os)
{
    const auto start = os.tellp();
    os.write(reinterpret_cast<const char*>(biases_.data()), biases_.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(scales_.data()), scales_.size() * sizeof(float));
    PX_CHECK(os.good(), "Could not write layer normalization parameters");
    return os.tellp() - start;
}

template<>
std::ostream& CpuLayer::print(std::ostream& os)
{
    Layer<>::print(os, "layernorm",
                              { this->height(), this->width(), this->channels() },
                              { this->outHeight(), this->outWidth(), this->outChannels() });
    return os;
}

template<>
void CpuLayer::scaleGradients()
{
    Layer<>::scaleGradients();
    this->scaleTensor(biasUpdates_);
    this->scaleTensor(scaleUpdates_);
}

template<>
void CpuLayer::clipGradients()
{
    Layer<>::clipGradients();
    constrain(biasUpdates_.size(), this->gradientClipValue_, biasUpdates_.data(), 1);
    constrain(scaleUpdates_.size(), this->gradientClipValue_, scaleUpdates_.data(), 1);
}

template<>
void CpuLayer::copyScales(const V& scales)
{
    PX_CHECK(scales.size() == scales_.size(), "layer normalization scales have the wrong size");
    scales_.copy(scales);
}

template<>
void CpuLayer::copyBiases(const V& biases)
{
    PX_CHECK(biases.size() == biases_.size(), "layer normalization biases have the wrong size");
    biases_.copy(biases);
}

} // namespace px
