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

template<Device D>
class AvgPoolLayer : public Layer<D>
{
public:
    using V = typename Layer<D>::V;

    AvgPoolLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;

    std::ostream& print(std::ostream& os) override;

private:

};

template<Device D>
AvgPoolLayer<D>::AvgPoolLayer(Model<D>& model, const Node& layerDef) : Layer<D>(model, layerDef)
{
    this->setOutChannels(this->channels());
    this->setOutHeight(1);
    this->setOutWidth(1);
    this->setOutputs(this->channels());

    auto outputSize = this->batch() * this->outputs();

    this->output_ = V(outputSize, 0.0f);
    this->delta_ = V(outputSize, 0.0f);
}

template<Device D>
void AvgPoolLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    auto c = this->channels();
    auto area = std::max(1, this->width() * this->height());

    auto poutput = this->output_.data();
    auto pinput = input.data();

    for (auto b = 0; b < this->batch(); ++b) {
        for (auto k = 0; k < c; ++k) {
            auto outIndex = k + b * c;
            poutput[outIndex] = 0.0f;
            for (auto i = 0; i < area; ++i) {
                auto inIndex = i + area * (k + b * c);
                poutput[outIndex] += pinput[inIndex];
            }
            poutput[outIndex] /= area;
        }
    }
}

template<Device D>
void AvgPoolLayer<D>::backward(const V& input)
{
    Layer<D>::backward(input);

    auto c = this->channels();
    auto area = std::max(1, this->width() * this->height());

    auto pdelta = this->delta_.data();
    auto pNetDelta = this->netDelta()->data();

    for (auto b = 0; b < this->batch(); ++b) {
        for (auto k = 0; k < this->channels(); ++k) {
            auto outIndex = k + b * c;
            for (auto i = 0; i < area; ++i) {
                auto inIndex = i + area * (k + b * c);
                pNetDelta[inIndex] += pdelta[outIndex] / area;
            }
        }
    }
}

template<Device D>
std::ostream& AvgPoolLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "avgpool", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

}   // px
