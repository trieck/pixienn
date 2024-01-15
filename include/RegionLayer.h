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
class RegionLayer : public Layer<D>, public Detector
{
public:
    using V = typename Layer<D>::V;

    RegionLayer(Model<D>& model, const YAML::Node& layerDef);

    void forward(const V& input) override;
    void backward(const V& input) override;
    void update() override;

    std::ostream& print(std::ostream& os) override;

    void addDetects(Detections& detections, float threshold) override;
    void addDetects(Detections& detections, int width, int height, float threshold) override;

private:
    void forwardCpu(const PxCpuVector& input, PxCpuVector& output);

    void addDetects(Detections& detections, int width, int height, float threshold, const float* predictions) const;
    void addDetects(Detections& detections, float threshold, const float* predictions) const;

    DarkBox regionBox(const float* p, int n, int index, int i, int j) const;
    int entryIndex(int batch, int location, int entry) const noexcept;

    LogisticActivation <Device::CPU> logistic_;

    PxCpuVector biases_;
    std::vector<float> anchors_;
    bool biasMatch_, softmax_, rescore_;
    int coords_, num_;
    float objectScale_, noobjectScale_, classScale_, coordScale_, thresh_;
};

template<Device D>
RegionLayer<D>::RegionLayer(Model<D>& model, const YAML::Node& layerDef) : Layer<D>(model, layerDef)
{
    anchors_ = this->template property<std::vector<float>>("anchors");
    biasMatch_ = this->template property<bool>("bias_match", false);
    classScale_ = this->template property<float>("class_scale", 1.0f);
    coordScale_ = this->template property<float>("coord_scale", 1.0f);
    coords_ = this->template property<int>("coords", 4);
    noobjectScale_ = this->template property<float>("noobject_scale", 1.0f);
    num_ = this->template property<int>("num", 1);
    objectScale_ = this->template property<float>("object_scale", 1.0f);
    rescore_ = this->template property<bool>("rescore", false);
    softmax_ = this->template property<bool>("softmax", false);
    thresh_ = this->template property<float>("thresh", 0.5f);

    PX_CHECK(anchors_.size() == 2 * num_, "Anchors size does not match number of regions.");

    this->setOutChannels(num_ * (coords_ + 1 + this->classes()));
    this->setOutHeight(this->height());
    this->setOutWidth(this->width());
    this->setOutputs(this->height() * this->width() * num_ * (coords_ + 1 + this->classes()));

    biases_ = PxCpuVector(num_ * 2, 0.0f);
    this->output_ = V(this->batch() * this->outputs(), 0.0f);
    this->delta_ = V(this->batch() * this->outputs(), 0.0f);

    for (auto i = 0; i < num_ * 2; ++i) {
        biases_[i] = anchors_[i];
    }
}

template<Device D>
std::ostream& RegionLayer<D>::print(std::ostream& os)
{
    Layer<D>::print(os, "region", { this->height(), this->width(), this->channels() },
                    { this->outHeight(), this->outWidth(), this->outChannels() });

    return os;
}

template<Device D>
int RegionLayer<D>::entryIndex(int batch, int location, int entry) const noexcept
{
    auto n = location / (this->width() * this->height());
    auto loc = location % (this->width() * this->height());

    return batch * this->outputs() + n * this->width() * this->height() * (coords_ + 1 + this->classes()) +
           entry * this->width() * this->height() + loc;
}

template<Device D>
void RegionLayer<D>::forward(const V& input)
{
    Layer<D>::forward(input);

    forwardCpu(input, this->output_);
}

template<Device D>
void RegionLayer<D>::forwardCpu(const PxCpuVector& input, PxCpuVector& output)
{
    output.copy(input);

    auto size = coords_ + 1 + this->classes();
    auto* poutput = output.data();

    flatten(poutput, this->width() * this->height(), size * num_, this->batch(), 1);

    for (auto b = 0; b < this->batch(); ++b) {
        for (auto i = 0; i < this->height() * this->width() * num_; ++i) {
            auto index = i * size + b * this->outputs();
            //poutput[index + coords_] = logistic_.apply(poutput[index + coords_]);
            logistic_.apply(&poutput[index + coords_], &poutput[index + coords_ + 1]);

            if (softmax_) {
                softmax(poutput + index + coords_ + 1, this->classes(), 1, poutput + index + coords_ + 1, 1);
            }
        }
    }

    if (this->inferring()) {
        return;
    }
}

template<Device D>
void RegionLayer<D>::backward(const V& input)
{
}

template<Device D>
void RegionLayer<D>::update()
{
}

template<Device D>
void RegionLayer<D>::addDetects(Detections& detects, float threshold)
{
    addDetects(detects, threshold, this->output_.data());
}

template<Device D>
void RegionLayer<D>::addDetects(Detections& detects, int width, int height, float threshold)
{
    addDetects(detects, width, height, threshold, this->output_.data());
}

template<Device D>
void RegionLayer<D>::addDetects(Detections& detections, int width, int height, float threshold,
                                const float* predictions) const
{
    for (auto i = 0; i < this->height() * this->width(); ++i) {
        auto row = i / this->width();
        auto col = i % this->width();
        for (auto n = 0; n < num_; ++n) {
            auto index = i * num_ + n;
            auto pindex = index * (coords_ + 1 + this->classes()) + coords_;
            auto scale = predictions[pindex];
            auto bindex = index * (coords_ + 1 + this->classes());
            auto box = regionBox(predictions, n, bindex, col, row);

            auto clsIndex = index * (coords_ + 1 + this->classes()) + coords_ + 1;

            for (auto j = 0; j < this->classes(); ++j) {
                auto prob = scale * predictions[clsIndex + j];
                if (prob >= threshold) {
                    auto rect = lightBox(box, { width, height });
                    Detection det(rect, j, prob);
                    detections.emplace_back(std::move(det));
                }
            }
        }
    }
}

template<Device D>
DarkBox RegionLayer<D>::regionBox(const float* p, int n, int index, int i, int j) const
{
    auto* biases = biases_.data();

    /*auto x = (i + logistic_.apply(p[index + 0])) / this->width();
    auto y = (j + logistic_.apply(p[index + 1])) / this->height();*/

    auto x = (i + p[index + 0]) / this->width();
    auto y = (j + p[index + 1]) / this->height();


    auto w = std::exp(p[index + 2]) * biases[2 * n] / this->width();
    auto h = std::exp(p[index + 3]) * biases[2 * n + 1] / this->height();

    return { x, y, w, h };
}

template<Device D>
void RegionLayer<D>::addDetects(Detections& detections, float threshold, const float* predictions) const
{
    // TODO: implement
}

using CpuRegion = RegionLayer<>;
using CudaRegion = RegionLayer<Device::CUDA>;

} // px

#ifdef USE_CUDA

#include "cuda/RegionLayer.h"

#endif
